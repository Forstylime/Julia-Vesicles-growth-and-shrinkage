# File: src/Solvers.jl

using LinearMaps
using LinearAlgebra
#using IterativeSolvers
using Krylov

"""
Step 1: 求解分裂后的线性方程组组件
"""
function solve_step1(present::FieldState, old::FieldState, ops::Operators, conf::Config, bdf::BDFCoeff)
    dt = conf.dt
    a, b, c = bdf.a, bdf.b, bdf.c  # 在独立的STRUCT中存储了 BDF2 系数
    A0 = present.A0  # 预定义的目标面积 A0，作为 get_H1 的输入参数
    
    # 1. 构造外推状态 (Star states)
    phi_star = @. 2.0 * present.phi - old.phi
    psi_star = @. 2.0 * present.psi - old.psi
    phi_star_hat = @. 2.0 * present.phi_hat - old.phi_hat
    psi_star_hat = @. 2.0 * present.psi_hat - old.psi_hat
    
    # 2. 计算变系数 M_psi
    M_psi = @. get_M_psi(phi_star, conf)
    M_max = maximum(M_psi)

    # 3. 预定义算子 (L_phi)
    L_phi = @. conf.S1 * ops.Biharmonic - conf.S2 * ops.Laplacian + conf.S3
    L_psi = conf.S4
    lhs_phi = @. (a / dt) + conf.M_phi * L_phi

    # 4. 定义 BiCGSTAB 求解器
    # y = (a/dt)I - div(M_psi * grad(L_psi * x))
    # note：这里经过测试，发现irfft会导致虚部信息丢失从而降低求解器效率，
    # 因此采用全频域
    # ==========================================
    # 4.1. 内部的全频域求解器 (直接在全复数域构造和求解线性系统)
    # ==========================================
    function solve_psi_full_complex(rhs_full, conf::Config)
        full_len = conf.Nx * conf.Ny * conf.N
        
        # 算子：使用全尺寸 fft 和 ifft
        function A_psi_func_complex!(y_vec, x_vec)
            x_hat = reshape(x_vec, conf.Nx, conf.Ny, conf.N)
            y_hat = reshape(y_vec, conf.Nx, conf.Ny, conf.N)
            
            # L_psi * x_hat
            temp_comp = L_psi .* x_hat 
            
            # 频域梯度
            grad_x_hat = ops.D1_full[1] .* temp_comp
            grad_y_hat = ops.D1_full[2] .* temp_comp
            
            # 转物理空间 (必须用 ifft)
            grad_x_phys = ifft(grad_x_hat, (1, 2))
            grad_y_phys = ifft(grad_y_hat, (1, 2))
            
            # 乘 M_psi
            flux_x_phys = M_psi .* grad_x_phys
            flux_y_phys = M_psi .* grad_y_phys
            
            # 转回频域 (必须用 fft)
            flux_x_hat = fft(flux_x_phys, (1, 2))
            flux_y_hat = fft(flux_y_phys, (1, 2))
            
            # 组装最终结果
            @. y_hat = (a / dt) * x_hat - (ops.D1_full[1] * flux_x_hat + ops.D1_full[2] * flux_y_hat)
            return y_vec
        end

        A_psi = LinearMap{ComplexF64}(A_psi_func_complex!, full_len; ismutating=true)

        # 预处理子 (注意：P_inv_hat 需要是全尺寸 Nx × Ny 的)
        # 如果你外部传进来的 P_inv_hat 是一半大小，你需要在这里重新计算全尺寸的
        P_inv_hat_full = @. 1.0 / ( (a / dt) - M_max * ops.Laplacian_full * L_psi ) 
        
        function prec_func_complex!(y_vec, r_vec)
            r_hat = reshape(r_vec, conf.Nx, conf.Ny, conf.N)
            y_hat = reshape(y_vec, conf.Nx, conf.Ny, conf.N)
            @. y_hat = r_hat * P_inv_hat_full
            return y_vec
        end
        P_inv = LinearMap{ComplexF64}(prec_func_complex!, full_len; ismutating=true)

        # 在全复数域求解
        rhs_vec = reshape(rhs_full, full_len)
        sol_vec, stats = bicgstab(A_psi, rhs_vec; M=P_inv, atol=1e-14, rtol=conf.tol, itmax=20, history=true)
        
        # 检查收敛性, 如果不收敛，输出警告信息
        !stats.solved && @warn "BiCGSTAB did not converge within the maximum iterations. Final residual: $(stats.resnorm)"

        return reshape(sol_vec, conf.Nx, conf.Ny, conf.N)
    end


    # ==========================================
    # 4.2. 外部包装器 (处理 rfft 的桥接)
    # ==========================================
    function solve_psi(rhs_rhat, conf::Config)
        # 步骤 A: 将一半大小的 rhs_rhat "无损补全" 为全尺寸复数 rhs_full
        # 巧妙利用 irfft -> fft，自动生成严格共轭对称的全频谱，省去手动 pad 的烦恼！
        rhs_phys = irfft(rhs_rhat, conf.Nx, (1, 2)) 
        rhs_full = fft(rhs_phys, (1, 2))
        
        # 步骤 B: 调用全复数域求解器，获得全尺寸复数解
        sol_full = solve_psi_full_complex(rhs_full, conf)
        
        # 步骤 C: 截断回一半大小，完美适配外部的 rfft 生态
        # 傅里叶谱的正半轴对应前 (Nx ÷ 2 + 1) 个元素
        sol_rhat = sol_full[1:(conf.Nx ÷ 2 + 1), :, :]
        
        return sol_rhat
    end

    # --- 开始分步求解 (11, 12, 13, 14, 21, ...) ---
    
    # 初始化结果容器 (NamedTuple)
    res = Dict{Symbol, Any}()

    # 11: 历史项
    rhs_phi_11 = @. -(b * present.phi_hat + c * old.phi_hat) / dt
    res[:phi_11_np1_hat] = rhs_phi_11 ./ lhs_phi
    res[:mu_11_np1_hat] = L_phi .* res[:phi_11_np1_hat]

    rhs_psi_11 = @. -(b * present.psi_hat + c * old.psi_hat) / dt
    res[:psi_11_np1_hat] = solve_psi(rhs_psi_11, conf)
    res[:nu_11_np1_hat] = L_psi .* res[:psi_11_np1_hat]

    # 12: H1/G 项
    H1_star_hat = to_spectral!(ops.temp_comp1, get_H1(phi_star, ops, conf, A0), ops)
    rhs_phi_12 = @. -conf.M_phi * H1_star_hat
    res[:phi_12_np1_hat] = rhs_phi_12 ./ lhs_phi
    res[:mu_12_np1_hat]  = L_phi .* res[:phi_12_np1_hat] .+ H1_star_hat
    
    # G 的散度项: div(M_psi * grad(G_star))
    G_star_hat  = to_spectral!(ops.temp_comp1, get_MG(phi_star, psi_star, conf), ops)
    grad_G_x = to_physical!(ops.temp_real1, ops.D1[1] .* G_star_hat, ops)
    grad_G_y = to_physical!(ops.temp_real2, ops.D1[2] .* G_star_hat, ops)
    rhs_psi_12 = ops.D1[1] .* (to_spectral!(ops.temp_comp2, M_psi .* grad_G_x, ops)) .+ 
                 ops.D1[2] .* (to_spectral!(ops.temp_comp3, M_psi .* grad_G_y, ops))
    res[:psi_12_np1_hat] = solve_psi(rhs_psi_12, conf)
    res[:nu_12_np1_hat] = L_psi .* res[:psi_12_np1_hat] .+ G_star_hat

    # 13 & 14 (仅 phi)
    H2_star_hat = to_spectral!(ops.temp_comp2, get_H2(phi_star, ops, conf), ops)
    H3_star_hat = to_spectral!(ops.temp_comp3, get_H3(phi_star, psi_star, conf), ops)
    res[:phi_13_np1_hat] = (@. -conf.M_phi * H2_star_hat) ./ lhs_phi
    res[:phi_14_np1_hat] = (@. -conf.M_phi * H3_star_hat) ./ lhs_phi
    res[:mu_13_np1_hat] = L_phi .* res[:phi_13_np1_hat] .+ H2_star_hat
    res[:mu_14_np1_hat] = L_phi .* res[:phi_14_np1_hat] .+ H3_star_hat

    # 21: 对流项
    # 计算 u_star · grad(phi_star)
    u_star = @. 2.0 * present.u - old.u
    grad_phi_star_x = to_physical!(ops.temp_real1, ops.D1[1] .* phi_star_hat, ops)
    grad_phi_star_y = to_physical!(ops.temp_real2, ops.D1[2] .* phi_star_hat, ops)
    u_grad_phi = @. u_star[:,:,1] * grad_phi_star_x + u_star[:,:,2] * grad_phi_star_y
    res[:phi_21_np1_hat] = (-(to_spectral!(ops.temp_comp1, u_grad_phi, ops))) ./ lhs_phi
    res[:mu_21_np1_hat] = L_phi .* res[:phi_21_np1_hat]

    grad_psi_star_x = to_physical!(ops.temp_real1, ops.D1[1] .* psi_star_hat, ops)
    grad_psi_star_y = to_physical!(ops.temp_real2, ops.D1[2] .* psi_star_hat, ops)
    u_grad_psi = @. u_star[:,:,1] * grad_psi_star_x + u_star[:,:,2] * grad_psi_star_y
    rhs_psi_21 = -(to_spectral!(ops.temp_comp1, u_grad_psi, ops))
    res[:psi_21_np1_hat] = solve_psi(rhs_psi_21, conf)
    res[:nu_21_np1_hat] = L_psi .* res[:psi_21_np1_hat]

    # 逻辑优化：22, 23, 24 与之前的计算结果一致，直接复用
    res[:phi_22_np1_hat] = res[:phi_12_np1_hat]
    res[:psi_22_np1_hat] = res[:psi_12_np1_hat]
    res[:phi_23_np1_hat] = res[:phi_13_np1_hat]
    res[:phi_24_np1_hat] = res[:phi_14_np1_hat]

    # 计算配套的 mu 和 nu (频谱)  
    res[:mu_22_np1_hat] = res[:mu_12_np1_hat]
    res[:nu_22_np1_hat] = res[:nu_12_np1_hat]
    res[:mu_23_np1_hat] = res[:mu_13_np1_hat]
    res[:mu_24_np1_hat] = res[:mu_14_np1_hat]

    return res
end

# File: src/Solvers.jl

"""
Step 2: 求解分裂后的辅助变量 R1, R2, R3
"""
function solve_step2(present::FieldState, old::FieldState, ops::Operators, conf::Config, step1_res::Dict, bdf::BDFCoeff)
    dx, dy = conf.dx, conf.dy
    a, b, c = bdf.a, bdf.b, bdf.c
    
    # 1. 构造外推状态和非线性变分项 (物理空间)
    phi_n = present.phi
    phi_nm1 = old.phi
    phi_star = @. 2.0 * phi_n - phi_nm1
    phi_star_hat = @. 2.0 * present.phi_hat - old.phi_hat
    
    psi_n = present.psi
    psi_nm1 = old.psi
    psi_star = @. 2.0 * psi_n - psi_nm1

    # 获取物理空间的变分导数
    H1_star = get_H1(phi_star, ops, conf, present.A0)
    H2_star = get_H2(phi_star, ops, conf)
    H3_star = get_H3(phi_star, psi_star, conf)
    G_star  = get_MG(phi_star, psi_star, conf)

    # 2. 积分辅助函数: int_dot(A, B) = sum(A .* B) * dx * dy
    # 使用 Julia 内置 dot 更高效
    # 这里的积分是全域的求和，即包含所有囊泡。
    int_dot(A, B) = dot(A, B) * dx * dy

    # 3. 计算中间标量 U1, U2, U3
    # U1 = -(b*R1_n + c*R1_nm1)/a + integral(H1_star * (b*phi_n + c*phi_nm1)) / (2a)
    term_phi_hist = @. b * phi_n + c * phi_nm1
    term_psi_hist = @. b * psi_n + c * psi_nm1
    
    U1 = -(b * present.R1 + c * old.R1) / a + int_dot(H1_star, term_phi_hist) / (2.0 * a)
    U2 = -(b * present.R2 + c * old.R2) / a + int_dot(H2_star, term_phi_hist) / (2.0 * a)
    U3 = -(b * present.R3 + c * old.R3) / a + (int_dot(H3_star, term_phi_hist) + int_dot(G_star, term_psi_hist)) / (2.0 * a)

    # 4. 准备 Step 1 结果的物理空间场
    # 我们需要 phi_12, phi_13, phi_14, phi_22, phi_23, phi_24 以及对应的 psi
    # 注意：为了节省内存，我们在这里即时进行 IFFT
    
    phi_12 = real(ops.ifft_plan * step1_res[:phi_12_np1_hat])
    phi_13 = real(ops.ifft_plan * step1_res[:phi_13_np1_hat])
    phi_14 = real(ops.ifft_plan * step1_res[:phi_14_np1_hat])
    
    psi_12 = real(ops.ifft_plan * step1_res[:psi_12_np1_hat])
    
    # 5. 组装 LHS_1 和 RHS_1 (3x3 系统)
    LHS_1 = zeros(3, 3)
    RHS_1 = zeros(3)

    # 这里的 phi_11 和 psi_11 也需要 IFFT
    phi_11 = real(ops.ifft_plan * step1_res[:phi_11_np1_hat])
    psi_11 = real(ops.ifft_plan * step1_res[:psi_11_np1_hat])

    LHS_1[1,1] = 1.0 - 0.5 * int_dot(H1_star, phi_12)
    LHS_1[1,2] = -0.5 * int_dot(H1_star, phi_13)
    LHS_1[1,3] = -0.5 * int_dot(H1_star, phi_14)

    LHS_1[2,1] = -0.5 * int_dot(H2_star, phi_12)
    LHS_1[2,2] = 1.0 - 0.5 * int_dot(H2_star, phi_13)
    LHS_1[2,3] = -0.5 * int_dot(H2_star, phi_14)

    LHS_1[3,1] = -0.5 * int_dot(H3_star, phi_12)
    LHS_1[3,2] = -0.5 * int_dot(H3_star, phi_13)
    LHS_1[3,3] = 1.0 - 0.5 * (int_dot(H3_star, phi_14) + int_dot(G_star, psi_12))

    RHS_1[1] = 0.5 * int_dot(H1_star, phi_11) + U1
    RHS_1[2] = 0.5 * int_dot(H2_star, phi_11) + U2
    RHS_1[3] = 0.5 * (int_dot(H3_star, phi_11) + int_dot(G_star, psi_11)) + U3

    # 6. 组装 LHS_2 和 RHS_2
    # 逻辑优化：在 Step 1 中我们知道 phi_22=phi_12, phi_23=phi_13, phi_24=phi_14
    # 因此 LHS_2 实际上等于 LHS_1。我们只需计算 RHS_2。
    # LHS_2 = LHS_1 
    # LHS_2 = LHS_1 是浅拷贝（别名），不是独立矩阵，在Julia中，LHS_2 = LHS_1 让两者指向同一个数组。
    # 若后续有任何对 LHS_2 的修改（当前代码虽然没有，但这是隐患），将会意外修改 LHS_1。应改为：
    LHS_2 = copy(LHS_1)

    phi_21 = real(ops.ifft_plan * step1_res[:phi_21_np1_hat])
    psi_21 = real(ops.ifft_plan * step1_res[:psi_21_np1_hat])
    
    RHS_2 = zeros(3)
    RHS_2[1] = 0.5 * int_dot(H1_star, phi_21)
    RHS_2[2] = 0.5 * int_dot(H2_star, phi_21)
    RHS_2[3] = 0.5 * (int_dot(H3_star, phi_21) + int_dot(G_star, psi_21))

    # 7. 求解
    sol1 = LHS_1 \ RHS_1
    sol2 = LHS_2 \ RHS_2

    # 返回结果
    return (
        R_11_np1 = sol1[1], R_21_np1 = sol1[2], R_31_np1 = sol1[3],
        R_12_np1 = sol2[1], R_22_np1 = sol2[2], R_32_np1 = sol2[3]
    )
end

# File: src/Solvers.jl

"""
Step 3: 根据求解出的 SAV 标量 R，线性组合分裂的场变量。
"""
function solve_step3(step1_res::Dict, step2_res::NamedTuple)
    # 提取 Step 2 算出的标量系数
    R11, R21, R31 = step2_res.R_11_np1, step2_res.R_21_np1, step2_res.R_31_np1
    R12, R22, R32 = step2_res.R_12_np1, step2_res.R_22_np1, step2_res.R_32_np1

    # --- 组合 第一组场 (phi_1, mu_1, psi_1, nu_1) ---
    # 使用 @. 确保所有操作都在一个循环内完成
    phi_1_np1_hat = @. step1_res[:phi_11_np1_hat] + 
                       R11 * step1_res[:phi_12_np1_hat] + 
                       R21 * step1_res[:phi_13_np1_hat] + 
                       R31 * step1_res[:phi_14_np1_hat]

    mu_1_np1_hat  = @. step1_res[:mu_11_np1_hat] + 
                       R11 * step1_res[:mu_12_np1_hat] + 
                       R21 * step1_res[:mu_13_np1_hat] + 
                       R31 * step1_res[:mu_14_np1_hat]

    psi_1_np1_hat = @. step1_res[:psi_11_np1_hat] + R31 * step1_res[:psi_12_np1_hat]
    nu_1_np1_hat  = @. step1_res[:nu_11_np1_hat]  + R31 * step1_res[:nu_12_np1_hat]

    # --- 组合 第二组场 (phi_2, mu_2, psi_2, nu_2) ---
    phi_2_np1_hat = @. step1_res[:phi_21_np1_hat] + 
                       R12 * step1_res[:phi_22_np1_hat] + 
                       R22 * step1_res[:phi_23_np1_hat] + 
                       R32 * step1_res[:phi_24_np1_hat]

    mu_2_np1_hat  = @. step1_res[:mu_21_np1_hat] + 
                       R12 * step1_res[:mu_22_np1_hat] + 
                       R22 * step1_res[:mu_23_np1_hat] + 
                       R32 * step1_res[:mu_24_np1_hat]

    psi_2_np1_hat = @. step1_res[:psi_21_np1_hat] + R32 * step1_res[:psi_22_np1_hat]
    nu_2_np1_hat  = @. step1_res[:nu_21_np1_hat]  + R32 * step1_res[:nu_22_np1_hat]

    return (
        phi_1_hat = phi_1_np1_hat, mu_1_hat = mu_1_np1_hat,
        psi_1_hat = psi_1_np1_hat, nu_1_hat = nu_1_np1_hat,
        phi_2_hat = phi_2_np1_hat, mu_2_hat = mu_2_np1_hat,
        psi_2_hat = psi_2_np1_hat, nu_2_hat = nu_2_np1_hat
    )
end

# File: src/Solvers.jl

"""
Step 4: 求解分裂的中间速度场 u_tilde_1 和 u_tilde_2
"""
function solve_step4(present::FieldState, old::FieldState, ops::Operators, conf::Config, bdf::BDFCoeff)
    dt = conf.dt
    eta = conf.eta
    lamda = conf.lamda
    a, b, c = bdf.a, bdf.b, bdf.c
    Nx, Ny = conf.Nx, conf.Ny

    # 1. 构造外推状态 (Star states)
    # 物理空间
    phi_star = @. 2.0 * present.phi - old.phi
    psi_star = @. 2.0 * present.psi - old.psi
    u_star   = @. 2.0 * present.u - old.u
    
    # 谱空间 (用于计算梯度)
    # 注意：mu 和 nu 通常在 Step 3 已经算出 _hat 形式
    # 这里我们直接外推谱空间变量
    u_star_hat = @. 2.0 * present.u_hat - old.u_hat
    
    # 2. 构造左端项算子 (LHS)
    # (a/dt) - eta * Laplacian
    lhs = @. (a / dt) - eta * ops.Laplacian

    # 3. 计算 RHS 1 (历史项 + 压力梯度)
    # rhs_1 = -(b*u_n + c*u_nm1)/dt - grad(p_n)
    rhs_1_hat = zeros(ComplexF64, Nx÷2+1, Ny, 2)
    for d in 1:2
        rhs_1_hat[:, :, d] = -(b * present.u_hat[:, :, d] + c * old.u_hat[:, :, d]) / dt .- 
                                 (ops.D1[d] .* present.p_hat)
    end

    # 4. 计算非线性项 (Convective & Coupling)
    # 我们需要先算出梯度项的物理空间表示
    
    # --- 对流项: (u* · grad) u* = (u_x .* dx + u_y .* dy) .* [u_x, u_y] ---
    # grad_ux = [dux/dx, dux/dy], grad_uy = [duy/dx, duy/dy]
    dux_dx = real(ops.ifft_plan_1 * (ops.D1[1] .* u_star_hat[:, :, 1]))
    dux_dy = real(ops.ifft_plan_1 * (ops.D1[2] .* u_star_hat[:, :, 1]))
    duy_dx = real(ops.ifft_plan_1 * (ops.D1[1] .* u_star_hat[:, :, 2]))
    duy_dy = real(ops.ifft_plan_1 * (ops.D1[2] .* u_star_hat[:, :, 2]))

    conv_x = @. u_star[:, :, 1] * dux_dx + u_star[:, :, 2] * dux_dy
    conv_y = @. u_star[:, :, 1] * duy_dx + u_star[:, :, 2] * duy_dy

    # --- 相场耦合项: phi* * grad(mu*) + psi* * grad(nu*) ---
    # 首先外推化学势的谱空间
    mu_star_hat = @. 2.0 * present.mu_hat - old.mu_hat
    nu_star_hat = @. 2.0 * present.nu_hat - old.nu_hat
    
    dmu_dx = real(ops.ifft_plan * (ops.D1[1] .* mu_star_hat))
    dmu_dy = real(ops.ifft_plan * (ops.D1[2] .* mu_star_hat))
    dnu_dx = real(ops.ifft_plan * (ops.D1[1] .* nu_star_hat))
    dnu_dy = real(ops.ifft_plan * (ops.D1[2] .* nu_star_hat))

    coupl_x = @. phi_star * dmu_dx + psi_star * dnu_dx
    coupl_y = @. phi_star * dmu_dy + psi_star * dnu_dy

    # 5. 构造 RHS 2 (谱空间)
    rhs_2_hat = zeros(ComplexF64, Nx÷2+1, Ny, 2)
    # 注意：fft_plan_1 是针对二维矩阵数据的
    rhs_2_hat[:, :, 1] = ops.fft_plan_1 * (-conv_x .- lamda .* dropdims(sum(coupl_x, dims=3), dims=3))
    rhs_2_hat[:, :, 2] = ops.fft_plan_1 * (-conv_y .- lamda .* dropdims(sum(coupl_y, dims=3), dims=3))

    # 6. 求解 (谱空间除法) 并转回物理空间
    u_tilde_1_hat = zeros(ComplexF64, Nx÷2+1, Ny, 2)
    u_tilde_2_hat = zeros(ComplexF64, Nx÷2+1, Ny, 2)

    for d in 1:2
        @. u_tilde_1_hat[:, :, d] = rhs_1_hat[:, :, d] / lhs
        @. u_tilde_2_hat[:, :, d] = rhs_2_hat[:, :, d] / lhs
    end

    # 转回物理空间 (用于后续步骤)
    u_tilde_1 = zeros(Float64, Nx, Ny, 2)
    u_tilde_2 = zeros(Float64, Nx, Ny, 2)
    
    u_tilde_1 .= real(ops.ifft_plan_2 * u_tilde_1_hat)
    u_tilde_2 .= real(ops.ifft_plan_2 * u_tilde_2_hat)

    return (
        u_tilde_1_hat = u_tilde_1_hat,
        u_tilde_2_hat = u_tilde_2_hat,
        u_tilde_1 = u_tilde_1,
        u_tilde_2 = u_tilde_2
    )
end

# File: src/Solvers.jl

"""
Step 5: 求解标量 Q^{n+1}
"""
function solve_step5(present::FieldState, old::FieldState, ops::Operators, conf::Config, step3_res::NamedTuple, step4_res::NamedTuple, bdf::BDFCoeff)
    dt, dx, dy = conf.dt, conf.dx, conf.dy
    a, b, c = bdf.a, bdf.b, bdf.c
    lamda = conf.lamda

    # 1. 构造外推状态的频谱 (Star states)
    phi_star_hat = @. 2.0 * present.phi_hat - old.phi_hat
    psi_star_hat = @. 2.0 * present.psi_hat - old.psi_hat
    mu_star_hat  = @. 2.0 * present.mu_hat - old.mu_hat
    nu_star_hat  = @. 2.0 * present.nu_hat - old.nu_hat
    u_star_hat   = @. 2.0 * present.u_hat - old.u_hat
    
    # 物理空间外推速度
    u_star = @. 2.0 * present.u - old.u

    # 2. 准备物理空间的梯度项 (用于对流和耦合计算)
    grad_phi_star_x = real(ops.ifft_plan * (ops.D1[1] .* phi_star_hat))
    grad_phi_star_y = real(ops.ifft_plan * (ops.D1[2] .* phi_star_hat))
    
    grad_psi_star_x = real(ops.ifft_plan * (ops.D1[1] .* psi_star_hat))
    grad_psi_star_y = real(ops.ifft_plan * (ops.D1[2] .* psi_star_hat))
    
    grad_mu_star_x = real(ops.ifft_plan * (ops.D1[1] .* mu_star_hat))
    grad_mu_star_y = real(ops.ifft_plan * (ops.D1[2] .* mu_star_hat))
    
    grad_nu_star_x = real(ops.ifft_plan * (ops.D1[1] .* nu_star_hat))
    grad_nu_star_y = real(ops.ifft_plan * (ops.D1[2] .* nu_star_hat))

    # 3. 计算对流项: (u* · grad) u*
    # u_star_hat 的梯度
    dux_dx = real(ops.ifft_plan_1 * (ops.D1[1] .* u_star_hat[:,:,1]))
    dux_dy = real(ops.ifft_plan_1 * (ops.D1[2] .* u_star_hat[:,:,1]))
    duy_dx = real(ops.ifft_plan_1 * (ops.D1[1] .* u_star_hat[:,:,2]))
    duy_dy = real(ops.ifft_plan_1 * (ops.D1[2] .* u_star_hat[:,:,2]))
    
    conv_x = @. u_star[:,:,1] * dux_dx + u_star[:,:,2] * dux_dy
    conv_y = @. u_star[:,:,1] * duy_dx + u_star[:,:,2] * duy_dy

    # 4. 准备 Step 3 和 Step 4 的物理空间变量
    mu_1 = real(ops.ifft_plan * step3_res.mu_1_hat)
    nu_1 = real(ops.ifft_plan * step3_res.nu_1_hat)
    mu_2 = real(ops.ifft_plan * step3_res.mu_2_hat)
    nu_2 = real(ops.ifft_plan * step3_res.nu_2_hat)
    
    u_tilde_1 = step4_res.u_tilde_1
    u_tilde_2 = step4_res.u_tilde_2

    # 5. 计算 theta_1 (积分项)
    # theta_11 = (u* · grad phi*) * mu_1
    t11 = dot(@.(u_star[:,:,1] * grad_phi_star_x + u_star[:,:,2] * grad_phi_star_y), mu_1)
    # theta_12 = (u* · grad psi*) * nu_1
    t12 = dot(@.(u_star[:,:,1] * grad_psi_star_x + u_star[:,:,2] * grad_psi_star_y), nu_1)
    # theta_13 = ∑_(phi·grad mu + psi·grad nu, dim=3) · u_tilde_1
    phi_star = @. 2.0 * present.phi - old.phi
    psi_star = @. 2.0 * present.psi - old.psi 
    t3_sub_x = phi_star .* grad_mu_star_x + psi_star .* grad_nu_star_x
    t3_sub_y = phi_star .* grad_mu_star_y + psi_star .* grad_nu_star_y
    t13 = dot(sum(t3_sub_x, dims=3), u_tilde_1[:,:,1]) + dot(sum(t3_sub_y, dims=3), u_tilde_1[:,:,2])

    # theta_14 = conv · u_tilde_1
    t14 = dot(conv_x, u_tilde_1[:,:,1]) + dot(conv_y, u_tilde_1[:,:,2])

    # t11,t12,t13是张量求和形成的标量，t14是矩阵计算得出的标量，直接求和在积分会因为t14的自动广播而被重复计算，应该分别计算积分
    theta_1 = dx * dy * (lamda * (t11 + t12 + t13) + t14)

    # 6. 计算 theta_2 (分量与 theta_1 对应，但使用 index '2')
    t21 = dot(@.(u_star[:,:,1] * grad_phi_star_x + u_star[:,:,2] * grad_phi_star_y), mu_2)
    t22 = dot(@.(u_star[:,:,1] * grad_psi_star_x + u_star[:,:,2] * grad_psi_star_y), nu_2)
    t23 = dot(sum(t3_sub_x, dims=3), u_tilde_2[:,:,1]) + dot(sum(t3_sub_y, dims=3), u_tilde_2[:,:,2])
    t24 = dot(conv_x, u_tilde_2[:,:,1]) + dot(conv_y, u_tilde_2[:,:,2])

    theta_2 = dx * dy * (lamda * (t21 + t22 + t23) + t24)

    # 7. 求解 Q^{n+1} (标量方程)
    rhs_Q = -(b * present.Q + c * old.Q) / dt + theta_1
    lhs_Q = (a / dt) - theta_2 # 理论上，`theta_2`应总是大于零，此标量方程总是有唯一解
    Q_np1 = rhs_Q / lhs_Q

    return Q_np1
end

# File: src/Solvers.jl

"""
Step 6: 合并 Q 分裂项
根据算出的标量 Q^{n+1}，将分裂的物理场进行最终合并。
"""
function solve_step6(step2_res::NamedTuple, step3_res::NamedTuple, step4_res::NamedTuple, Q_np1::Float64)
    # 线性组合： Result = Part1 + Q * Part2
    phi_np1_hat = @. step3_res.phi_1_hat + Q_np1 * step3_res.phi_2_hat
    psi_np1_hat = @. step3_res.psi_1_hat + Q_np1 * step3_res.psi_2_hat
    
    mu_np1_hat  = @. step3_res.mu_1_hat  + Q_np1 * step3_res.mu_2_hat
    nu_np1_hat  = @. step3_res.nu_1_hat  + Q_np1 * step3_res.nu_2_hat
    
    # SAV 标量 R 的合并
    R1_np1 = step2_res.R_11_np1 + Q_np1 * step2_res.R_12_np1
    R2_np1 = step2_res.R_21_np1 + Q_np1 * step2_res.R_22_np1
    R3_np1 = step2_res.R_31_np1 + Q_np1 * step2_res.R_32_np1
    
    # 中间速度场合并
    u_tilde_hat = @. step4_res.u_tilde_1_hat + Q_np1 * step4_res.u_tilde_2_hat

    return (
        phi_hat = phi_np1_hat, psi_hat = psi_np1_hat,
        mu_hat  = mu_np1_hat,  nu_hat  = nu_np1_hat,
        R1 = R1_np1, R2 = R2_np1, R3 = R3_np1,
        u_tilde_hat = u_tilde_hat
    )
end

"""
Step 7: 压力修正 (旋转增量投影法)
确保速度场散度为零，并更新压力场。
"""
function solve_step7(present::FieldState, ops::Operators, conf::Config, step6_res::NamedTuple, bdf::BDFCoeff)
    a, dt, eta = bdf.a, conf.dt, conf.eta
    Nx, Ny = conf.Nx, conf.Ny

    # 1. 计算中间速度的散度 (谱空间)
    # div(u_tilde) = i*kx*ux_hat + i*ky*uy_hat
    div_u_tilde_hat = @. ops.D1[1] * step6_res.u_tilde_hat[:, :, 1] + 
                         ops.D1[2] * step6_res.u_tilde_hat[:, :, 2]

    # 2. 求解压力 Poisson 方程 (谱空间直接除法)
    # Equation: Δ (p_std_hat - p_n_hat) = (a/dt) * div(u_tilde)_hat
    
    # 处理 Laplacian 在频率 (0,0) 处的奇异性 (Laplacian[1,1] = 0)
    delta_p_hat = Matrix{ComplexF64}(undef, Nx÷2+1, Ny) 

    # 提取常量因子，避免在循环中重复计算除法
    factor = a / dt

    # 使用 @. 宏进行纯原地的广播计算（零分配、自动启用 SIMD）
    @. delta_p_hat = factor * div_u_tilde_hat / ops.Laplacian

    # 事后单独修复零频率处的奇异性
    delta_p_hat[1, 1] = 0.0im

    # 3. 速度修正
    # u^{n+1}_hat = u_tilde_hat - (dt/a) * grad(p_std - p_n)
    u_np1_hat = zeros(ComplexF64, Nx÷2+1, Ny, 2)
    for d in 1:2
        @. u_np1_hat[:, :, d] = step6_res.u_tilde_hat[:, :, d] - (dt / a) * ops.D1[d] * delta_p_hat
    end

    # 4. 压力更新 (旋转修正)
    # p^{n+1}_hat = p_std_hat - eta * div(u_tilde)_hat
    p_np1_hat = @. present.p_hat + delta_p_hat - eta * div_u_tilde_hat

    return (
        u_hat = u_np1_hat,
        p_hat = p_np1_hat
    )
end