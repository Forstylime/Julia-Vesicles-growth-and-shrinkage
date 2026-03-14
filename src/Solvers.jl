# File: src/Solvers.jl

using LinearMaps
using LinearAlgebra
#using IterativeSolvers
using Krylov

"""
Step 1: 求解分裂后的线性方程组组件
"""
# src/Solvers.jl
# solve_step1 签名增加 cache 参数

function solve_step1(present::FieldState, old::FieldState,
                     ops::Operators, conf::Config, bdf::BDFCoeff,
                     cache::Step1Cache)        # ← 新增参数
    dt = conf.dt
    a, b, c = bdf.a, bdf.b, bdf.c
    A0 = present.A0

    phi_star     = @. 2.0 * present.phi     - old.phi
    psi_star     = @. 2.0 * present.psi     - old.psi
    phi_star_hat = @. 2.0 * present.phi_hat - old.phi_hat
    psi_star_hat = @. 2.0 * present.psi_hat - old.psi_hat

    M_psi = @. get_M_psi(phi_star, conf)
    M_max = maximum(M_psi)
    L_phi   = @. conf.S1 * ops.Biharmonic - conf.S2 * ops.Laplacian + conf.S3
    L_psi   = conf.S4
    lhs_phi = @. (a / dt) + conf.M_phi * L_phi

    # BiCGSTAB 内核（与上版相同，零分配）
    D1f = ops.D1_full
    pfull, ifull = ops.fft_plan_full, ops.ifft_plan_full
    b1, b2, b3, b4, b5 = ops.buf_mv1, ops.buf_mv2, ops.buf_mv3,
                          ops.buf_mv4, ops.buf_mv5
    a_dt = a / dt

    function A_psi_func!(y_vec, x_vec)
        x_hat = reshape(x_vec, conf.Nx, conf.Ny, conf.N)
        y_hat = reshape(y_vec, conf.Nx, conf.Ny, conf.N)
        @. b1 = L_psi * x_hat
        @. b2 = D1f[1] * b1
        @. b3 = D1f[2] * b1
        mul!(b4, ifull, b2)
        mul!(b5, ifull, b3)
        @. b4 = M_psi * b4
        @. b5 = M_psi * b5
        mul!(b2, pfull, b4)
        mul!(b3, pfull, b5)
        @. y_hat = a_dt * x_hat - (D1f[1] * b2 + D1f[2] * b3)
        return y_vec
    end

    full_len   = conf.Nx * conf.Ny * conf.N
    A_psi      = LinearMap{ComplexF64}(A_psi_func!, full_len; ismutating=true)
    P_inv_diag = @. 1.0 / (a_dt - M_max * ops.Laplacian_full * L_psi)

    function prec_func!(y_vec, r_vec)
        r_hat = reshape(r_vec, conf.Nx, conf.Ny, conf.N)
        y_hat = reshape(y_vec, conf.Nx, conf.Ny, conf.N)
        @. y_hat = r_hat * P_inv_diag
        return y_vec
    end
    P_inv = LinearMap{ComplexF64}(prec_func!, full_len; ismutating=true)

    # solve_psi：将结果写入目标缓冲区 dest
    function solve_psi!(dest::Array{ComplexF64, 3},
                        rhs_rhat::Array{ComplexF64, 3})
        rhs_phys = irfft(rhs_rhat, conf.Nx, (1, 2))
        rhs_full = fft(rhs_phys, (1, 2))
        rhs_vec  = reshape(rhs_full, full_len)
        sol_vec, stats = bicgstab(A_psi, rhs_vec;
                                  M=P_inv, atol=1e-14,
                                  rtol=conf.tol, itmax=20, history=true)
        !stats.solved && @warn "BiCGSTAB 未收敛: $(stats.resnorm)"
        sol_full = reshape(sol_vec, conf.Nx, conf.Ny, conf.N)
        # 原地截取到 dest（半谱）
        dest .= sol_full[1:(conf.Nx ÷ 2 + 1), :, :]
        return dest
    end

    # ── 分步求解：全部改为 .= 原地写入 ─────────────────────────────

    # 11
    @. ops.buf_rhat1         = -(b * present.phi_hat + c * old.phi_hat) / dt
    @. cache.phi_11          = ops.buf_rhat1 / lhs_phi
    @. cache.mu_11           = L_phi * cache.phi_11

    @. ops.buf_rhat1         = -(b * present.psi_hat + c * old.psi_hat) / dt
    solve_psi!(cache.psi_11, ops.buf_rhat1)
    @. cache.nu_11           = L_psi * cache.psi_11

    # 12
    H1_star_hat = to_spectral!(ops.temp_comp1, get_H1(phi_star, ops, conf, A0), ops)
    @. cache.phi_12          = (-conf.M_phi * H1_star_hat) / lhs_phi
    @. cache.mu_12           = L_phi * cache.phi_12 + H1_star_hat

    G_star_hat  = to_spectral!(ops.temp_comp1, get_MG(phi_star, psi_star, conf), ops)
    grad_G_x    = to_physical!(ops.temp_real1, ops.D1[1] .* G_star_hat, ops)
    grad_G_y    = to_physical!(ops.temp_real2, ops.D1[2] .* G_star_hat, ops)
    # ✅ 修复写法：拆分为两步，先计算谱变换，再广播乘法
    to_spectral!(ops.temp_comp2, M_psi .* grad_G_x, ops)  # 原地写入 temp_comp2
    to_spectral!(ops.temp_comp3, M_psi .* grad_G_y, ops)  # 原地写入 temp_comp3
    @. ops.buf_rhat1 = ops.D1[1] * ops.temp_comp2 + ops.D1[2] * ops.temp_comp3
    solve_psi!(cache.psi_12, ops.buf_rhat1)
    @. cache.nu_12           = L_psi * cache.psi_12 + G_star_hat

    # 13 & 14
    H2_star_hat = to_spectral!(ops.temp_comp2, get_H2(phi_star, ops, conf), ops)
    H3_star_hat = to_spectral!(ops.temp_comp3, get_H3(phi_star, psi_star, conf), ops)
    @. cache.phi_13          = (-conf.M_phi * H2_star_hat) / lhs_phi
    @. cache.phi_14          = (-conf.M_phi * H3_star_hat) / lhs_phi
    @. cache.mu_13           = L_phi * cache.phi_13 + H2_star_hat
    @. cache.mu_14           = L_phi * cache.phi_14 + H3_star_hat

    # 21
    u_star          = @. 2.0 * present.u - old.u
    grad_phi_star_x = to_physical!(ops.temp_real1, ops.D1[1] .* phi_star_hat, ops)
    grad_phi_star_y = to_physical!(ops.temp_real2, ops.D1[2] .* phi_star_hat, ops)
    @. ops.temp_real3        = u_star[:,:,1] * grad_phi_star_x +
                                u_star[:,:,2] * grad_phi_star_y
    to_spectral!(ops.temp_comp1, ops.temp_real3, ops)
    @. cache.phi_21 = -ops.temp_comp1 / lhs_phi
    @. cache.mu_21  = L_phi * cache.phi_21

    grad_psi_star_x = to_physical!(ops.temp_real1, ops.D1[1] .* psi_star_hat, ops)
    grad_psi_star_y = to_physical!(ops.temp_real2, ops.D1[2] .* psi_star_hat, ops)
    @. ops.temp_real3        = u_star[:,:,1] * grad_psi_star_x +
                                u_star[:,:,2] * grad_psi_star_y
    to_spectral!(ops.temp_comp1, ops.temp_real3, ops)
    @. ops.buf_rhat1 = -ops.temp_comp1
    solve_psi!(cache.psi_21, ops.buf_rhat1)
    @. cache.nu_21 = L_psi * cache.psi_21

    return cache   # ← 返回 cache 而非 Dict
end

# File: src/Solvers.jl

"""
Step 2: 求解分裂后的辅助变量 R1, R2, R3
"""
function solve_step2(present::FieldState, old::FieldState,
                     ops::Operators, conf::Config,
                     cache::Step1Cache, bdf::BDFCoeff)  # ← cache 替换 step1_res::Dict    dx, dy = conf.dx, conf.dy
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
    int_dot(A, B) = dot(A, B) * conf.dx * conf.dy

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
    # 原：phi_12 = real(ops.ifft_plan * step1_res[:phi_12_np1_hat])
    phi_12 = real(ops.ifft_plan * cache.phi_12)
    phi_13 = real(ops.ifft_plan * cache.phi_13)
    phi_14 = real(ops.ifft_plan * cache.phi_14)
    psi_12 = real(ops.ifft_plan * cache.psi_12)
    phi_11 = real(ops.ifft_plan * cache.phi_11)
    psi_11 = real(ops.ifft_plan * cache.psi_11)
    phi_21 = real(ops.ifft_plan * cache.phi_21)
    psi_21 = real(ops.ifft_plan * cache.psi_21)
    
    # 5. 组装 LHS_1 和 RHS_1 (3x3 系统)
    LHS_1 = zeros(3, 3)
    RHS_1 = zeros(3)

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
# 签名从 Dict 改为 Step1Cache，字段访问从 [:key] 改为 .field
function solve_step3(cache::Step1Cache, step2_res::NamedTuple)
    R11, R21, R31 = step2_res.R_11_np1, step2_res.R_21_np1, step2_res.R_31_np1
    R12, R22, R32 = step2_res.R_12_np1, step2_res.R_22_np1, step2_res.R_32_np1

    # phi_22=phi_12, phi_23=phi_13, phi_24=phi_14（逻辑复用，无需改变）
    phi_1 = @. cache.phi_11 + R11*cache.phi_12 + R21*cache.phi_13 + R31*cache.phi_14
    mu_1  = @. cache.mu_11  + R11*cache.mu_12  + R21*cache.mu_13  + R31*cache.mu_14
    psi_1 = @. cache.psi_11 + R31*cache.psi_12
    nu_1  = @. cache.nu_11  + R31*cache.nu_12

    phi_2 = @. cache.phi_21 + R12*cache.phi_12 + R22*cache.phi_13 + R32*cache.phi_14
    mu_2  = @. cache.mu_21  + R12*cache.mu_12  + R22*cache.mu_13  + R32*cache.mu_14
    psi_2 = @. cache.psi_21 + R32*cache.psi_12
    nu_2  = @. cache.nu_21  + R32*cache.nu_12

    return (phi_1_hat=phi_1, mu_1_hat=mu_1, psi_1_hat=psi_1, nu_1_hat=nu_1,
            phi_2_hat=phi_2, mu_2_hat=mu_2, psi_2_hat=psi_2, nu_2_hat=nu_2)
end

# File: src/Solvers.jl

"""
Step 4: 求解分裂的中间速度场 u_tilde_1 和 u_tilde_2
"""
# solve_step4 签名增加 ops 中已有的缓冲区（buf_uhat1/2, buf_uphys1/2）
function solve_step4(present::FieldState, old::FieldState,
                     ops::Operators, conf::Config, bdf::BDFCoeff)
    dt, eta, lamda = conf.dt, conf.eta, conf.lamda
    a, b, c = bdf.a, bdf.b, bdf.c
    Nx, Ny  = conf.Nx, conf.Ny

    phi_star = @. 2.0 * present.phi - old.phi
    psi_star = @. 2.0 * present.psi - old.psi
    u_star   = @. 2.0 * present.u   - old.u
    u_star_hat = @. 2.0 * present.u_hat - old.u_hat
    mu_star_hat = @. 2.0 * present.mu_hat - old.mu_hat
    nu_star_hat = @. 2.0 * present.nu_hat - old.nu_hat

    lhs = @. (a / dt) - eta * ops.Laplacian

    # ── RHS 1：原地写入 ops.buf_uhat1 ───────────────────────────
    for d in 1:2
        @. ops.buf_uhat1[:,:,d] = -(b * present.u_hat[:,:,d] +
                                     c * old.u_hat[:,:,d]) / dt -
                                    ops.D1[d] * present.p_hat
    end

    # 对流项梯度（复用 temp_comp1/2/3 和 temp_real1/2/3）
    #to_physical!(ops.temp_real1,
    #    ops.D1[1] .* u_star_hat[:,:,1], ops)  # dux_dx — 注意：需要2D plan
    # ⚠️ 注意：u_star_hat[:,:,1] 是2D切片，需要使用 ifft_plan_1
    dux_dx = real(ops.ifft_plan_1 * (ops.D1[1] .* u_star_hat[:,:,1]))
    dux_dy = real(ops.ifft_plan_1 * (ops.D1[2] .* u_star_hat[:,:,1]))
    duy_dx = real(ops.ifft_plan_1 * (ops.D1[1] .* u_star_hat[:,:,2]))
    duy_dy = real(ops.ifft_plan_1 * (ops.D1[2] .* u_star_hat[:,:,2]))

    # 对流项写入 buf_uphys1（复用）
    @. ops.buf_uphys1[:,:,1] = u_star[:,:,1]*dux_dx + u_star[:,:,2]*dux_dy
    @. ops.buf_uphys1[:,:,2] = u_star[:,:,1]*duy_dx + u_star[:,:,2]*duy_dy

    # 耦合项
    dmu_dx = real(ops.ifft_plan * (ops.D1[1] .* mu_star_hat))
    dmu_dy = real(ops.ifft_plan * (ops.D1[2] .* mu_star_hat))
    dnu_dx = real(ops.ifft_plan * (ops.D1[1] .* nu_star_hat))
    dnu_dy = real(ops.ifft_plan * (ops.D1[2] .* nu_star_hat))

    coupl_x = @. phi_star * dmu_dx + psi_star * dnu_dx
    coupl_y = @. phi_star * dmu_dy + psi_star * dnu_dy

    # ── RHS 2：写入 ops.buf_uhat2 ───────────────────────────────
    ops.buf_uhat2[:,:,1] .= ops.fft_plan_1 *
        (-dropdims(sum(ops.buf_uphys1[:,:,1:1], dims=3), dims=3)
         .- lamda .* dropdims(sum(coupl_x, dims=3), dims=3))
    ops.buf_uhat2[:,:,2] .= ops.fft_plan_1 *
        (-dropdims(sum(ops.buf_uphys1[:,:,2:2], dims=3), dims=3)
         .- lamda .* dropdims(sum(coupl_y, dims=3), dims=3))

    # ── 求解（谱空间除法）→ 原地写入 buf_uhat1/2 ────────────────
    for d in 1:2
        @. ops.buf_uhat1[:,:,d] = ops.buf_uhat1[:,:,d] / lhs  # u_tilde_1_hat
        @. ops.buf_uhat2[:,:,d] = ops.buf_uhat2[:,:,d] / lhs  # u_tilde_2_hat
    end

    # ── 转物理空间 → 原地写入 buf_uphys1/2 ──────────────────────
    ops.buf_uphys1 .= real(ops.ifft_plan_2 * ops.buf_uhat1)
    ops.buf_uphys2 .= real(ops.ifft_plan_2 * ops.buf_uhat2)

    # 直接返回 ops 中的 view，不分配新数组
    return (
        u_tilde_1_hat = ops.buf_uhat1,
        u_tilde_2_hat = ops.buf_uhat2,
        u_tilde_1     = ops.buf_uphys1,
        u_tilde_2     = ops.buf_uphys2
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