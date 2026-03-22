# File: src/Utils.jl

# ----------------------------------------------------------------
# 基础标量函数（用 @inline 确保被调用处内联，无函数调用开销）
# ----------------------------------------------------------------

"双井势 g(φ) = ¼(φ²-1)²，执行的是点乘，对Array友好"
@inline g(phi) = @. 0.25 * (phi^2 - 1.0)^2

"g'(φ) = φ(φ²-1)，，执行的是点乘，对Array友好"
@inline g_prime(phi) = @. phi * (phi^2 - 1.0)

"插值函数 p(φ) = -½φ³ + 3/2·φ，，执行的是点乘，对Array友好"
@inline p_phi(phi) = @. -0.5 * phi^3 + 1.5 * phi

"p'(φ) = -3/2·φ² + 3/2，，执行的是点乘，对Array友好"
@inline p_phi_prime(phi) = @. -1.5 * phi^2 + 1.5

"可变扩散系数 M_ψ，，执行的是点乘，对Array友好"
@inline get_M_psi(phi, conf::Config) = @. 0.1 * (1.0 - conf.M0_psi * (phi^2 - 1.0)^2)

# ----------------------------------------------------------------
# 积分辅助（dx*dy 梯形近似）
# ----------------------------------------------------------------
"普通积分"
@inline integrate(f, conf::Config) = sum(f) * conf.dx * conf.dy * conf.dz

"平方积分，适用于涉及范数||x||^2的计算"
@inline integrate_sq(f, conf::Config) = sum(abs2, f) * conf.dx * conf.dy * conf.dz

# ----------------------------------------------------------------
# 能量密度函数, 执行点乘，返回和ϕ，ψ同结构的数组
# ----------------------------------------------------------------

"表面能密度 f_surf"
function f_surf(phi::Array{Float64,4}, ops::Operators, conf::Config)
    phi_hat = to_spectral!(ops.temp_comp1, phi, ops)
    gradx_phi = to_physical!(ops.temp_real1, ops.D1[1] .* phi_hat, ops)
    grady_phi = to_physical!(ops.temp_real2, ops.D1[2] .* phi_hat, ops)
    ε = conf.epsilon
    return @. (3 * sqrt(2) / 4) * (g(phi) / ε) + ε * (gradx_phi^2 + grady_phi^2) / 2
end

"弯曲能密度 f_bend"
function f_bend(phi, ops::Operators, conf::Config)
    phi_hat = to_spectral!(ops.temp_comp1, phi, ops)
    ε = conf.epsilon
    laplacian_phi = to_physical!(ops.temp_real1, ops.Laplacian .* phi_hat, ops)
    return @. (3 * sqrt(2) / (16ε)) * (g_prime(phi) / ε - ε * laplacian_phi)^2
end

"渗透能密度 f_osmotic"
function f_osmotic(phi, psi, conf::Config)
    P = p_phi(phi)
    psi_in = reshape(conf.psi_in_v, 1, 1, 1, :)
    psi_out = reshape(conf.psi_out_v, 1, 1, 1, :)
    f_in = @. 0.5 * conf.gamma_in * (psi - psi_in)^2 + conf.beta_in
    f_out = @. 0.5 * conf.gamma_out * (psi - psi_out)^2 + conf.beta_out
    return @. ((1 + P) / 2) * f_in + ((1 - P) / 2) * f_out
end

# ----------------------------------------------------------------
# 面积计算
# ----------------------------------------------------------------

function calculate_area(phi::Array{Float64,4}, ops::Operators, conf::Config)
    surf = f_surf(phi, ops, conf) # 这里的 surf 是一个4维数组，第4维是囊泡索引
    area = zeros(Float64, conf.N)
    for n in 1:conf.N
        area[n] = integrate(surf[:, :, :, n], conf) # 对每个囊泡的 surf 切片进行积分，得到对应的面积
    end
    return area # 返回一个长度为 N 的向量，每个元素是对应囊泡的面积
end

# ----------------------------------------------------------------
# 原始物理能量
# ----------------------------------------------------------------

function compute_original_energy(state::FieldState, ops::Operators, conf::Config)
    kinetic = 0.5 * integrate_sq(state.u, conf)

    F_s = conf.gamma_surf * integrate(f_surf(state.phi, ops, conf), conf)
    F_b = conf.gamma_bend * integrate(f_bend(state.phi, ops, conf), conf)
    F_o = integrate(f_osmotic(state.phi, state.psi, conf), conf)

    A_sperse = calculate_area(state.phi, ops, conf)  # 一次搞定所有囊泡
    F_a = sum(0.5 .* conf.gamma_area .* (A_sperse .- state.A0) .^ 2)

    return kinetic + conf.lambda * (F_s + F_b + F_o + F_a)
end

# ----------------------------------------------------------------
# SAV 修改能量
# ----------------------------------------------------------------

function compute_modified_energy(present::FieldState, old::FieldState,
    ops::Operators, conf::Config)
    # 1. 动能：两个时间层的平均
    kinetic = 0.25 * integrate_sq(present.u, conf) +
              0.25 * integrate_sq(@.(2 * present.u - old.u), conf)

    # 2. 压力伪能量
    grad_p_x = ops.ifft_plan_1 * (ops.D1[1] .* present.p_hat)
    grad_p_y = ops.ifft_plan_1 * (ops.D1[2] .* present.p_hat)
    pressure = (present.dt^2 / 3.0) * (integrate_sq(grad_p_x, conf) +
                                       integrate_sq(grad_p_y, conf))

    # 3. SAV 标量能量 和 Q 标量能量
    scale_energy(r_new, r_old) = 0.5 * r_new^2 + 0.5 * (2r_new - r_old)^2
    R_energy = scale_energy(present.R1, old.R1) +
               scale_energy(present.R2, old.R2) +
               scale_energy(present.R3, old.R3)
    Q_energy = 0.5 * scale_energy(present.Q, old.Q)

    # 4. 线性算子稳定项
    phi_star = @. 2 * present.phi - old.phi
    psi_star = @. 2 * present.psi - old.psi
    phi_star_hat = @. 2 * present.phi_hat - old.phi_hat

    lap_new = ops.ifft_plan * (ops.Laplacian .* present.phi_hat)
    lap_star = ops.ifft_plan * (ops.Laplacian .* phi_star_hat)

    gx_new = ops.ifft_plan * (ops.D1[1] .* present.phi_hat)
    gy_new = ops.ifft_plan * (ops.D1[2] .* present.phi_hat)
    gx_star = ops.ifft_plan * (ops.D1[1] .* phi_star_hat)
    gy_star = ops.ifft_plan * (ops.D1[2] .* phi_star_hat)

    S1_term = 0.25 * conf.S1 * (integrate_sq(lap_new, conf) +
                                integrate_sq(lap_star, conf))
    S2_term = 0.25 * conf.S2 * (integrate_sq(gx_new, conf) + integrate_sq(gy_new, conf) +
                                integrate_sq(gx_star, conf) + integrate_sq(gy_star, conf))
    S3_term = 0.25 * conf.S3 * (integrate_sq(present.phi, conf) +
                                integrate_sq(phi_star, conf))
    S4_term = 0.25 * conf.S4 * (integrate_sq(present.psi, conf) +
                                integrate_sq(psi_star, conf))

    return (kinetic + pressure +
            conf.lamda * R_energy + Q_energy +
            conf.lamda * (S1_term + S2_term + S3_term + S4_term) -
            conf.lamda * (conf.C1 + conf.C2 + conf.C3) - 0.5)
end

# ----------------------------------------------------------------
# 变分中间量，结构与 phi, psi 相同，第4维是对应囊泡索引
# ----------------------------------------------------------------

"ω = (φ³-φ)/ε - ε·Δφ"
function get_omega(phi, ops::Operators, conf::Config)
    phi_hat = to_spectral!(ops.temp_comp1, phi, ops)
    ε = conf.epsilon
    lap_phi = to_physical!(ops.temp_real1, ops.Laplacian .* phi_hat, ops)
    return @. (phi^3 - phi) / ε - ε * lap_phi
end

# ----------------------------------------------------------------
# SAV 根号内能量 W1, W2, W3，这里计算的都是各个囊泡的能量之和
# ----------------------------------------------------------------

function get_W1(phi, ops::Operators, conf::Config, A0::Vector{Float64})
    A_sperse = calculate_area(phi, ops, conf)
    W1 = sum(0.5 .* conf.gamma_area .* (A_sperse .- A0) .^ 2)
    W1 + conf.C1 < 0 && error("FATAL: W1 + C1 < 0，C1 偏小！W1 = $W1")
    return W1
end

function get_W2(phi, ops::Operators, conf::Config)
    phi_hat = to_spectral!(ops.temp_comp1, phi, ops)
    F_surf_val = conf.gamma_surf * integrate(f_surf(phi, ops, conf), conf)
    F_bend_val = conf.gamma_bend * integrate(f_bend(phi, ops, conf), conf)

    lap_phi = to_physical!(ops.temp_real1, ops.Laplacian .* phi_hat, ops)
    grad_x = to_physical!(ops.temp_real2, ops.D1[1] .* phi_hat, ops)
    grad_y = to_physical!(ops.temp_real3, ops.D1[2] .* phi_hat, ops)

    E2 = integrate(@.(0.5 * conf.S1 * lap_phi^2 +
                      0.5 * conf.S2 * (grad_x^2 + grad_y^2) +
                      0.5 * conf.S3 * phi^2), conf)

    W2 = F_surf_val + F_bend_val - E2
    W2 + conf.C2 < 0 && error("FATAL: W2 + C2 < 0，C2 偏小！W2 = $W2")
    return W2
end

function get_W3(phi, psi, conf::Config)
    W_osm = integrate(f_osmotic(phi, psi, conf), conf)
    E3 = 0.5 * conf.S4 * integrate_sq(psi, conf)
    W3 = W_osm - E3
    W3 + conf.C3 < 0 && error("FATAL: W3 + C3 < 0，C3 偏小！W3 = $W3")
    return W3
end

# ----------------------------------------------------------------
# 变分导数 H1, H2, H3, MG，这里分别计算各个囊泡对应的变分导数，不合并，
# 因此返回的数组结构与 phi, psi 相同。这里H_i的第4维也是变分导数索引，和囊泡索引对应。
# 输入和返回都是实空间的
# ----------------------------------------------------------------

function get_H1(phi::Array{Float64,4}, ops::Operators, conf::Config, A0::Vector{Float64})
    W1 = get_W1(phi, ops, conf, A0) # 这里的 W1 是所有囊泡的第一能量之和
    R1 = sqrt(W1 + conf.C1)
    A_sperse = calculate_area(phi, ops, conf)
    omega = get_omega(phi, ops, conf)
    factor = @. conf.gamma_area * (A_sperse - A0) * (3 * sqrt(2) / 4)
    factor_view = reshape(factor, 1, 1, 1, conf.N) # 将 factor 从 (N,) 变为 (1, 1, 1, N)，以便与 omega 点乘广播
    return @. (factor_view * omega) / R1
end

function get_H2(phi, ops::Operators, conf::Config)
    phi_hat = to_spectral!(ops.temp_comp1, phi, ops)
    W2 = get_W2(phi, ops, conf) # 这里的 W2 是所有囊泡的第二能量之和
    R2 = sqrt(W2 + conf.C2)
    ε = conf.epsilon
    omega = get_omega(phi, ops, conf)
    omega_hat = to_spectral!(ops.temp_comp2, omega, ops)
    lap_omega = to_physical!(ops.temp_real1, ops.Laplacian .* omega_hat, ops)

    H2_Surf_Bend = @. conf.gamma_surf * (3 * sqrt(2) / 4) * omega +
                      conf.gamma_bend * (3 * sqrt(2) / (8ε)) *
                      (omega * (3phi^2 - 1) / ε - ε * lap_omega)

    bih_phi = to_physical!(ops.temp_real2, ops.Biharmonic .* phi_hat, ops)
    lap_phi = to_physical!(ops.temp_real3, ops.Laplacian .* phi_hat, ops)
    H2_lin = @. conf.S1 * bih_phi - conf.S2 * lap_phi + conf.S3 * phi

    return @. (H2_Surf_Bend - H2_lin) / R2
end

function get_H3(phi, psi, conf::Config)
    W3 = get_W3(phi, psi, conf) # 这里的 W3 是所有囊泡的第三能量之和
    R3 = sqrt(W3 + conf.C3)
    psi_in = reshape(conf.psi_in_v, 1, 1, 1, :)
    psi_out = reshape(conf.psi_out_v, 1, 1, 1, :)
    f_in = @. 0.5 * conf.gamma_in * (psi - psi_in)^2 + conf.beta_in
    f_out = @. 0.5 * conf.gamma_out * (psi - psi_out)^2 + conf.beta_out
    return @. (0.5 * p_phi_prime(phi) * (f_in - f_out)) / R3
end

function get_MG(phi, psi, conf::Config)
    W3 = get_W3(phi, psi, conf) # 这里的 W3 是所有囊泡的第三能量之和
    R3 = sqrt(W3 + conf.C3)
    P = p_phi(phi)
    psi_in = reshape(conf.psi_in_v, 1, 1, 1, :)
    psi_out = reshape(conf.psi_out_v, 1, 1, 1, :)
    dF_in = @. conf.gamma_in * (psi - psi_in)
    dF_out = @. conf.gamma_out * (psi - psi_out)
    G1 = @. 0.5 * (1 + P) * dF_in + 0.5 * (1 - P) * dF_out
    G2 = @. conf.S4 * psi
    return @. (G1 - G2) / R3
end