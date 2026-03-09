# File: src/Utils.jl

# ----------------------------------------------------------------
# 基础标量函数（用 @inline 确保被调用处内联，无函数调用开销）
# ----------------------------------------------------------------

"双井势 g(φ) = ¼(φ²-1)²"
@inline g(phi) = @. 0.25 * (phi^2 - 1.0)^2

"g'(φ) = φ(φ²-1)"
@inline g_prime(phi) = @. phi * (phi^2 - 1.0)

"插值函数 p(φ) = -½φ³ + 3/2·φ"
@inline p_phi(phi) = @. -0.5 * phi^3 + 1.5 * phi

"p'(φ) = -3/2·φ² + 3/2"
@inline p_phi_prime(phi) = @. -1.5 * phi^2 + 1.5

"可变扩散系数 M_ψ"
@inline get_M_psi(phi, conf::Config) = @. 0.005 * (1.0 - conf.M0_psi * (phi^2 - 1.0)^2)

# ----------------------------------------------------------------
# 积分辅助（dx*dy 梯形近似）
# ----------------------------------------------------------------

@inline integrate(f, conf::Config)    = sum(f)        * conf.dx * conf.dy
@inline integrate_sq(f, conf::Config) = sum(abs2, f)  * conf.dx * conf.dy

# ----------------------------------------------------------------
# 能量密度函数
# ----------------------------------------------------------------

"表面能密度 f_surf"
function f_surf(phi, ops::Operators, conf::Config)
    phi_hat = to_spectral(phi, ops)
    grad_x = to_physical(ops.D1[1] .* phi_hat, ops)
    grad_y = to_physical(ops.D1[2] .* phi_hat, ops)
    ε = conf.epsilon
    return @. (3 * sqrt(2) / 4) * (g(phi) / ε) + ε * (grad_x^2 + grad_y^2) / 2
end

"弯曲能密度 f_bend"
function f_bend(phi, ops::Operators, conf::Config)
    phi_hat = to_spectral(phi, ops)
    ε = conf.epsilon
    lap_phi = real(ops.ifft_plan * (ops.Laplacian .* phi_hat))
    return @. (3 * sqrt(2) / (16ε)) * (g_prime(phi) / ε - ε * lap_phi)^2
end

"渗透能密度 f_osmotic"
function f_osmotic(phi, psi, conf::Config)
    P     = p_phi(phi)
    f_in  = @. 0.5 * conf.gamma_in  * (psi - conf.psi_in)^2  + conf.beta_in
    f_out = @. 0.5 * conf.gamma_out * (psi - conf.psi_out)^2 + conf.beta_out
    return @. ((1 + P) / 2) * f_in + ((1 - P) / 2) * f_out
end

# ----------------------------------------------------------------
# 面积计算
# ----------------------------------------------------------------

function calculate_area(phi, ops::Operators, conf::Config)
    return integrate(f_surf(phi, ops, conf), conf)
end

# ----------------------------------------------------------------
# 原始物理能量
# ----------------------------------------------------------------

function compute_original_energy(state::FieldState, ops::Operators, conf::Config)
    kinetic  = 0.5 * integrate_sq(state.u, conf)

    F_s = conf.gamma_surf * integrate(f_surf(state.phi, ops, conf), conf)
    F_b = conf.gamma_bend * integrate(f_bend(state.phi, ops, conf), conf)
    F_o = integrate(f_osmotic(state.phi, state.psi, conf), conf)

    A_current = calculate_area(state.phi, ops, conf)
    F_a = 0.5 * conf.gamma_area * (A_current - conf.A0)^2

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
    grad_p_x = real(ops.ifft_plan * (ops.D1[1] .* present.p_hat))
    grad_p_y = real(ops.ifft_plan * (ops.D1[2] .* present.p_hat))
    pressure = (conf.dt^2 / 3.0) * (integrate_sq(grad_p_x, conf) +
                                     integrate_sq(grad_p_y, conf))

    # 3. SAV 标量能量
    sav_energy(r_new, r_old) = 0.25 * r_new^2 + 0.25 * (2r_new - r_old)^2
    R_energy = sav_energy(present.R1, old.R1) +
               sav_energy(present.R2, old.R2) +
               sav_energy(present.R3, old.R3)
    Q_energy = 0.5 * sav_energy(present.Q, old.Q)

    # 4. 线性算子稳定项
    phi_star     = @. 2 * present.phi - old.phi
    psi_star     = @. 2 * present.psi - old.psi
    phi_star_hat = @. 2 * present.phi_hat - old.phi_hat

    lap_new  = real(ops.ifft_plan * (ops.Laplacian .* present.phi_hat))
    lap_star = real(ops.ifft_plan * (ops.Laplacian .* phi_star_hat))

    gx_new  = real(ops.ifft_plan * (ops.D1[1] .* present.phi_hat))
    gy_new  = real(ops.ifft_plan * (ops.D1[2] .* present.phi_hat))
    gx_star = real(ops.ifft_plan * (ops.D1[1] .* phi_star_hat))
    gy_star = real(ops.ifft_plan * (ops.D1[2] .* phi_star_hat))

    S1_term = 0.25 * conf.S1 * (integrate_sq(lap_new,  conf) +
                                 integrate_sq(lap_star, conf))
    S2_term = 0.25 * conf.S2 * (integrate_sq(gx_new,  conf) + integrate_sq(gy_new,  conf) +
                                 integrate_sq(gx_star, conf) + integrate_sq(gy_star, conf))
    S3_term = 0.25 * conf.S3 * (integrate_sq(present.phi, conf) +
                                 integrate_sq(phi_star,    conf))
    S4_term = 0.25 * conf.S4 * (integrate_sq(present.psi, conf) +
                                 integrate_sq(psi_star,    conf))

    return (kinetic + pressure +
            conf.lamda * R_energy + Q_energy +
            conf.lamda * (S1_term + S2_term + S3_term + S4_term) -
            conf.lamda * (conf.C1 + conf.C2 + conf.C3) - 0.5)
end

# ----------------------------------------------------------------
# 变分中间量
# ----------------------------------------------------------------

"ω = (φ³-φ)/ε - ε·Δφ"
function get_omega(phi, ops::Operators, conf::Config)
    phi_hat = ops.fft_plan * phi
    ε = conf.epsilon
    lap_phi = real(ops.ifft_plan * (ops.Laplacian .* phi_hat))
    return @. (phi^3 - phi) / ε - ε * lap_phi
end

# ----------------------------------------------------------------
# SAV 根号内能量 W1, W2, W3
# ----------------------------------------------------------------

function get_W1(phi, ops::Operators, conf::Config)
    A_cur = calculate_area(phi, ops, conf)
    W1    = 0.5 * conf.gamma_area * (A_cur - conf.A0)^2
    W1 + conf.C1 < 0 && error("FATAL: W1 + C1 < 0，C1 偏小！W1 = $W1")
    return W1
end

function get_W2(phi, ops::Operators, conf::Config)
    phi_hat = to_spectral(phi, ops)
    F_surf_val = conf.gamma_surf * integrate(f_surf(phi, ops, conf), conf)
    F_bend_val = conf.gamma_bend * integrate(f_bend(phi, ops, conf), conf)

    lap_phi  = real(ops.ifft_plan * (ops.Laplacian .* phi_hat))
    grad_x   = real(ops.ifft_plan * (ops.D1[1]     .* phi_hat))
    grad_y   = real(ops.ifft_plan * (ops.D1[2]     .* phi_hat))

    E2 = integrate(@.(0.5 * conf.S1 * lap_phi^2 +
                      0.5 * conf.S2 * (grad_x^2 + grad_y^2) +
                      0.5 * conf.S3 * phi^2), conf)

    W2 = F_surf_val + F_bend_val - E2
    W2 + conf.C2 < 0 && error("FATAL: W2 + C2 < 0，C2 偏小！W2 = $W2")
    return W2
end

function get_W3(phi, psi, conf::Config)
    W_osm = integrate(f_osmotic(phi, psi, conf), conf)
    E3    = 0.5 * conf.S4 * integrate_sq(psi, conf)
    W3    = W_osm - E3
    W3 + conf.C3 < 0 && error("FATAL: W3 + C3 < 0，C3 偏小！W3 = $W3")
    return W3
end

# ----------------------------------------------------------------
# 变分导数 H1, H2, H3, MG
# ----------------------------------------------------------------

function get_H1(phi, ops::Operators, conf::Config)
    W1      = get_W1(phi, ops, conf)
    R1      = sqrt(W1 + conf.C1)
    A_cur   = calculate_area(phi, ops, conf)
    omega   = get_omega(phi, ops, conf)
    factor  = conf.gamma_area * (A_cur - conf.A0) * (3 * sqrt(2) / 4)
    return @. (factor * omega) / R1
end

function get_H2(phi, ops::Operators, conf::Config)
    phi_hat = to_spectral(phi, ops)
    W2    = get_W2(phi, ops, conf)
    R2    = sqrt(W2 + conf.C2)
    ε     = conf.epsilon
    omega = get_omega(phi, ops, conf)

    omega_hat = ops.fft_plan * omega
    lap_omega = real(ops.ifft_plan * (ops.Laplacian .* omega_hat))

    H2_Surf_Bend = @. conf.gamma_surf * (3 * sqrt(2) / 4) * omega +
               conf.gamma_bend * (3 * sqrt(2) / (8ε)) *
               (omega * (3phi^2 - 1) / ε - ε * lap_omega)

    bih_phi = real(ops.ifft_plan * (ops.Biharmonic .* phi_hat))
    lap_phi = real(ops.ifft_plan * (ops.Laplacian  .* phi_hat))
    H2_lin  = @. conf.S1 * bih_phi - conf.S2 * lap_phi + conf.S3 * phi

    return @. (H2_Surf_Bend - H2_lin) / R2
end

function get_H3(phi, psi, conf::Config)
    W3    = get_W3(phi, psi, conf)
    R3    = sqrt(W3 + conf.C3)
    f_in  = @. 0.5 * conf.gamma_in  * (psi - conf.psi_in)^2  + conf.beta_in
    f_out = @. 0.5 * conf.gamma_out * (psi - conf.psi_out)^2 + conf.beta_out
    return @. (0.5 * p_phi_prime(phi) * (f_in - f_out)) / R3
end

function get_MG(phi, psi, conf::Config)
    W3    = get_W3(phi, psi, conf)
    R3    = sqrt(W3 + conf.C3)
    P     = @. p_phi(phi)
    dF_in  = @. conf.gamma_in  * (psi - conf.psi_in)
    dF_out = @. conf.gamma_out * (psi - conf.psi_out)
    G1    = @. 0.5 * (1 + P) * dF_in + 0.5 * (1 - P) * dF_out
    G2    = @. conf.S4 * psi
    return @. (G1 - G2) / R3
end