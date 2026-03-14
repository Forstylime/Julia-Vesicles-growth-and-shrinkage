# File: src/Init.jl

using FFTW

"""
第一阶段：不含真实 A，用于生成初始场
"""
function set_para_base(dt::Float64, T::Float64)
    N = 2   # single phase field for now
    Nx, Ny   = 256, 256
    Lx, Ly   = 1.0 * 1, 1.0
    epsilon  = 0.01
    gamma_bend = 0.1
    gamma_in, gamma_out = 1e5, 1e5
    goal =:g

    if N == 1
        psi_in_v  = [0.1]
        psi_out_v = [0.8]
    elseif N == 2
        psi_in_v  = [0.65, 0.65]
        psi_out_v = [0.8, 0.8]
    elseif N == 3
        psi_in_v  = [0.1, 0.65, 0.5]
        psi_out_v = [0.8, 0.6, 0.4]
    end

    S1 = (3 * sqrt(2) / 8) * epsilon * gamma_bend
    S2 = 0.0
    S3 = 1.0 * gamma_bend / (epsilon^3)
    S4 = 1.0 * min(gamma_in, gamma_out)

    # 使用 keyword 构造函数，Nt/dx/dy 自动计算
    return Config(
        N = N,
        epsilon     = epsilon,
        M_phi       = 1.0,
        M0_psi      = 0.5,
        eta         = 1.0,
        gamma_surf  = 1.0,
        gamma_area  = 5e4,
        gamma_bend  = gamma_bend,
        gamma_in    = gamma_in,
        beta_in     = 0.0,
        psi_in_v    = psi_in_v,
        gamma_out   = gamma_out,
        beta_out    = 0.0,
        psi_out_v   = psi_out_v,
        lamda       = 1.0,
        S1 = S1, S2 = S2, S3 = S3, S4 = S4,
        C1 = 1.0, C2 = 5.0e5, C3 = 1.0e6,
        dt = dt, T = T,
        Nx = Nx, Ny = Ny, Lx = Lx, Ly = Ly,
        tol  = 1e-12,
        goal = goal
)
end

"""
生成初始物理场
"""
function generate_initial_condition(conf::Config, ops::Operators, state_type::Int)
    Nx, Ny   = conf.Nx, conf.Ny
    dx, dy   = conf.dx, conf.dy
    Lx, Ly   = conf.Lx, conf.Ly
    epsilon  = conf.epsilon
    N        = conf.N

    x_nodes = range(0, Lx - dx, length=Nx)
    y_nodes = range(0, Ly - dy, length=Ny)
    X = [x for x in x_nodes, y in y_nodes]
    Y = [y for x in x_nodes, y in y_nodes]

    phi = zeros(Nx, Ny, N)  # 3D array for N phase fields; currently N=1

    if N == 1
        cx = [Lx / 2]
        cy = [Ly / 2]
    elseif N == 2
        cx = [0.3, 0.8]
        cy = [Ly / 2, Ly / 2]
    elseif N == 3
        cx = [Lx / 4, Lx / 2, 3Lx / 4]
        cy = [Ly / 2, Ly / 2, Ly / 2]
    end

    if N == 1
        if state_type > 6
            error("错误：state_type 值不合法 (state_type = $state_type, N = $N)，请检查输入数据！")
        end
        cx_n, cy_n = cx, cy
        if state_type == 1  # 单个椭圆
            R      = 0.2 * Lx
            dist   = @. sqrt((X - cx_n)^2 / 2 + (Y - cy_n)^2)
            phi[:, :, 1] .= @. tanh((R - dist) / (sqrt(2) * epsilon))

        elseif state_type == 2  # 三角形
            Xc, Yc    = X .- cx_n, Y .- cy_n
            radius    = 0.65 * Lx
            smoothness = 40.0
            k_sin, k_cos = sqrt(3) / 2, 0.5
            d1 = @. -Yc
            d2 = @. k_sin * Xc + k_cos * Yc
            d3 = @. -k_sin * Xc + k_cos * Yc
            mx       = @. max(d1, max(d2, d3))
            d_shape  = @. mx + (1 / smoothness) * log(
                        exp(smoothness * (d1 - mx)) +
                        exp(smoothness * (d2 - mx)) +
                        exp(smoothness * (d3 - mx)))
            d_final  = @. d_shape - (radius / 3)
            phi[:, :, 1] .= @. -tanh(d_final / (sqrt(2) * epsilon))
        else state_type == 3  # star-shape
            if conf.goal === :s
                R0 = 0.3
                amplitude = 0.01
            else
                R0 = 0.18
                amplitude = 0.02
            end
            k = 10
            r = @. sqrt((X - cx_n)^2 + (Y - cy_n)^2)
            theta = @. atan(X - cx_n, Y - cy_n)
            R_theta = @. R0 + amplitude * cos(k * theta)
            phi[:, :, 1] .= @. tanh((R_theta - r) / (sqrt(2) * epsilon))
        end
    else
        if state_type <= 6
            error("错误：state_type 值不合法 (state_type = $state_type, N = $N)，对N>1,state_type>6！")
        end
        for n in 1:N
            cx_n, cy_n = cx[n], cy[n]
            if state_type == 7 # 两个star-shape
                if conf.goal ===:s
                    error("对state_type=$state_type,conf.goal应该是g")
                end
                R0 = 0.18
                amplitude = 0.02
                k = 10
                r = @. sqrt((X - cx_n)^2 + (Y - cy_n)^2)
                theta = @. atan(X - cx_n, Y - cy_n)
                R_theta = @. R0 + amplitude * cos(k * theta)
                phi[:, :, n] .= @. tanh((R_theta - r) / (sqrt(2) * epsilon))
            end
        end
    end
    # 初始化 psi
    psi = zeros(Nx, Ny, N)
    for n in 1:N
        psi[:, :, n] = if conf.goal === :s
            @. -0.1 * phi[:, :, n] + 0.7
        else
            @. -0.35 * phi[:, :, n] + 0.45
        end
    end

    # FFT —— 注意 u 沿前两维变换，第3维是方向分量
    phi_hat = ops.fft_plan * phi
    psi_hat = ops.fft_plan * psi
    u       = zeros(Nx, Ny, 2)
    u_hat   = ops.fft_plan_2 * u
    p       = zeros(Nx, Ny)
    p_hat   = ops.fft_plan_1 * p
    mu      = zeros(Nx, Ny, N);  mu_hat = ops.fft_plan * mu
    nu      = zeros(Nx, Ny, N);  nu_hat = ops.fft_plan * nu

    return FieldState(
        phi, phi_hat, psi, psi_hat,
        u, u_hat, p, p_hat,
        mu, mu_hat, nu, nu_hat,
        1.0, 1.0, 1.0, 1.0, [0.0], 0.0     # Q, R1, R2, R3, A0
    )
end