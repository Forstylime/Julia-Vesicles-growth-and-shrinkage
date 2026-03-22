## simulation function
function run_simulation(dt_val::Float64, T_val::Float64, state_type::Int;
    save_path::String="./results", #默认保存结果到"./results"文件夹
    save_interval::Float64=0.001)

    # Ensure MAT directory exists
    mat_folder = joinpath(@__DIR__, "..", "MAT\\case3")
    mkpath(mat_folder)

    # ── 1. 初始化配置与算子 ──────────────────────────────────────
    conf = set_para_base(dt_val, T_val)
    ops = build_operators(conf)
    mkpath(save_path)

    t1 = range(0.0, 1e-6, step=1e-8)   # 第一段
    t2 = range(1e-6, T_val, step=conf.dt)    # 第二段
    # 拼接：注意使用 vcat，为了避免重叠点，通常拼接时剔除后续段的起始点
    Dt = vcat(t1, t2[2:end])
    Nt = length(Dt) # 理论上，总步数 = 100 + (T - 1e-6)/dt

    # ── 2. 生成初始场 ────────────────────────────────────────────
    present = generate_initial_condition(conf, ops, state_type)

    # 用 f_surf 积分计算每个囊泡的真实面积（与模型约定一致)
    A_0 = calculate_area(present.phi, ops, conf)
    present.A0 = A_0  # 将初始面积目标存入状态变量

    # 初始化 SAV 变量
    present.R1 = sqrt(get_W1(present.phi, ops, conf, A_0) + conf.C1)
    present.R2 = sqrt(get_W2(present.phi, ops, conf) + conf.C2)
    present.R3 = sqrt(get_W3(present.phi, present.psi, conf) + conf.C3)
    present.Q = 1.0

    # 计算初始化学势
    L_phi = @. conf.S1 * ops.Biharmonic - conf.S2 * ops.Laplacian + conf.S3

    H1 = get_H1(present.phi, ops, conf, A_0)
    H2 = get_H2(present.phi, ops, conf)
    H3 = get_H3(present.phi, present.psi, conf)
    G = get_MG(present.phi, present.psi, conf)

    present.mu_hat .= L_phi .* present.phi_hat .+
                      present.R1 .* (ops.fft_plan * H1) .+
                      present.R2 .* (ops.fft_plan * H2) .+
                      present.R3 .* (ops.fft_plan * H3)
    present.nu_hat .= conf.S4 .* present.psi_hat .+
                      present.R3 .* (ops.fft_plan * G)

    present.mu .= ops.ifft_plan * present.mu_hat
    present.nu .= ops.ifft_plan * present.nu_hat

    # BDF2 需要两个时间层，初始令 old = present
    old = deepcopy(present)
    if ndims(present.phi) == 3
        step1_cache = Step1Cache(conf.Nx, conf.Ny, conf.N)
    elseif ndims(present.phi) == 4
        step1_cache = Step1Cache(conf.Nx, conf.Ny, conf.Nz, conf.N)
    else
        error("数据维度错误, size = $(size(presnet.phi))")
    end
    # ── 3. 监控变量 ──────────────────────────────────────────────
    energy_history = Float64[]
    area_ratio_history = Float64[]

    @info "仿真开始。初始面积 = $(present.A0), 总步数 = $(Nt-1)"

    # ── 4. 时间推进循环 ──────────────────────────────────────────
    p_meter = Progress(Nt - 1, 1, "Computing...")

    for n in 1:Nt-1

        present.dt = Dt[n+1] - Dt[n]

        # BDF1（首步）或 BDF2
        bdf = bdf_coeff(n) #n == 1 ? BDFCoeff(1.0, -1.0, 0.0) : BDFCoeff(1.5, -2.0, 0.5)

        # ── 核心 7 步 ──
        step1_res = solve_step1(present, old, ops, conf, bdf, step1_cache)
        step2_res = solve_step2(present, old, ops, conf, step1_res, bdf)
        step3_res = solve_step3(step1_res, step2_res)
        step4_res = solve_step4(present, old, ops, conf, bdf)
        step5_res = solve_step5(present, old, ops, conf, step3_res, step4_res, bdf)
        step6_res = solve_step6(step2_res, step3_res, step4_res, step5_res)
        step7_res = solve_step7(present, ops, conf, step6_res, bdf)

        # ── 状态更新 ──
        update_state!(old, present)   # old <-- present

        # 谱空间
        present.phi_hat .= step6_res.phi_hat
        present.psi_hat .= step6_res.psi_hat
        present.mu_hat .= step6_res.mu_hat
        present.nu_hat .= step6_res.nu_hat
        present.u_hat .= step7_res.u_hat
        present.p_hat .= step7_res.p_hat

        # 实空间（用 view 避免切片分配）
        to_physical!(present.phi, present.phi_hat, ops)
        to_physical!(present.psi, present.psi_hat, ops)
        to_physical!(present.mu, present.mu_hat, ops)
        to_physical!(present.nu, present.nu_hat, ops)
        present.p .= ops.ifft_plan_1 * present.p_hat # 使用特制的plan_1
        present.u .= ops.ifft_plan_2 * present.u_hat # 使用特制的plan_2

        # 标量
        present.R1 = step6_res.R1
        present.R2 = step6_res.R2
        present.R3 = step6_res.R3
        present.Q = step5_res

        # 保存中间时刻结果

        if mod(Dt[n], save_interval) == 0
            t_str = @sprintf("%.2e", Dt[n])

            # 准备要保存的状态数据字典
            state_to_save = Dict(
                "phi" => present.phi,
                "psi" => present.psi,
                "mu" => present.mu,
                "nu" => present.nu,
                "p" => present.p,
                "u" => present.u,
                "R1" => present.R1,
                "R2" => present.R2,
                "R3" => present.R3,
                "Q" => present.Q
            )

            if ndims(present.phi) == 3
                phi_plot = dropdims(sum(present.phi, dims=3), dims=3) .+ conf.N .- 1
                plot_field(phi_plot, conf,
                    title=@sprintf("phi  (t = %.2e)", Dt[n]),
                    filename=joinpath(save_path, "phi_t$(t_str).png"))
                matwrite(joinpath(mat_folder, "state_2d_t$(t_str).mat"), state_to_save)
            elseif ndims(present.phi) == 4
                matwrite(joinpath(mat_folder, "state_3d_t$(t_str).mat"), state_to_save)
            else
                error("Size of field (phi, psi, ...) = $(size(present.phi)) not correct!")
            end
        end

        # ── 监控 ──
        push!(energy_history, compute_modified_energy(present, old, ops, conf))
        area_ratio = sum(abs.(calculate_area(present.phi, ops, conf) .- A_0)) ./ sum(A_0)
        push!(area_ratio_history, area_ratio)

        next!(p_meter) # 更新进度条

    end

    return present, energy_history, area_ratio_history, Dt
end