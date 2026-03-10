# File: Main.jl
println("加载模块...\n")

using FFTW
using LinearAlgebra
using Printf
using ProgressMeter
using CairoMakie
using IterativeSolvers

include("../src/Types.jl")
include("../src/SpectralUtils.jl")
include("../src/Utils.jl")
include("../src/Solvers.jl")
include("../src/Init.jl")

println("模块加载完成。开始仿真...")

## simulation function
function run_simulation(dt_val::Float64, T_val::Float64, state_type::Int;
                        save_path::String="./results",
                        save_frames::Int=20)

    # ── 1. 初始化配置与算子 ──────────────────────────────────────
    conf = set_para_base(dt_val, T_val; goal=:s)
    ops  = build_operators(conf)
    mkpath(save_path)
    # 预分配内存
    #present = FieldState(conf.Nx, conf.Ny, conf.N)

    # ── 2. 生成初始场 ────────────────────────────────────────────
    present = generate_initial_condition(conf, ops, state_type)

    # 用 f_surf 积分计算每个囊泡的真实面积（与模型约定一致)
    for n in 1:conf.N
        A_n = calculate_area(present.phi[:, :, n], ops, conf)
        conf.A0[n] = A_n
    end

    # 初始化 SAV 变量
    present.R1 = sqrt(get_W1(present.phi, ops, conf) + conf.C1)
    present.R2 = sqrt(get_W2(present.phi, ops, conf) + conf.C2)
    present.R3 = sqrt(get_W3(present.phi, present.psi, conf) + conf.C3)
    present.Q  = 1.0

    # 计算初始化学势
    L_phi = @. conf.S1 * ops.Biharmonic - conf.S2 * ops.Laplacian + conf.S3

    H1 = get_H1(present.phi, ops, conf)
    H2 = get_H2(present.phi, ops, conf)
    H3 = get_H3(present.phi, present.psi, conf)
    G  = get_MG(present.phi, present.psi, conf)

    present.mu_hat .= L_phi          .* present.phi_hat .+
                      present.R1     .* (ops.fft_plan * H1) .+
                      present.R2     .* (ops.fft_plan * H2) .+
                      present.R3     .* (ops.fft_plan * H3)
    present.nu_hat .= conf.S4        .* present.psi_hat .+
                      present.R3     .* (ops.fft_plan * G)

    present.mu .= ops.ifft_plan * present.mu_hat
    present.nu .= ops.ifft_plan * present.nu_hat

    # BDF2 需要两个时间层，初始令 old = present
    old = deepcopy(present)

    # ── 3. 监控变量 ──────────────────────────────────────────────
    energy_history    = Float64[]
    save_interval     = max(1, conf.Nt ÷ save_frames)

    @info "仿真开始。初始 Q = $(present.Q)，初始面积 = $(conf.A0)"

    # ── 4. 时间推进循环 ──────────────────────────────────────────
    p_meter = Progress(conf.Nt, 1, "Computing...")

    for n in 1:conf.Nt

        # BDF1（首步）或 BDF2
        bdf = bdf_coeff(n) #n == 1 ? BDFCoeff(1.0, -1.0, 0.0) : BDFCoeff(1.5, -2.0, 0.5)

        # ── 核心 7 步 ──
        step1_res = solve_step1(present, old, ops, conf, bdf)
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
        present.mu_hat  .= step6_res.mu_hat
        present.nu_hat  .= step6_res.nu_hat
        present.u_hat   .= step7_res.u_hat
        present.p_hat   .= step7_res.p_hat

        # 实空间（用 view 避免切片分配）
        present.phi .= real(ops.ifft_plan * present.phi_hat)
        present.psi .= real(ops.ifft_plan * present.psi_hat)
        present.mu  .= real(ops.ifft_plan * present.mu_hat)
        present.nu  .= real(ops.ifft_plan * present.nu_hat)
        present.p   .= real(ops.ifft_plan * present.p_hat)
        @views present.u[:,:,1] .= real(ops.ifft_plan * present.u_hat[:,:,1])
        @views present.u[:,:,2] .= real(ops.ifft_plan * present.u_hat[:,:,2])

        # 标量
        present.R1 = step6_res.R1
        present.R2 = step6_res.R2
        present.R3 = step6_res.R3
        present.Q  = step5_res

        # ── 监控 ──
        push!(energy_history, compute_modified_energy(present, old, ops, conf))

        if n % save_interval == 0
            t = n * conf.dt
            save_visualization(present, conf, t, save_path)
            @info @sprintf("Step %d / %d  t = %.4f  E = %.6e",
                           n, conf.Nt, t, last(energy_history))
        end

        next!(p_meter)
    end

    return present, energy_history
end

## ────────────────────────────────────────────────────────────────
# 辅助函数
"""
function copy_fieldstate!(dest::FieldState, src::FieldState)
    for f in (:phi, :phi_hat, :psi, :psi_hat,
              :u, :u_hat, :p, :p_hat,
              :mu, :mu_hat, :nu, :nu_hat)
        getfield(dest, f) .= getfield(src, f)
    end
    dest.Q  = src.Q
    dest.R1 = src.R1
    dest.R2 = src.R2
    dest.R3 = src.R3
    dest.A  = src.A    # 修正：补全 A 字段
end
"""
function save_visualization(state::FieldState, conf::Config, t::Float64, path::String)
    fig = Figure(size = (1200, 1000), backgroundcolor = :white)  # 修正：size
    ax  = Axis(fig[1, 1], aspect=DataAspect(),
               title=@sprintf("t = %.3f", t))

    hm = heatmap!(ax, state.phi, colormap=:viridis, colorrange=(-1.0, 1.0))
    Colorbar(fig[1, 2], hm)

    stride = 8
    xs = 1:stride:conf.Nx
    ys = 1:stride:conf.Ny
    arrows!(ax,
            vec([i for i in xs, j in ys]),
            vec([j for i in xs, j in ys]),
            vec(state.u[xs, ys, 1]),
            vec(state.u[xs, ys, 2]),
            arrowsize=10, lengthscale=10.0, color=:black)

    save(joinpath(path, @sprintf("phi_vel_%.3f.png", t)), fig)
end


## ── 入口 ──────────────────────────────────────────────────────
run_simulation(0.000001, 0.0001, 1)