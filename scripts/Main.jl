# ╔══════════════════════════════════════════════════════════════╗
# ║  Multiple_SAV  ·  scripts/Main.jl                            ║
# ║                                                              ║
# ║  Workflow:                                                   ║
# ║    1. First run: include("scripts/Main.jl") in Julia REPL    ║
# ║    2. After editing any src/*.jl, Revise auto-reloads it;    ║
# ║       just call main() again — no restart needed.            ║
# ║    3. Simulation parameters can be changed freely in REPL.   ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 0. Revise must come first so it can track all subsequent includet calls ──
using Revise

# ── 1. Standard library & third-party packages ───────────────────────────────
using FFTW
using LinearAlgebra
using Printf
using ProgressMeter
using CairoMakie
using GLMakie
using MAT
# using Infiltrator          # debug aid; remove for production runs

FFTW.set_num_threads(1)  # 256×256 单线程已是最优
BLAS.set_num_threads(4)  # BLAS 的矩阵运算可以多线程，但你的矩阵都很小（3×3），实际也没用

# ── 2. Project source files (includet = include + Revise tracking) ───────────
# Load order matters: dependencies before dependents.
_SRC = joinpath(@__DIR__, "..", "src")
_SCR = joinpath(@__DIR__, "..", "scripts")


#    joinpath(_SRC, "Types_3d.jl"),
#    joinpath(_SRC, "SpectralUtils_3d.jl"),
#    joinpath(_SRC, "Utils_3d.jl"),
#    joinpath(_SRC, "Init_3d.jl"),
#    joinpath(_SRC, "Solvers_3d.jl"),

#    joinpath(_SRC, "Types.jl"),
#    joinpath(_SRC, "SpectralUtils.jl"),
#    joinpath(_SRC, "Utils.jl"),
#    joinpath(_SRC, "Init.jl"),
#    joinpath(_SRC, "Solvers.jl"),

for file in (
    joinpath(_SRC, "Types.jl"),
    joinpath(_SRC, "SpectralUtils.jl"),
    joinpath(_SRC, "Utils.jl"),
    joinpath(_SRC, "Init.jl"),
    joinpath(_SRC, "Solvers.jl"),
    joinpath(_SRC, "simulation.jl"),
    joinpath(_SCR, "visualize.jl"),
)
    includet(file)
end

@info "All modules loaded. Revise hot-reload is active."

# ══════════════════════════════════════════════════════════════════════════════
#  Simulation entry point
#
#  Wrapping everything in a function provides:
#    · No global variable pollution → correct type inference → better perf
#    · Multiple calls with different parameters in a single REPL session
#    · After Revise reloads a source file, just call main() again
#
#  NOTE on run_simulation return value:
#    run_simulation must return (state, energy_history, Dt) where Dt is the
#    full time-node vector built inside the solver:
#
#        t1 = range(0.0, 1e-6, step=1e-8)   # fine warm-up phase
#        t2 = range(1e-6, T,   step=dt)      # main phase
#        Dt = vcat(t1, t2[2:end])
#
#    The time axis is non-uniform, so we must use Dt explicitly for plotting.
# ══════════════════════════════════════════════════════════════════════════════

"""
    main(; dt, T, state_type, save_path, save_frames) -> (state, energy_history, Dt)

Simulation entry point. All arguments have defaults; override as needed:

```julia
state, E, Dt = main()                                    # defaults
state, E, Dt = main(dt=1e-5, T=1e-3, state_type=7)
state, E, Dt = main(dt=1e-6, T=1e-3, save_frames=500)
```
"""
function main(;
    dt::Float64=1e-6,
    T::Float64=1e-4,
    state_type::Int=1,
    save_path::String=joinpath(@__DIR__, "..", "results"),
    save_frames::Int=100,
)
    mkpath(save_path)
    @info "Simulation started" dt T state_type save_path

    local state, energy_history, area_ratio_history, Dt
    try
        # run_simulation must return (state, energy_history, Dt)
        state, energy_history, area_ratio_history, Dt = run_simulation(
            dt, T, state_type;
            save_path=save_path,
            save_frames=save_frames,
        )
    catch e
        @error "Simulation terminated with error" exception = (e, catch_backtrace())
        rethrow()
    end

    @info "Simulation finished" steps = length(energy_history) final_energy = last(energy_history)

    _visualize(state, energy_history, area_ratio_history, Dt, T, save_path)

    return state, energy_history, area_ratio_history, Dt
end


# ── Internal visualisation helpers (not intended for direct REPL use) ─────────

function _visualize(state, energy_history, area_history, Dt::AbstractVector, T, save_path)
    conf = set_para_base(Dt[end], T)
    ndim = ndims(state.phi)
    # Phase field: sum over N vesicles so the plot works for any N >= 1
    phi_sum = dropdims(sum(state.phi, dims=ndim), dims=ndim) .+ conf.N .- 1

    # Ensure MAT directory exists
    mat_folder = joinpath(@__DIR__, "..", "MAT")
    mkpath(mat_folder)

    if ndim == 3
        fig_phi = plot_field(
            phi_sum, conf;
            title    = @sprintf("phi  (t = %.2e)", T),
            filename=joinpath(save_path, "phi_final.png")
        )
        matwrite(joinpath(mat_folder, "phi_2d.mat"), Dict("phi" => phi_sum))
    elseif ndim == 4
        # 去掉画图，只使用MAT.jl保存数据
        matwrite(joinpath(mat_folder, "phi_3d.mat"), Dict("phi" => phi_sum))
    else
        error("数据维度不匹配: size = $(size(state.phi))")
    end

    if length(energy_history) > 1
        t_energy = Dt[2:length(energy_history)+1]
        fig_E = plot_vector(energy_history, collect(t_energy);
            ylabel="Modified energy",
            title="Modified energy vs. time",
            filename=joinpath(save_path, "energy_history.png"))
    end
    if length(area_history) > 1
        t_area = Dt[2:length(area_history)+1]
        fig_E = plot_vector(area_history, collect(t_area);
            ylabel="Surface Ratio",
            title="Surface Ratio vs. time",
            filename=joinpath(save_path, "area_history.png"))
    end
end


# ══════════════════════════════════════════════════════════════════════════════
#  Script mode vs REPL mode
#
#  · include("scripts/Main.jl") in REPL  → loads definitions only, no autorun
#  · julia scripts/Main.jl               → also calls main() automatically
# ══════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end