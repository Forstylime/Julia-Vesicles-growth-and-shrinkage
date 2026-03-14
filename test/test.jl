using BenchmarkTools
println("-------------------------------------")
println("加载模块...\n")
include("../src/Types.jl")
include("../src/SpectralUtils.jl")
include("../src/Utils.jl")
include("../src/Solvers.jl")
include("../src/Init.jl")
conf = set_para_base(1e-6, 1e-3)
ops  = build_operators(conf)
present = generate_initial_condition(conf, ops, 1)
old  = deepcopy(present)
step1_cache = Step1Cache(conf.Nx, conf.Ny, conf.N)
bdf  = BDFCoeff(Val(2))

## ── 逐步 benchmark（只跑一次用 @time，要统计用 @btime）──
println("=== Step 1 ===")
@btime solve_step1($present, $old, $ops, $conf, $bdf, $step1_cache)

println("=== Step 2 ===")
step1_res = solve_step1(present, old, ops, conf, bdf, step1_cache)
@btime solve_step2($present, $old, $ops, $conf, $step1_cache, $bdf)

println("=== Step 4 ===")
@btime solve_step4($present, $old, $ops, $conf, $bdf)

println("=== Step 5 ===")
step2_res = solve_step2(present, old, ops, conf, step1_res, bdf)
step3_res = solve_step3(step1_res, step2_res)
step4_res = solve_step4(present, old, ops, conf, bdf)
@btime solve_step5($present, $old, $ops, $conf, $step3_res, $step4_res, $bdf)

##
using Profile, ProfileView
using FFTW
using LinearAlgebra
using Printf
using ProgressMeter
using CairoMakie
println("-------------------------------------")
println("加载模块...\n")
include("../src/Types.jl")
include("../src/SpectralUtils.jl")
include("../src/Utils.jl")
include("../src/Solvers.jl")
include("../src/Init.jl")
include("../src/simulation.jl")
include("../scripts/visualize.jl")
println("模块加载完成.\n")
# 先 warm up（让 JIT 编译完成）
run_simulation(1e-6, 1e-4, 1)  # 短时间热身

# 正式采样
Profile.clear()
@profile run_simulation(1e-6, 1e-3, 1)

# 方法 A：终端文本输出（按时间降序）
Profile.print(sortedby=:count, mincount=50)

# 方法 B：交互式火焰图（强烈推荐）
ProfileView.view()

## 
# 检查 solve_step1 的类型推断
@code_warntype solve_step1(present, old, ops, conf, bdf)

##
# 命令行启动 Julia，开启行级内存追踪
#julia --track-allocation=user scripts/Main.jl

##
# 确认无类型不稳定警告
@code_warntype solve_step1(present, old, ops, conf, bdf)

##
@btime solve_step1($present, $old, $ops, $conf, $bdf)