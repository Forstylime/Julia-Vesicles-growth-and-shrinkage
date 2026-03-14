# File: Main.jl

println("=============================================")
println("加载依赖库...\n")
using FFTW
using LinearAlgebra
using Printf
using ProgressMeter
using CairoMakie
FFTW.set_num_threads(8)  # 或者 Sys.CPU_THREADS
BLAS.set_num_threads(8)

#using IterativeSolvers
println("依赖库加载完成。\n")
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

## ── 入口 ──────────────────────────────────────────────────────
println("开始仿真...")
# 主函数：设置参数并运行仿真，三个位置参数：时间步长、总时间、初始场类型（1-4），
# 两个关键字参数：数据保存路径和保存时间间隔，默认保存到当前目录下的 results 文件夹，每 100 步保存一次
dt = 1e-6
T  = 1e-4
results = run_simulation(dt, T, 3);
# 结果保存完成后，results 是一个包含时间序列数据的字典，可以用于后续分析和可视化
println("仿真完成！")

## 可视化
println("正在生成可视化图像...")
conf = set_para_base(dt, T) # goal 参数不影响可视化，可以不指定，直接使用默认值
phi = dropdims(results[1].phi, dims=3)
fig_phi = plot_field(phi, conf; filename="./results/final_phi.png");
display(fig_phi)