# install_env.jl
#
# 一键初始化 Multiple_SAV 项目的 Julia 环境。
# 适用场景：首次克隆仓库后在新机器上配置环境。
#
# 使用方法（在项目根目录下）：
#   julia install_env.jl
#   julia --project=. install_env.jl   # 效果相同
#
# 该脚本不会修改 Project.toml 或 Manifest.toml，
# 只负责"按图施工"：解析现有的清单文件并安装/预编译所有依赖。

# ── 0. Julia 版本检查 ────────────────────────────────────────────────────────
const MIN_JULIA = v"1.9"

if VERSION < MIN_JULIA
    @error """
    当前 Julia 版本 ($VERSION) 低于项目最低要求 ($MIN_JULIA)。
    请从 https://julialang.org/downloads/ 下载更新版本后重试。
    """
    exit(1)
end

@info "Julia $VERSION  ✓"

# ── 1. 激活项目环境 ──────────────────────────────────────────────────────────
# @__DIR__ 始终指向本脚本所在目录（项目根目录），
# 与工作目录无关，因此在任何路径下执行均安全。
import Pkg

const PROJECT_ROOT = @__DIR__
const PROJECT_TOML = joinpath(PROJECT_ROOT, "Project.toml")
const MANIFEST_TOML = joinpath(PROJECT_ROOT, "Manifest.toml")

if !isfile(PROJECT_TOML)
    @error "找不到 Project.toml，请确认本脚本位于项目根目录。路径：$PROJECT_ROOT"
    exit(1)
end

Pkg.activate(PROJECT_ROOT)
@info "已激活项目环境：$PROJECT_ROOT"

# ── 2. 读取并展示声明的依赖（仅供用户确认，不作任何修改）──────────────────
declared_deps = collect(keys(Pkg.project().dependencies))
@info "Project.toml 中声明了 $(length(declared_deps)) 个直接依赖：" *
      "\n  " * join(sort(declared_deps), "\n  ")

# ── 3. Manifest 状态检查 ─────────────────────────────────────────────────────
if isfile(MANIFEST_TOML)
    @info "检测到 Manifest.toml，将严格按照锁文件版本安装（可复现构建）。"
else
    @warn """
    未检测到 Manifest.toml。
    将由 Pkg 解析最新兼容版本，结果可能与开发者环境略有不同。
    建议将 Manifest.toml 纳入版本控制以保证可复现性。
    """
end

# ── 4. 解析依赖关系 ──────────────────────────────────────────────────────────
@info "正在解析依赖关系..."
t_resolve = @elapsed try
    Pkg.resolve()
catch e
    @error "依赖解析失败，请检查网络连接或包版本兼容性。" exception = (e, catch_backtrace())
    exit(1)
end
@info @sprintf("解析完成（%.1f 秒）", t_resolve)

# ── 5. 实例化：下载并安装所有缺失的包 ───────────────────────────────────────
# instantiate 会自动跳过已安装的包，仅安装缺失部分，幂等且安全。
@info "正在实例化环境（首次运行可能需要数分钟下载依赖）..."
t_inst = @elapsed try
    Pkg.instantiate()
catch e
    @error "实例化失败。常见原因：网络不通、磁盘空间不足、包注册表缓存过期。" *
           "\n提示：可先运行 `julia -e 'import Pkg; Pkg.update()'` 刷新注册表后重试。" exception = (e, catch_backtrace())
    exit(1)
end
@info @sprintf("实例化完成（%.1f 秒）", t_inst)

# ── 6. 预编译 ────────────────────────────────────────────────────────────────
# 将所有包提前编译为原生缓存，大幅缩短后续 using/import 的加载时间。
# CairoMakie / GLMakie 体积较大，此步骤首次运行可能需要 5–15 分钟。
@info "正在预编译所有依赖（GLMakie / CairoMakie 首次编译耗时较长，请耐心等待）..."
t_pc = @elapsed try
    Pkg.precompile()
catch e
    # 预编译失败通常不致命（运行时仍会即时编译），降级为警告而非退出。
    @warn "预编译过程中出现警告或错误，后续运行时仍可能正常工作。" exception = (e, catch_backtrace())
end
@info @sprintf("预编译完成（%.1f 秒）", t_pc)

# ── 7. 验证关键包可以正常加载 ────────────────────────────────────────────────
# 仅验证项目核心依赖，避免对 GUI 包（GLMakie）做无头加载测试。
VERIFY_PKGS = ["FFTW", "Krylov", "LinearMaps", "ProgressMeter", "MAT", "Revise"]

@info "正在验证关键包的可加载性..."
failed = String[]
for pkg in VERIFY_PKGS
    try
        @eval using $(Symbol(pkg))
        @info "  $pkg  ✓"
    catch e
        push!(failed, pkg)
        @warn "  $pkg  ✗  →  $(sprint(showerror, e))"
    end
end

# ── 8. 汇总报告 ──────────────────────────────────────────────────────────────
println()
println("=" ^ 60)
if isempty(failed)
    println("""
    ✅  环境配置成功！

    现在可以在 Julia REPL 中运行：
        include("scripts/Main.jl")
        main()

    或直接从命令行启动：
        julia --project=. scripts/Main.jl

    提示：
      · 建议使用 `julia --threads auto` 以启用多线程。
      · 首次执行 `main()` 时 CairoMakie/GLMakie 仍有
        一次性 JIT 编译延迟，属正常现象。
    """)
else
    println("""
    ⚠️  以下包验证失败，请检查后手动安装：
        $(join(failed, ", "))

    可在 Julia REPL 中运行：
        import Pkg
        Pkg.activate("$PROJECT_ROOT")
        Pkg.add($(repr(failed)))
        Pkg.precompile()
    """)
end
println("=" ^ 60)
