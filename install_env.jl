using Pkg

# 1. 激活项目根目录（更健壮，不依赖工作目录）
Pkg.activate(@__DIR__)

# 2. 待安装的包列表
dependencies = [
    "FFTW", "IterativeSolvers", "LinearMaps",
    "CairoMakie", "ProgressMeter",
    "StaticArrays", "StructArrays"
]

@info "正在检查并同步 Julia 项目环境..."

# 3. 智能安装：检查直接依赖中是否已存在
for pkg in dependencies
    if !haskey(Pkg.project().dependencies, pkg)
        @info "正在添加缺失的包: $pkg"
        try
            Pkg.add(pkg)
        catch e
            @warn "包 $pkg 安装失败，尝试跳过。错误信息: $e"
        end
    end
end

# 4. 解析依赖关系后再实例化，顺序更可靠
@info "正在解析依赖关系..."
Pkg.resolve()

@info "正在根据 Project.toml 实例化环境..."
Pkg.instantiate()

# 5. 预编译
@info "正在进行预编译（首次可能需要数分钟）..."
Pkg.precompile()

@info "------------------------------------------------"
@info "环境配置完成！现在你可以运行 Main.jl 了。"
@info "提示：CairoMakie 首次绘图会有额外的 JIT 编译延迟，属正常现象。"