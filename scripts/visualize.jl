# 单个场的可视化函数
function plot_field(field::Matrix{Float64}, cfg::Config;
                   title::Union{String,Nothing} = nothing, # 默认设为 nothing
                   colormap::Symbol = :viridis,       # 颜色映射,
                   filename::Union{String,Nothing} = nothing)
    
    fig = Figure(; size = (600, 500))
    
    # 逻辑优化：创建一个字典或者只在 title 不为 nothing 时传递该参数
    axis_kwargs = (xlabel = "x", ylabel = "y", aspect = DataAspect())
    if !isnothing(title)
        axis_kwargs = (title = title, axis_kwargs...)
    end
    
    ax = Axis(fig[1, 1]; axis_kwargs...)

    # Build physical coordinate axes
    x = range(0.0, cfg.Lx; length = cfg.Nx)
    y = range(0.0, cfg.Ly; length = cfg.Ny)

    hm = heatmap!(ax, x, y, field;
                  colormap  = colormap,
                  colorrange = (-1.0, 1.0))

    hidedecorations!(ax)
    hidespines!(ax)

    # 如果没有标题且 hidedecorations，空间利用更紧凑
    if isnothing(title)
        colgap!(fig.layout, 0)
    end

    isnothing(filename) || save(filename, fig)
    return fig
end

# 综合可视化函数：一次性展示所有主要场
function plot_all_fields(state::FieldState, cfg::Config, t::Float64;
                         filename::Union{String,Nothing} = nothing)
    x = range(0.0, cfg.Lx; length = cfg.Nx)
    y = range(0.0, cfg.Ly; length = cfg.Ny)

    fig = Figure(; size = (1000, 1000))
    Label(fig[0, 1:3], "t = $(round(t; digits=4))"; fontsize = 16)

    specs = [
        (state.phi, :berlin,   "φ (Phase 1)",   (-1.0,  1.0)),
        (state.psi, :berlin,   "ψ (Phase 2)",   (-1.0,  1.0)),
        (state.mu,  :viridis,  "μ (Chem. Pot)", :auto      ),
        (state.nu,  :viridis,  "ν (Chem. Pot)", :auto      ),
        (state.p,   :bwr,      "p (Pressure)",  :auto      ),
        (state.ux,  :coolwarm, "u_x (Velocity)",:auto      ),
    ]

    for (idx, (fld, cmap, ttl, clims)) in enumerate(specs)
        row, col = fld2rowcol(idx)   # see helper below
        ax = Axis(fig[row, col]; title = ttl, aspect = DataAspect())
        hm = heatmap!(ax, x, y, fld; colormap = cmap,
                      colorrange = clims === :auto ? extrema(fld) : clims)
        Colorbar(fig[row, col+1], hm; width = 12)   # narrow colorbars
    end

    isnothing(filename) || save(filename, fig)
    return fig
end

fld2rowcol(i) = (div(i-1, 3) + 1,  2*mod(i-1, 3) + 1)

# 专门的速度场可视化：背景显示速度大小，箭头显示方向
function plot_velocity(ux::Matrix{Float64}, uy::Matrix{Float64},
                       cfg::Config; stride::Int = 8)
    # Subsample — quiver on every point is unreadable
    xs = 1:stride:cfg.Nx
    ys = 1:stride:cfg.Ny
    x  = range(0.0, cfg.Lx; length = cfg.Nx)[xs]
    y  = range(0.0, cfg.Ly; length = cfg.Ny)[ys]

    fig = Figure(; size = (1000, 1000))
    ax  = Axis(fig[1,1]; title = "Velocity Field", aspect = DataAspect())

    # Background: speed magnitude
    speed = @. sqrt(ux^2 + uy^2)
    heatmap!(ax, range(0,cfg.Lx;length=cfg.Nx),
                 range(0,cfg.Ly;length=cfg.Ny),
                 speed; colormap = :gray, alpha = 0.5)

    # Arrows
    arrows!(ax, x, y, ux[xs, ys], uy[xs, ys];
            arrowsize = 8, lengthscale = 0.3)
    return fig
end

# 向量值绘图，一般是画能量和面积变化比例。
"""
    plot_vector(ys, ts;
                ylabel, title, filename,
                phase_switch, labels)

通用曲线图：将一条或多条时间序列 `ys` 画在同一个坐标系中。

# 参数
- `ys`            : 单条曲线时传 `Vector{Float64}`；多条曲线时传
                    `Vector{Vector{Float64}}`，每条长度须与 `ts` 一致。
- `ts`            : 时间轴向量（与仿真中的 `Dt` 切片对齐）。
- `ylabel`        : y 轴标签，默认 `"Value"`。
- `title`         : 图标题，默认 `nothing`（不显示）。
- `filename`      : 保存路径；`nothing` 时不保存。
- `phase_switch`  : 在此时刻画竖虚线（标记 warm-up/main 分界），
                    默认 `1e-6`；设为 `nothing` 关闭。
- `labels`        : 多条曲线的图例标签，`Vector{String}`，与 `ys` 等长。
                    单条曲线或不需要图例时传 `nothing`。
"""
function plot_vector(ys         :: Union{Vector{Float64}, Vector{<:Vector{Float64}}},
                     ts         :: AbstractVector{Float64};
                     ylabel     :: String                       = "Value",
                     title      :: Union{String, Nothing}       = nothing,
                     filename   :: Union{String, Nothing}       = nothing,
                     phase_switch :: Union{Float64, Nothing}    = 1e-6,
                     labels     :: Union{Vector{String}, Nothing} = nothing)

    # 统一处理为多条曲线
    curves = ys isa Vector{Float64} ? [ys] : ys
    t_col  = collect(ts)            # 确保是普通 Vector，兼容所有 AbstractVector

    fig = Figure(size = (750, 380))

    axis_kwargs = (xlabel = "t", ylabel = ylabel)
    if !isnothing(title)
        axis_kwargs = (title = title, axis_kwargs...)
    end
    ax = Axis(fig[1, 1]; axis_kwargs...)

    for (i, y) in enumerate(curves)
        lbl = (!isnothing(labels) && i ≤ length(labels)) ? labels[i] : nothing
        if isnothing(lbl)
            lines!(ax, t_col, y; linewidth = 1.5)
        else
            lines!(ax, t_col, y; linewidth = 1.5, label = lbl)
        end
    end

    # 图例（仅当有 label 时才显示）
    if !isnothing(labels)
        axislegend(ax; position = :rt)
    end

    # warm-up / main 分界竖线
    if !isnothing(phase_switch) && first(t_col) < phase_switch < last(t_col)
        vlines!(ax, [phase_switch];
                color     = :gray,
                linestyle = :dash,
                linewidth = 1,
                label     = "phase switch (t = $(phase_switch))")
    end

    isnothing(filename) || save(filename, fig)
    return fig
end


# ── visualize.jl ──────────────────────────────────────────────────────────────
# MATLAB-style theme：白底、黑框、Times 字体、网格线，与论文图表风格一致。
# 通过 set_theme! 全局生效，本文件内所有绘图函数自动继承，无需逐处传参。

const MATLAB_THEME = Theme(
    # 背景与字体
    backgroundcolor = :white,
    fontsize        = 12,
    fonts           = (; regular = "Times New Roman",
                         bold    = "Times New Roman Bold",
                         italic  = "Times New Roman Italic"),

    Axis = (
        # 坐标轴外框（四边都显示，类似 MATLAB box on）
        topspinevisible    = true,
        rightspinevisible  = true,
        topspinecolor      = :black,
        rightspinecolor    = :black,
        leftspinecolor     = :black,
        bottomspinecolor   = :black,

        # 刻度朝内
        xtickalign  = 1,
        ytickalign  = 1,
        xticksize   = 5,
        yticksize   = 5,

        # 网格线（浅灰，类似 MATLAB 默认网格）
        xgridvisible      = true,
        ygridvisible      = true,
        xgridcolor        = RGBAf(0, 0, 0, 0.12),
        ygridcolor        = RGBAf(0, 0, 0, 0.12),
        xgridstyle        = :solid,
        ygridstyle        = :solid,

        # 背景
        backgroundcolor   = :white,
    ),

    # 默认线条：MATLAB 蓝 #0072BD，宽度 1.5
    Lines = (
        color     = RGBf(0/255, 114/255, 189/255),
        linewidth = 1.5,
    ),

    # 图例样式
    Legend = (
        framevisible  = true,
        framecolor    = :black,
        bgcolor       = :white,
        padding       = (6f0, 6f0, 4f0, 4f0),
    ),
)

# 在文件加载时立即生效，整个会话内所有图都使用此主题
set_theme!(MATLAB_THEME)