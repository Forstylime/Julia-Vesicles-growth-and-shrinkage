
# 单个场的可视化函数
function plot_field(field::Matrix{Float64}, cfg::Config;
    title::Union{String,Nothing}=nothing, # 默认设为 nothing
    colormap::Symbol=:viridis, # 颜色映射,
    filename::Union{String,Nothing}=nothing)

    fig = Figure(; size=(600, 500))
    axis_kwargs = (xlabel="x", ylabel="y", aspect=DataAspect())
    if !isnothing(title)
        axis_kwargs = (title=title, axis_kwargs...)
    end
    ax = Axis(fig[1, 1]; axis_kwargs...)

    x = range(0.0, cfg.Lx; length=cfg.Nx)
    y = range(0.0, cfg.Ly; length=cfg.Ny)

    hm = heatmap!(ax, x, y, field;
        colormap=colormap,
        interpolate=true,
        colorrange=(-0.5, 0.5))

    #Colorbar(fig[1, 2], hm)
    hidedecorations!(ax)
    hidespines!(ax)

    isnothing(filename) || save(filename, fig)
    return fig
end

# 综合可视化函数：一次性展示所有主要场
function plot_all_fields(state::FieldState, cfg::Config, t::Float64;
    filename::Union{String,Nothing}=nothing)
    x = range(0.0, cfg.Lx; length=cfg.Nx)
    y = range(0.0, cfg.Ly; length=cfg.Ny)

    fig = Figure(; size=(1000, 1000))
    Label(fig[0, 1:3], "t = $(round(t; digits=4))"; fontsize=16)

    specs = [
        (state.phi, :berlin, "φ (Phase 1)", (-1.0, 1.0)),
        (state.psi, :berlin, "ψ (Phase 2)", (-1.0, 1.0)),
        (state.mu, :viridis, "μ (Chem. Pot)", :auto),
        (state.nu, :viridis, "ν (Chem. Pot)", :auto),
        (state.p, :bwr, "p (Pressure)", :auto),
        (state.ux, :coolwarm, "u_x (Velocity)", :auto),
    ]

    for (idx, (fld, cmap, ttl, clims)) in enumerate(specs)
        row, col = fld2rowcol(idx)   # see helper below
        ax = Axis(fig[row, col]; title=ttl, aspect=DataAspect())
        hm = heatmap!(ax, x, y, fld; colormap=cmap,
            colorrange=clims === :auto ? extrema(fld) : clims)
        Colorbar(fig[row, col+1], hm; width=12)   # narrow colorbars
    end

    isnothing(filename) || save(filename, fig)
    return fig
end

fld2rowcol(i) = (div(i - 1, 3) + 1, 2 * mod(i - 1, 3) + 1)

# 专门的速度场可视化：背景显示速度大小，箭头显示方向
function plot_velocity(ux::Matrix{Float64}, uy::Matrix{Float64},
    cfg::Config; stride::Int=8)
    # Subsample — quiver on every point is unreadable
    xs = 1:stride:cfg.Nx
    ys = 1:stride:cfg.Ny
    x = range(0.0, cfg.Lx; length=cfg.Nx)[xs]
    y = range(0.0, cfg.Ly; length=cfg.Ny)[ys]

    fig = Figure(; size=(1000, 1000))
    ax = Axis(fig[1, 1]; title="Velocity Field", aspect=DataAspect())

    # Background: speed magnitude
    speed = @. sqrt(ux^2 + uy^2)
    heatmap!(ax, range(0, cfg.Lx; length=cfg.Nx),
        range(0, cfg.Ly; length=cfg.Ny),
        speed; colormap=:gray, alpha=0.5)

    # Arrows
    arrows!(ax, x, y, ux[xs, ys], uy[xs, ys];
        arrowsize=8, lengthscale=0.3)
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
function plot_vector(ys::Union{Vector{Float64},Vector{<:Vector{Float64}}},
    ts::AbstractVector{Float64};
    ylabel::String="Value",
    title::Union{String,Nothing}=nothing,
    filename::Union{String,Nothing}=nothing,
    phase_switch::Union{Float64,Nothing}=nothing,
    labels::Union{Vector{String},Nothing}=nothing)

    # 统一处理为多条曲线
    curves = ys isa Vector{Float64} ? [ys] : ys
    t_col = collect(ts)            # 确保是普通 Vector，兼容所有 AbstractVector

    fig = Figure(size=(750, 480))

    axis_kwargs = (xlabel="t", ylabel=ylabel)
    if !isnothing(title)
        axis_kwargs = (title=title, axis_kwargs...)
    end
    ax = Axis(fig[1, 1]; axis_kwargs...)

    for (i, y) in enumerate(curves)
        lbl = (!isnothing(labels) && i ≤ length(labels)) ? labels[i] : nothing
        if isnothing(lbl)
            lines!(ax, t_col, y; linewidth=2)
        else
            lines!(ax, t_col, y; linewidth=2, label=lbl)
        end
    end

    # 图例（仅当有 label 时才显示）
    if !isnothing(labels)
        axislegend(ax; position=:rt)
    end

    # warm-up / main 分界竖线
    if !isnothing(phase_switch) && first(t_col) < phase_switch < last(t_col)
        vlines!(ax, [phase_switch];
            color=:gray,
            linestyle=:dash,
            linewidth=1,
            label="phase switch (t = $(phase_switch))")
    end

    isnothing(filename) || save(filename, fig)
    return fig
end


# ─── 三维 isosurface 可视化 ───────────────────────────────────────────────────
using Meshing
using GeometryBasics

"""
    plot_iso(field, cfg; isovalue, color, alpha, title, filename)

对 `Array{Float64,3}` 的三维标量场提取 **连续封闭等值面** 并渲染。

# 实现要点
- 使用 `MarchingTetrahedra` 算法，保证输出网格为流形（manifold），
  即每个顶点唯一、面片形成连续封闭曲面。
- 通过 X, Y, Z 范围参数直接将顶点生成在物理坐标系中，
  无需手动坐标变换。
"""
function plot_iso(field::Array{Float64,3}, cfg;
    isovalue::Float64=0.0,
    color=:steelblue,
    alpha::Float64=0.85,
    title::Union{String,Nothing}=nothing,
    filename::Union{String,Nothing}=nothing)

    # ── 1. MarchingTetrahedra → (verts, faces) ───────────────────
    #    eps 参数控制体素角点附近的容差，保证流形网格生成
    #    X, Y, Z 范围参数将顶点直接映射到物理坐标系
    algo = MarchingTetrahedra(iso=isovalue, eps=1e-6)
    verts_raw, faces_raw = isosurface(field, algo,
        0.0:cfg.Lx, 0.0:cfg.Ly, 0.0:cfg.Lz)

    if isempty(verts_raw)
        @warn "plot_iso: 等值面为空，isovalue=$isovalue 超出数据范围 $(extrema(field))"
        return Figure()
    end

    # ── 2. 转为 GeometryBasics 类型 ──────────────────────────────
    #    顶点已在物理坐标系下，无需额外变换
    verts = [Point3f(v...) for v in verts_raw]
    faces = [TriangleFace(f...) for f in faces_raw]
    mesh_phys = GeometryBasics.Mesh(verts, faces)

    # ── 3. 渲染 ──────────────────────────────────────────────────
    fig = Figure(size=(700, 650))
    ax_kw = (xlabel="x", ylabel="y", zlabel="z",
        aspect=:equal,
        perspectiveness=0.4f0,
        azimuth=1.275π,
        elevation=0.2π)
    !isnothing(title) && (ax_kw = (title=title, ax_kw...))
    ax = Axis3(fig[1, 1]; ax_kw...)

    mesh!(ax, mesh_phys;
        color=color,
        alpha=Float32(alpha),
        transparency=alpha < 1.0,
        shading=true)

    if alpha < 0.95
        wireframe!(ax, mesh_phys;
            color=(:black, 0.06),
            linewidth=0.3)
    end

    isnothing(filename) || save(filename, fig)
    return fig
end