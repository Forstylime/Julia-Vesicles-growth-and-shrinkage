# 单个场的可视化函数
function plot_field(field::Matrix{Float64}, cfg::Config;
                   title::Union{String,Nothing} = nothing, # 默认设为 nothing
                   colormap::Symbol = :RdBu,
                   filename::Union{String,Nothing} = nothing)
    
    fig = Figure(; size = (500, 500))
    
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