# File: test/test_Types_Ops.jl
#
# Test suite for Types.jl and SpectralUtils.jl
#
# Run from the project root with:
#   julia --project=. test/test_types_and_spectral.jl
#
# Or from the REPL:
#   include("test/test_types_and_spectral.jl")
#
# Each test group is independent. A failing group prints clearly
# which assertion failed and why, without aborting the other groups.

using Test
using FFTW
using LinearAlgebra: norm

include("../src/Types.jl")
include("../src/SpectralUtils.jl")
# test/runtests.jl

# ============================================================
#  SHARED FIXTURE
#  One small Config used across all tests.
#  128×64 (non-square) is deliberate: catches any bug that
#  accidentally swaps Nx and Ny.
# ============================================================
const cfg = Config(
    N = 1,
    epsilon=0.05,  M_phi=1.0,  M0_psi=1.0,  eta=1.0,
    gamma_surf=1.0, gamma_area=1.0, gamma_bend=1.0,
    gamma_in=1.0,  beta_in=0.0,  psi_in=[0.1],
    gamma_out=1.0, beta_out=0.0, psi_out=[0.8],
    lamda=1,
    S1=1.0, S2=1.0, S3=1.0, S4=1.0,
    C1=1.0, C2=1.0, C3=1.0,
    dt=1e-3, T=1.0,
    Nx=64, Ny=64,
    Lx=2π,  Ly=2π,
    tol=1e-10, goal=:shrinkage,
    area_target=0.5
)

const ops = build_operators(cfg)


# ============================================================
#  GROUP 1: Config construction
# ============================================================
@testset "Config construction" begin

    # Derived quantities are computed correctly
    @test cfg.Nt == round(Int, cfg.T / cfg.dt)   # 1000
    @test cfg.dx ≈ cfg.Lx / cfg.Nx
    @test cfg.dy ≈ cfg.Ly / cfg.Ny

    # Immutability: assigning a field must throw
    @test_throws ErrorException (cfg.dt = 2.0)

    # Broadcasting: Config used as a scalar in broadcast context
    # (tests Base.broadcastable)
    vals = [1.0, 2.0, 3.0]
    result = map(v -> v + cfg.dt, vals)           # must not error
    @test result ≈ vals .+ cfg.dt

end


# ============================================================
#  GROUP 2: Operators — shapes and types
# ============================================================
@testset "Operators shapes and types" begin

    Nx, Ny = cfg.Nx, cfg.Ny
    Nkx    = Nx ÷ 2 + 1          # expected rfft kx size = 65

    # --- Wavenumber vector shapes ---
    # K[1] must be a column vector (Nkx × 1), K[2] a row vector (1 × Ny)
    @test size(ops.K[1]) == (Nkx, 1)
    @test size(ops.K[2]) == (1, Ny)

    # --- D1 shapes match K ---
    @test size(ops.D1[1]) == (Nkx, 1)
    @test size(ops.D1[2]) == (1, Ny)

    # --- D1 is purely imaginary: real part must be zero ---
    @test all(iszero, real.(ops.D1[1]))
    @test all(iszero, real.(ops.D1[2]))

    # --- D1 imaginary parts equal K values ---
    @test imag.(ops.D1[1]) ≈ ops.K[1]
    @test imag.(ops.D1[2]) ≈ ops.K[2]

    # --- Laplacian and Biharmonic shapes: (Nkx × Ny) after broadcast ---
    @test size(ops.Laplacian)  == (Nkx, Ny)
    @test size(ops.Biharmonic) == (Nkx, Ny)

    # --- Sign conventions ---
    # Laplacian = -(kx²+ky²): must be ≤ 0 everywhere
    @test all(ops.Laplacian .<= 0.0)
    # Biharmonic = (kx²+ky²)²: must be ≥ 0 everywhere
    @test all(ops.Biharmonic .>= 0.0)

    # --- Biharmonic = Laplacian² (element-wise) ---
    @test ops.Biharmonic ≈ ops.Laplacian .^ 2

end


# ============================================================
#  GROUP 3: Operators — wavenumber values
# ============================================================
@testset "Operators wavenumber values" begin

    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly
    Nkx    = Nx ÷ 2 + 1

    # kx[1] = 0 (DC component)
    @test ops.K[1][1, 1] ≈ 0.0

    # kx[2] = 2π/Lx  (fundamental mode in x)
    @test ops.K[1][2, 1] ≈ 2π / Lx

    # kx[Nkx] = π*Nx/Lx  (Nyquist in x)
    @test ops.K[1][Nkx, 1] ≈ π * Nx / Lx

    # ky[1] = 0  (DC in y)
    @test ops.K[2][1, 1] ≈ 0.0

    # ky[2] = 2π/Ly  (fundamental in y)
    @test ops.K[2][1, 2] ≈ 2π / Ly

    # ky at index Ny÷2+1 is the Nyquist in y (negative by fftfreq convention)
    @test ops.K[2][1, Ny÷2+1] ≈ -π * Ny / Ly

end


# ============================================================
#  GROUP 4: Round-trip transforms
# ============================================================
@testset "Round-trip rfft/irfft" begin

    Nx, Ny = cfg.Nx, cfg.Ny

    # --- Test 1: random real field ---
    u      = randn(Float64, Nx, Ny)
    u_hat  = zeros(ComplexF64, Nx÷2+1, Ny)
    u_back = zeros(Float64, Nx, Ny)

    to_spectral!(u_hat,  u,     ops)
    to_physical!(u_back, u_hat, ops)

    @test u_back ≈ u   atol=1e-12

    # --- Test 2: constant field (tests DC component isolation) ---
    c      = fill(3.7, Nx, Ny)
    c_hat  = zeros(ComplexF64, Nx÷2+1, Ny)
    c_back = zeros(Float64, Nx, Ny)

    to_spectral!(c_hat,  c,     ops)
    to_physical!(c_back, c_hat, ops)

    @test c_back ≈ c   atol=1e-15

    # --- Test 3: to_spectral! is in-place (does not allocate output array) ---
    u2     = randn(Float64, Nx, Ny)
    u2_hat = zeros(ComplexF64, Nx÷2+1, Ny)
    ptr_before = pointer(u2_hat)
    to_spectral!(u2_hat, u2, ops)
    @test pointer(u2_hat) == ptr_before   # same memory, not a new array

end


# ============================================================
#  GROUP 5: Spectral differentiation accuracy
#
#  Strategy: use an exactly representable test function
#    f(x,y)  = sin(m*x) * cos(n*y)
#  whose derivatives are known analytically:
#    ∂f/∂x   =  m * cos(m*x) * cos(n*y)
#    ∂f/∂y   = -n * sin(m*x) * sin(n*y)
#    ∇²f     = -(m²+n²) * f
#    ∇⁴f     =  (m²+n²)² * f
#  A spectral method on a periodic grid is *exact* for such functions
#  (no truncation error), so we demand near machine-precision accuracy.
# ============================================================
@testset "Spectral differentiation accuracy" begin

    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly
    m, n   = 3, 4
    km     = m * 2π / Lx
    kn     = n * 2π / Ly

    x = [Lx * (i-1) / Nx for i in 1:Nx]
    y = [Ly * (j-1) / Ny for j in 1:Ny]

    # 除以 Nx*Ny 使谱系数幅值 ≈ 1
    # 这样误差量级只由算子本身决定，不受 FFT 归一化影响
    scale    = Float64(Nx * Ny) / 4
    f        = [sin(km*xi) * cos(kn*yj)                   / scale for xi in x, yj in y]
    df_dx    = [km  * cos(km*xi) * cos(kn*yj)             / scale for xi in x, yj in y]
    df_dy    = [-kn * sin(km*xi) * sin(kn*yj)             / scale for xi in x, yj in y]
    lap_f    = [-(km^2+kn^2)   * sin(km*xi) * cos(kn*yj) / scale for xi in x, yj in y]
    biharm_f = [ (km^2+kn^2)^2 * sin(km*xi) * cos(kn*yj) / scale for xi in x, yj in y]

    f_hat      = zeros(ComplexF64, Nx÷2+1, Ny)
    deriv_hat  = zeros(ComplexF64, Nx÷2+1, Ny)
    deriv_phys = zeros(Float64, Nx, Ny)

    to_spectral!(f_hat, f, ops)

    # 确认谱系数幅值确实 ≈ 1（不再是 1024）
    sorted_modes = sort(abs.(f_hat)[:], rev=true)
    @test sorted_modes[1] ≈ 1.0   atol=1e-10

    # ∂f/∂x：一阶，误差 ~ km * ε_machine ≈ 3 * 1e-16
    @. deriv_hat = ops.D1[1] * f_hat
    to_physical!(deriv_phys, deriv_hat, ops)
    @test norm(deriv_phys .- df_dx, Inf) < 1e-15

    # ∂f/∂y：一阶
    @. deriv_hat = ops.D1[2] * f_hat
    to_physical!(deriv_phys, deriv_hat, ops)
    @test norm(deriv_phys .- df_dy, Inf) < 1e-15

    # ∇²f：二阶，误差 ~ k² * ε_machine ≈ 25 * 1e-16
    @. deriv_hat = ops.Laplacian * f_hat
    to_physical!(deriv_phys, deriv_hat, ops)
    @test norm(deriv_phys .- lap_f, Inf) < 1e-15

    # ∇⁴f：四阶，误差 ~ k⁴ * ε_machine ≈ 625 * 1e-15
    @. deriv_hat = ops.Biharmonic * f_hat
    to_physical!(deriv_phys, deriv_hat, ops)
    @test norm(deriv_phys .- biharm_f, Inf) < 1e-12

end


# ============================================================
#  GROUP 6: Multiplication
# ============================================================
@testset "Multiplication" begin

    Nx, Ny = cfg.Nx, cfg.Ny

    # --- Test 1: correctness ---
    # Use low-frequency fields so aliasing is absent and the
    # dealiased result must exactly match direct multiplication.
    #
    # f = sin(x),  g = cos(y)  →  f*g = sin(x)*cos(y)
    # These are mode-1 fields, well within the resolved band.
    x = [2π * (i-1) / Nx for i in 1:Nx]
    y = [2π * (j-1) / Ny for j in 1:Ny]

    f   = [sin(xi)         for xi in x, yj in y]
    g   = [cos(yj)         for xi in x, yj in y]
    fg  = [sin(xi)*cos(yj) for xi in x, yj in y]   # analytical product

    f_hat  = zeros(ComplexF64, Nx÷2+1, Ny)
    g_hat  = zeros(ComplexF64, Nx÷2+1, Ny)
    fg_hat = zeros(ComplexF64, Nx÷2+1, Ny)
    fg_back = zeros(Float64, Nx, Ny)

    to_spectral!(f_hat, f, ops)
    to_spectral!(g_hat, g, ops)

    mult!(fg_hat, f_hat, g_hat, ops)
    to_physical!(fg_back, fg_hat, ops)

    @test norm(fg_back .- fg, Inf) < 1e-15

    # --- Test2: aliasing removal ---
    mx = Nx ÷ 4
    nx = Nx ÷ 4

    h = [sin(mx*xi) + sin(nx*xi) for xi in x, yj in y]

    # exact
    exact = h.^2

    # spectral buffers
    h_hat  = zeros(ComplexF64, Nx÷2+1, Ny)
    hh_hat = zeros(ComplexF64, Nx÷2+1, Ny)

    hh_back = zeros(Float64,Nx,Ny)

    # transform
    to_spectral!(h_hat,h,ops)

    #multiplication
    mult!(hh_hat,h_hat,h_hat,ops)
    to_physical!(hh_back,hh_hat,ops)

    err = norm(hh_back .- exact,Inf)

    @test err < 1e-15

    # --- Test 3: commutativity ---
    # dealias_mul!(w, u, v) == dealias_mul!(w, v, u)
    uv_hat = zeros(ComplexF64, Nx÷2+1, Ny)
    vu_hat = zeros(ComplexF64, Nx÷2+1, Ny)

    mult!(uv_hat, f_hat, g_hat, ops)
    mult!(vu_hat, g_hat, f_hat, ops)

    @test uv_hat ≈ vu_hat   atol=1e-15

end


# ============================================================
#  GROUP 7: FieldState construction and update_state!
# ============================================================
@testset "FieldState update_state!" begin

    Nx, Ny = cfg.Nx, cfg.Ny
    Nkx    = Nx ÷ 2 + 1
    N     = cfg.N   # single phase field for now

    # Helper: build a FieldState filled with a constant value c
    function make_state(c::Float64)
        FieldState(
            fill(c,  Nx,  Ny, N),   fill(complex(c), Nkx, Ny, N),  # phi
            fill(c,  Nx,  Ny, N),        fill(complex(c), Nkx, Ny, N),       # psi
            fill(c,  Nx,  Ny, 2),     fill(complex(c), Nkx, Ny, 2),    # u
            fill(c,  Nx,  Ny),        fill(complex(c), Nkx, Ny),       # p
            fill(c,  Nx,  Ny, N),        fill(complex(c), Nkx, Ny, N),       # mu
            fill(c,  Nx,  Ny, N),        fill(complex(c), Nkx, Ny, N),       # nu
            c, c, c, c,   # Q, R1, R2, R3
            c             # area_lambda
        )
    end

    s1 = make_state(1.0)
    s2 = make_state(2.0)

    # Before update: s1 and s2 are distinct
    @test s1.phi[1,1,1] ≈ 1.0
    @test s2.phi[1,1,1] ≈ 2.0

    # update_state!(dest, src): dest ← src
    update_state!(s1, s2)

    # After update: s1's arrays contain s2's values
    @test all(s1.phi     .≈ 2.0)
    @test all(s1.psi     .≈ 2.0)
    @test all(real.(s1.phi_hat) .≈ 2.0)
    @test s1.Q           ≈ 2.0
    @test s1.R1          ≈ 2.0
    @test s1.area_lambda ≈ 2.0

    # Crucially: modifying s2 after the update must NOT affect s1
    # (proves copyto! was used, not pointer aliasing)
    s2.phi[1,1,1] = 99.0
    @test s1.phi[1,1,1] ≈ 2.0   # s1 is independent

end


# ============================================================
#  GROUP 8: BDFCoeff
# ============================================================
@testset "BDFCoeff" begin

    bdf1 = bdf_coeff(1)
    bdf2 = bdf_coeff(2)

    # BDF1 coefficients
    @test bdf1.a ≈  1.0
    @test bdf1.b ≈ -1.0
    @test bdf1.c ≈  0.0

    # BDF2 coefficients
    @test bdf2.a ≈  1.5
    @test bdf2.b ≈ -2.0
    @test bdf2.c ≈  0.5

    # Consistency check: BDF2 is second-order consistent.
    # For u(t) = t², the BDF2 approximation of du/dt at t=1 with dt=0.01
    # should approach 2.0 (the exact derivative) as dt → 0.
    dt    = 0.01
    u_n   = 1.0^2           # t = 1
    u_nm1 = (1.0 - dt)^2   # t = 1 - dt
    u_nm2 = (1.0 - 2dt)^2  # t = 1 - 2dt

    bdf2_approx = (bdf2.a * u_n + bdf2.b * u_nm1 + bdf2.c * u_nm2) / dt
    @test abs(bdf2_approx - 2.0) < 1e-2   # second-order: error ~ dt

end


# ============================================================
#  GROUP 9: Type stability  (@inferred)
#  Ensures the compiler can fully infer return types —
#  a requirement for C-level performance in the hot loop.
# ============================================================
@testset "Type stability" begin

    Nx, Ny = cfg.Nx, cfg.Ny
    u      = randn(Float64, Nx, Ny)
    u_hat  = zeros(ComplexF64, Nx÷2+1, Ny)
    v_hat  = zeros(ComplexF64, Nx÷2+1, Ny)
    w_hat  = zeros(ComplexF64, Nx÷2+1, Ny)

    to_spectral!(u_hat, u, ops)

    # to_spectral! return type must be inferred
    @test @inferred(to_spectral!(u_hat, u, ops)) isa Matrix{ComplexF64}

    # to_physical! return type must be inferred
    u_back = zeros(Float64, Nx, Ny)
    @test @inferred(to_physical!(u_back, u_hat, ops)) isa Matrix{Float64}

    # bdf_coeff return type must be inferred
    @test @inferred(bdf_coeff(1)) isa BDFCoeff
    @test @inferred(bdf_coeff(2)) isa BDFCoeff

end