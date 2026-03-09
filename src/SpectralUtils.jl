# File: src/SpectralUtils.jl
#
# Builds spectral derivative operators and FFTW plans.
# Key design decisions vs. the MATLAB version:
#   - rfft/irfft instead of fft/ifft: exploits real-valuedness of all
#     physical fields, giving (Nx÷2+1)×Ny spectral arrays (~half storage)
#   - Wavenumber arrays kept as broadcast-compatible vectors, not full
#     matrices: kx is (Nx÷2+1)×1, ky is 1×Ny. Operations like kx.^2 .+ ky.^2
#     produce (Nx÷2+1)×Ny results without storing intermediate full arrays.
#   - Dealiasing (3/2 rule) plans are precomputed on the padded grid and
#     stored in Operators, so dealias_mul! has zero setup cost at runtime.

using FFTW

# ============================================================
#  OPERATOR CONSTRUCTION
# ============================================================

"""
    build_operators(cfg::Config) -> Operators

Precompute all spectral derivative operators and FFTW plans from `cfg`.
Called once before the time loop.

# Wavenumber layout (rfft convention)
For an Nx×Ny real field:
  - kx lives on the *rfft* frequency axis: indices 0, 1, ..., Nx÷2
    giving shape (Nx÷2+1, 1)  — a column vector
  - ky lives on the full frequency axis: 0, 1, ..., Ny÷2, -Ny÷2+1, ..., -1
    giving shape (1, Ny)       — a row vector
  - Any operator on the (Nx÷2+1)×Ny spectral grid is formed by broadcasting,
    e.g. kx.^2 .+ ky.^2 produces (Nx÷2+1)×Ny with no extra allocation.

# Sign convention
  D1[1] = i*kx,  D1[2] = i*ky      (first derivatives)
  Laplacian  = -(kx² + ky²)         (negative semi-definite)
  Biharmonic =  (kx² + ky²)²        (positive semi-definite)
"""
function build_operators(cfg::Config)
    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly

    # ----------------------------------------------------------
    # 1. Wavenumber vectors (rfft layout)
    #    rfftfreq: returns 0, 1/n, 2/n, ..., 1/2  (only non-redundant half)
    #    fftfreq:  returns 0, 1/n, ..., 1/2, -1/2+1/n, ..., -1/n
    #    Multiply by 2π and N/L to convert to rad/unit wavenumbers.
    #
    #    Shape is deliberately (Nx÷2+1, 1) and (1, Ny) for broadcasting.
    # ---------------------------------------------------------- 
    kx = convert(Matrix{Float64}, reshape(2π .* rfftfreq(Nx, Nx / Lx), :, 1))   # (Nx÷2+1) × 1
    ky = convert(Matrix{Float64}, reshape(2π .* fftfreq(Ny, Ny / Ly), 1, :))    # 1 × Ny

    K = (kx, ky)::NTuple{2, Matrix{Float64}}  # NTuple{2}: index as K[1], K[2]

    # ----------------------------------------------------------
    # 2. First-derivative operators in spectral space
    #    Stored as ComplexF64 broadcast vectors.
    #    Multiplying a spectral field u_hat by D1[1] gives ∂u/∂x in
    #    spectral space (pointwise multiply, then irfft for physical space).
    # ----------------------------------------------------------
    D1 = (complex.(0.0, kx), complex.(0.0, ky))  # (i*kx, i*ky)

    # ----------------------------------------------------------
    # 3. Laplacian and Biharmonic operators
    #    K2 = kx² + ky²  shape: (Nx÷2+1) × Ny  (via broadcast, no temp alloc)
    #    These are real-valued (no imaginary part needed).
    # ----------------------------------------------------------
    K2         = @. kx^2 + ky^2              # (Nx÷2+1) × Ny, Float64
    Laplacian        = @. -K2                      # negative semi-definite
    Biharmonic        = @. (kx^2 + ky^2)^2                    # positive semi-definite

    # ----------------------------------------------------------
    # 4. FFTW plans on the PHYSICAL grid (Nx × Ny)
    #    plan_rfft:  real (Nx×Ny) → complex (Nx÷2+1 × Ny)
    #    plan_irfft: complex (Nx÷2+1 × Ny) → real (Nx×Ny)
    #                needs Nx as the explicit output size
    #
    #    FFTW.MEASURE: benchmarks several algorithms at plan creation time
    #    and picks the fastest. Startup cost ~seconds; per-call cost minimal.
    #    Use FFTW.ESTIMATE during development if startup time is annoying.
    # ----------------------------------------------------------
    tmp_real    = zeros(Float64,    Nx, Ny)
    tmp_complex = zeros(ComplexF64, Nx÷2+1, Ny)

    fft_plan  = plan_rfft(tmp_real;           flags=FFTW.MEASURE)
    ifft_plan = plan_irfft(tmp_complex, Nx;   flags=FFTW.MEASURE)

    return Operators(
    K, D1,                          # NTuple, NTuple
    Laplacian, Biharmonic,          # Matrix, Matrix
    fft_plan,  ifft_plan,           # P,  IP   (物理网格)
    Nx, Ny                          # Int, Int
)
end


# ============================================================
#  SPECTRAL ↔ PHYSICAL TRANSFORMS  (thin wrappers)
# ============================================================

"""
    to_spectral!(u_hat, u, ops)

In-place forward transform: physical field `u` (Nx×Ny, real)
→ spectral field `u_hat` ((Nx÷2+1)×Ny, complex).
Writes result into pre-allocated `u_hat`.
"""
@inline function to_spectral!(u_hat::AbstractMatrix{ComplexF64},
                               u::AbstractMatrix{Float64},
                               ops::Operators)
    u_hat .= ops.fft_plan * u
    return u_hat
end

"""
    to_physical!(u, u_hat, ops)

In-place inverse transform: spectral field `u_hat` ((Nx÷2+1)×Ny)
→ physical field `u` (Nx×Ny, real).
Writes result into pre-allocated `u`.
"""
@inline function to_physical!(u::AbstractMatrix{Float64},
                               u_hat::AbstractMatrix{ComplexF64},
                               ops::Operators)
    u .= ops.ifft_plan * u_hat
    return u
end


# ============================================================
#  DEALIASED MULTIPLICATION  (3/2 rule)
# ============================================================

"""
谱系数点乘函数，不用去混叠
"""
function multpl!(
    w_hat::AbstractMatrix{ComplexF64},
    u_hat::AbstractMatrix{ComplexF64},
    v_hat::AbstractMatrix{ComplexF64},
    ops::Operators
)

Nx = ops.Nx
Ny = ops.Ny

    u = zeros(Float64,Nx,Ny)
    v = zeros(Float64,Nx,Ny)
    w = zeros(Float64,Nx,Ny)

    to_physical!(u,u_hat,ops)
    to_physical!(v,v_hat,ops)

    @. w = u*v

    to_spectral!(w_hat,w,ops)

    return w_hat
end