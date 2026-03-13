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
    N = cfg.N

    # ----------------------------------------------------------
    # 1. Wavenumber vectors (rfft layout)
    #    rfftfreq: returns 0, 1/n, 2/n, ..., 1/2  (only non-redundant half)
    #    fftfreq:  returns 0, 1/n, ..., 1/2, -1/2+1/n, ..., -1/n
    #    Multiply by 2π and N/L to convert to rad/unit wavenumbers.
    #
    #    Shape is deliberately (Nx÷2+1, 1) and (1, Ny) for broadcasting.
    # ---------------------------------------------------------- 
    kx = convert(Matrix{Float64}, reshape(2π .* rfftfreq(Nx, Nx / Lx), :, 1))   # (Nx÷2+1) × 1
    kx0 = convert(Matrix{Float64}, reshape(2π .* fftfreq(Nx, Nx / Lx), Nx, 1)) # full fft frequencies
    ky = convert(Matrix{Float64}, reshape(2π .* fftfreq(Ny, Ny / Ly), 1, :))    # 1 × Ny

    K = (kx, ky)::NTuple{2, Matrix{Float64}}  # NTuple{2}: index as K[1], K[2]
    K0 = (kx0, ky)  # full fft frequencies

    # ----------------------------------------------------------
    # 2. First-derivative operators in spectral space
    #    Stored as ComplexF64 broadcast vectors.
    #    Multiplying a spectral field u_hat by D1[1] gives ∂u/∂x in
    #    spectral space (pointwise multiply, then irfft for physical space).
    # ----------------------------------------------------------
    D1 = (complex.(0.0, kx), complex.(0.0, ky))  # (i*kx, i*ky)
    D1_full = (complex.(0.0, kx0), complex.(0.0, ky))  # full fft frequencies

    # ----------------------------------------------------------
    # 3. Laplacian and Biharmonic operators
    #    K2 = kx² + ky²  shape: (Nx÷2+1) × Ny  (via broadcast, no temp alloc)
    #    These are real-valued (no imaginary part needed).
    # ----------------------------------------------------------
    K2              = @. kx^2 + ky^2              # (Nx÷2+1) × Ny, Float64
    Laplacian       = @. -K2                      # negative semi-definite
    Laplacian_full  = @. -(kx0^2 + ky^2)      # full fft frequencies
    Biharmonic      = @. (kx^2 + ky^2)^2          # positive semi-definite

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
    tmp_real    = zeros(Float64,    Nx, Ny, N)
    tmp_complex = zeros(ComplexF64, Nx÷2+1, Ny, N)
    tmp_real0    = zeros(Float64,    Nx, Ny, N)
    tmp_complex0 = zeros(ComplexF64, Nx, Ny, N) # for full fft/ifft plans if needed in BiCGSTAB
    tmp_real_1    = zeros(Float64,    Nx, Ny)
    tmp_complex_1 = zeros(ComplexF64, Nx÷2+1, Ny)
    tmp_real_2    = zeros(Float64,    Nx, Ny, 2)
    tmp_complex_2 = zeros(ComplexF64, Nx÷2+1, Ny, 2)

    fft_plan  = plan_rfft(tmp_real,         (1, 2); flags=FFTW.MEASURE) # for phase fields (N components)
    ifft_plan = plan_irfft(tmp_complex, Nx, (1, 2); flags = FFTW.MEASURE) # for phase fields (N components)

    fft_plan0   = plan_fft(tmp_real0,         (1, 2); flags=FFTW.MEASURE) # full fft plan
    ifft_plan0  = plan_ifft(tmp_complex0,     (1, 2); flags=FFTW.MEASURE) # full ifft plan
    fft_plan_1  = plan_rfft(tmp_real_1,         (1, 2); flags=FFTW.MEASURE) # for scalar fields
    ifft_plan_1 = plan_irfft(tmp_complex_1, Nx, (1, 2); flags=FFTW.MEASURE) # for scalar fields
    fft_plan_2  = plan_rfft(tmp_real_2,         (1, 2); flags=FFTW.MEASURE) # for vector fields (2 components)
    ifft_plan_2 = plan_irfft(tmp_complex_2, Nx, (1, 2); flags=FFTW.MEASURE) # for vector fields (2 components)

    inv_s = 1.0 / (Nx * Ny)

    # ----------------------------------------------------------
    # 5. Temporary buffers for spectral-physical transforms and multiplications
    # ----------------------------------------------------------
    temp_real1 = zeros(Float64, Nx, Ny, N)          # for physical fields
    temp_real2 = zeros(Float64, Nx, Ny, N)          # for physical fields
    temp_real3 = zeros(Float64, Nx, Ny, N)          # for physical fields
    temp_comp1 = zeros(ComplexF64, Nx÷2+1, Ny, N)  # for spectral fields
    temp_comp2 = zeros(ComplexF64, Nx÷2+1, Ny, N)  # for spectral fields
    temp_comp3 = zeros(ComplexF64, Nx÷2+1, Ny, N)  # for spectral fields
    temp_complex_irfft = zeros(ComplexF64, Nx÷2+1, Ny, N)  # for in-place irfft

    return Operators(
    K, D1, D1_full,                         # NTuple, NTuple
    Laplacian, Laplacian_full, Biharmonic,          # Matrix, Matrix
    fft_plan,  ifft_plan,           # P,  IP   (物理网格)
    fft_plan0, ifft_plan0,
    fft_plan_1,  ifft_plan_1,
    fft_plan_2,  ifft_plan_2,
    Nx, Ny,                          # Int, Int
    inv_s,     # Float64
    temp_real1, temp_real2, temp_real3,
    temp_comp1, temp_comp2, temp_comp3,
    temp_complex_irfft
    )
end


# ============================================================
#  SPECTRAL ↔ PHYSICAL TRANSFORMS  (thin wrappers)
# ============================================================

"""
    to_spectral(u, ops)

In-place forward transform: physical field `u` (Nx×Ny, real)
→ spectral field `u_hat` ((Nx÷2+1)×Ny, complex).
Writes result into pre-allocated `u_hat`.
"""
@inline function to_spectral!(buffer::AbstractArray{ComplexF64}, u::AbstractArray{Float64}, ops::Operators)
    mul!(buffer, ops.fft_plan, u)  # in-place transform
    return buffer
end

"""
    to_physical(u, u_hat, ops)

In-place inverse transform: spectral field `u_hat` ((Nx÷2+1)×Ny)
→ physical field `u` (Nx×Ny, real).
Writes result into pre-allocated `u`.
"""
@inline function to_physical!(buffer::AbstractArray{Float64}, u_hat::AbstractArray{ComplexF64}, ops::Operators)
    copyto!(ops.temp_complex_irfft, u_hat) # 单独的 buffer 用于 irfft，避免覆盖输入 u_hat
    mul!(buffer, ops.ifft_plan, ops.temp_complex_irfft)  # in-place transform
    return buffer
end


# ============================================================
#  MULTIPLICATION
# ============================================================

"""
谱系数点乘函数，不用去混叠
"""
function mult!(buffer::AbstractArray{ComplexF64}, u_hat::AbstractArray{ComplexF64}, v_hat::AbstractArray{ComplexF64}, ops::Operators)

    u = to_physical!(ops.temp_real1, u_hat, ops)
    v = to_physical!(ops.temp_real2, v_hat, ops)

    w = u .* v

    # 变回频域
    to_spectral!(buffer, w, ops)
    return buffer
end