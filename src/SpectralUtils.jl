# File: src/SpectralUtils.jl
#
# Builds spectral derivative operators and FFTW plans.
#   - rfft/irfft instead of fft/ifft: exploits real-valuedness of all
#     physical fields, giving (Nx÷2+1)×Ny spectral arrays (~half storage)
#   - Wavenumber arrays kept as broadcast-compatible vectors, not full
#     matrices: kx is (Nx÷2+1)×1, ky is 1×Ny. Operations like kx.^2 .+ ky.^2
#     produce (N

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

    # ── 原有代码不变（wavenumbers, D1, Laplacian 等）──
    kx  = convert(Matrix{Float64}, reshape(2π .* rfftfreq(Nx, Nx / Lx), :, 1))
    kx0 = convert(Matrix{Float64}, reshape(2π .* fftfreq(Nx, Nx / Lx), Nx, 1))
    ky  = convert(Matrix{Float64}, reshape(2π .* fftfreq(Ny, Ny / Ly), 1, :))
    K   = (kx, ky)::NTuple{2, Matrix{Float64}}
    D1  = (complex.(0.0, kx),  complex.(0.0, ky))
    D1_full = (complex.(0.0, kx0), complex.(0.0, ky))
    K2         = @. kx^2 + ky^2
    Laplacian       = @. -K2
    Laplacian_full  = @. -(kx0^2 + ky^2)
    Biharmonic      = @. K2^2

    # ── 原有 FFTW plans（rfft 族）──
    tmp_real     = zeros(Float64,    Nx, Ny, N)
    tmp_complex  = zeros(ComplexF64, Nx÷2+1, Ny, N)
    tmp_real_1   = zeros(Float64,    Nx, Ny)
    tmp_complex_1= zeros(ComplexF64, Nx÷2+1, Ny)
    tmp_real_2   = zeros(Float64,    Nx, Ny, 2)
    tmp_complex_2= zeros(ComplexF64, Nx÷2+1, Ny, 2)

    fft_plan   = plan_rfft(tmp_real,          (1,2); flags=FFTW.MEASURE)
    ifft_plan  = plan_irfft(tmp_complex,  Nx, (1,2); flags=FFTW.MEASURE)
    fft_plan_1 = plan_rfft(tmp_real_1,        (1,2); flags=FFTW.MEASURE)
    ifft_plan_1= plan_irfft(tmp_complex_1,Nx, (1,2); flags=FFTW.MEASURE)
    fft_plan_2 = plan_rfft(tmp_real_2,        (1,2); flags=FFTW.MEASURE)
    ifft_plan_2= plan_irfft(tmp_complex_2,Nx, (1,2); flags=FFTW.MEASURE)

    # ── 新增：全尺寸 ComplexF64 plan（给 BiCGSTAB 内核用）──
    tmp_full = zeros(ComplexF64, Nx, Ny, N)
    fft_plan_full  = plan_fft(tmp_full,  (1,2); flags=FFTW.MEASURE)
    ifft_plan_full = plan_ifft(tmp_full, (1,2); flags=FFTW.MEASURE)

    inv_s = 1.0 / (Nx * Ny)

    # ── 原有临时缓冲区（类型改为具体的 Array）──
    temp_real1 = zeros(Float64,    Nx, Ny, N)
    temp_real2 = zeros(Float64,    Nx, Ny, N)
    temp_real3 = zeros(Float64,    Nx, Ny, N)
    temp_comp1 = zeros(ComplexF64, Nx÷2+1, Ny, N)
    temp_comp2 = zeros(ComplexF64, Nx÷2+1, Ny, N)
    temp_comp3 = zeros(ComplexF64, Nx÷2+1, Ny, N)
    temp_complex_irfft = zeros(ComplexF64, Nx÷2+1, Ny, N)

    # ── 新增：BiCGSTAB matvec 5 个工作缓冲区（全尺寸）──
    buf_mv1 = zeros(ComplexF64, Nx, Ny, N)
    buf_mv2 = zeros(ComplexF64, Nx, Ny, N)
    buf_mv3 = zeros(ComplexF64, Nx, Ny, N)
    buf_mv4 = zeros(ComplexF64, Nx, Ny, N)
    buf_mv5 = zeros(ComplexF64, Nx, Ny, N)

    Nk = Nx ÷ 2 + 1
    buf_rhat1  = zeros(ComplexF64, Nk, Ny, N)
    buf_rhat2  = zeros(ComplexF64, Nk, Ny, N)
    buf_uhat1  = zeros(ComplexF64, Nk, Ny, 2)
    buf_uhat2  = zeros(ComplexF64, Nk, Ny, 2)
    buf_uphys1 = zeros(Float64, Nx, Ny, 2)
    buf_uphys2 = zeros(Float64, Nx, Ny, 2)
    buf_phys2d = zeros(Float64, Nx, Ny)
    buf_hat2d  = zeros(ComplexF64, Nk, Ny)

    return Operators(
        K, D1, D1_full,
        Laplacian, Laplacian_full, Biharmonic,
        fft_plan, ifft_plan,
        fft_plan_full, ifft_plan_full,
        fft_plan_1, ifft_plan_1,
        fft_plan_2, ifft_plan_2,
        Nx, Ny, inv_s,
        temp_real1, temp_real2, temp_real3,
        temp_comp1, temp_comp2, temp_comp3,
        temp_complex_irfft,
        buf_mv1, buf_mv2, buf_mv3, buf_mv4, buf_mv5,
        buf_rhat1, buf_rhat2,
        buf_uhat1, buf_uhat2,
        buf_uphys1, buf_uphys2,
        buf_phys2d, buf_hat2d
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
@inline function to_spectral!(buffer::Array{ComplexF64, 3}, u::Array{Float64, 3}, ops::Operators)
    mul!(buffer, ops.fft_plan, u)  # in-place transform
    return buffer
end

"""
    to_physical(u, u_hat, ops)

In-place inverse transform: spectral field `u_hat` ((Nx÷2+1)×Ny)
→ physical field `u` (Nx×Ny, real).
Writes result into pre-allocated `u`.
"""
@inline function to_physical!(buffer::Array{Float64, 3}, u_hat::Array{ComplexF64, 3}, ops::Operators)
    copyto!(ops.temp_complex_irfft, u_hat) # 单独的 buffer 用于 irfft，避免覆盖输入 u_hat
    mul!(buffer, ops.ifft_plan, ops.temp_complex_irfft)  # in-place transform
    return buffer
end


# ============================================================
#  MULTIPLICATION
# ============================================================

"""
谱系数点乘函数
"""
function mult!(buffer::AbstractArray{ComplexF64}, u_hat::AbstractArray{ComplexF64}, v_hat::AbstractArray{ComplexF64}, ops::Operators)

    u = to_physical!(ops.temp_real1, u_hat, ops)
    v = to_physical!(ops.temp_real2, v_hat, ops)

    w = u .* v

    # 变回频域
    to_spectral!(buffer, w, ops)
    return buffer
end