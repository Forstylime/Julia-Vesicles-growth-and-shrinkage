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
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    N = cfg.N

    # 提前计算好频率向量（广播运算自动生成 Vector{Float64}）
    vec_kx0 = 2π .* fftfreq(Nx, Nx/Lx)
    vec_kx  = 2π .* rfftfreq(Nx, Nx/Lx)
    vec_ky  = 2π .* fftfreq(Ny, Ny/Ly)
    vec_kz  = 2π .* fftfreq(Nz, Nz/Lz)

    # 使用 reshape 返回原生的 Array{Float64, 3}，零内存拷贝（与原 Vector 共享内存）
    kx0 = reshape(vec_kx0, :, 1, 1)
    kx  = reshape(vec_kx,  :, 1, 1)
    ky  = reshape(vec_ky,  1, :, 1)
    kz  = reshape(vec_kz,  1, 1, :)

    # 利用 im 乘法生成 Array{ComplexF64, 3}，代码更紧凑
    D1      = (im .* kx,  im .* ky, im .* kz)
    D1_full = (im .* kx0, im .* ky, im .* kz)

    # 利用 @. (Loop Fusion) 避免生成中间变量 K2，节省了 134MB 的内存 (假设 Nx=Ny=Nz=256)
    Laplacian      = @. -(kx^2 + ky^2 + kz^2)
    Laplacian_full = @. -(kx0^2 + ky^2 + kz^2)

    # Biharmonic 就是拉普拉斯算子的平方，继续复用结果
    Biharmonic     = @. Laplacian^2

    # ── 原有 FFTW plans（rfft 族）──
    tmp_real     = zeros(Float64,    Nx, Ny, Nz, N)
    tmp_complex  = zeros(ComplexF64, Nx÷2+1, Ny, Nz, N)
    tmp_real_1   = zeros(Float64,    Nx, Ny, Nz)
    tmp_complex_1= zeros(ComplexF64, Nx÷2+1, Ny, Nz)
    tmp_real_2   = zeros(Float64,    Nx, Ny, Nz, 3)
    tmp_complex_2= zeros(ComplexF64, Nx÷2+1, Ny, Nz, 3)

    fft_plan   = plan_rfft(tmp_real,          (1,2,3); flags=FFTW.MEASURE)
    ifft_plan  = plan_irfft(tmp_complex,  Nx, (1,2,3); flags=FFTW.MEASURE)
    fft_plan_1 = plan_rfft(tmp_real_1,        (1,2,3); flags=FFTW.MEASURE)
    ifft_plan_1= plan_irfft(tmp_complex_1,Nx, (1,2,3); flags=FFTW.MEASURE)
    fft_plan_2 = plan_rfft(tmp_real_2,        (1,2,3); flags=FFTW.MEASURE)
    ifft_plan_2= plan_irfft(tmp_complex_2,Nx, (1,2,3); flags=FFTW.MEASURE)

    # ── 新增：全尺寸 ComplexF64 plan（给 BiCGSTAB 内核用）──
    tmp_full = zeros(ComplexF64, Nx, Ny, Nz, N)
    fft_plan_full  = plan_fft(tmp_full,  (1,2,3); flags=FFTW.MEASURE)
    ifft_plan_full = plan_ifft(tmp_full, (1,2,3); flags=FFTW.MEASURE)

    inv_s = 1.0 / (Nx * Ny * Nz)

    # ── 原有临时缓冲区（类型改为具体的 Array）──
    temp_real1 = zeros(Float64,    Nx, Ny, Nz, N)
    temp_real2 = zeros(Float64,    Nx, Ny, Nz, N)
    temp_real3 = zeros(Float64,    Nx, Ny, Nz, N)
    temp_real4 = zeros(Float64,    Nx, Ny, Nz, N)
    temp_comp1 = zeros(ComplexF64, Nx÷2+1, Ny, Nz, N)
    temp_comp2 = zeros(ComplexF64, Nx÷2+1, Ny, Nz, N)
    temp_comp3 = zeros(ComplexF64, Nx÷2+1, Ny, Nz, N)
    temp_comp4 = zeros(ComplexF64, Nx÷2+1, Ny, Nz, N)
    temp_complex_irfft = zeros(ComplexF64, Nx÷2+1, Ny, Nz, N)

    Nk = Nx ÷ 2 + 1
    buf_rhat1  = zeros(ComplexF64, Nk, Ny, Nz, N)
    buf_rhat2  = zeros(ComplexF64, Nk, Ny, Nz, N)
    buf_uhat1  = zeros(ComplexF64, Nk, Ny, Nz, 3)
    buf_uhat2  = zeros(ComplexF64, Nk, Ny, Nz, 3)
    buf_uphys1 = zeros(Float64, Nx, Ny, Nz, 3)
    buf_uphys2 = zeros(Float64, Nx, Ny, Nz, 3)
    buf_deriv  = zeros(Float64, Nx, Ny, Nz)
    buf_phys2d = zeros(Float64, Nx, Ny, Nz)
    buf_hat2d  = zeros(ComplexF64, Nk, Ny, Nz)

    return Operators(
        D1, D1_full,
        Laplacian, Laplacian_full, Biharmonic,
        fft_plan, ifft_plan,
        fft_plan_full, ifft_plan_full,
        fft_plan_1, ifft_plan_1,
        fft_plan_2, ifft_plan_2,
        Nx, Ny, Nz, inv_s,
        temp_real1, temp_real2, temp_real3, temp_real4,
        temp_comp1, temp_comp2, temp_comp3, temp_comp4,
        temp_complex_irfft,
        buf_rhat1, buf_rhat2,
        buf_uhat1, buf_uhat2,
        buf_uphys1, buf_uphys2, buf_deriv,
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
@inline function to_spectral!(buffer::Array{ComplexF64, 4}, u::Array{Float64, 4}, ops::Operators)
    mul!(buffer, ops.fft_plan, u)  # in-place transform
    return buffer
end

"""
    to_physical(u, u_hat, ops)

In-place inverse transform: spectral field `u_hat` ((Nx÷2+1)×Ny)
→ physical field `u` (Nx×Ny, real).
Writes result into pre-allocated `u`.
"""
@inline function to_physical!(buffer::Array{Float64, 4}, u_hat::Array{ComplexF64, 4}, ops::Operators)
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