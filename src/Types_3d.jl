# File: src/Types.jl
#
# Defines all core data structures for the simulation.
# Design principles:
#   - Config is fully immutable (constructed once, never mutated)
#   - FieldState is mutable only for scalar SAV variables; array fields
#     are updated in-place via copyto!() to avoid allocations
#   - Operators stores vary plans (real-to-complex)
#   - All structs are parametric where the type may vary (e.g., GPU path)


using FFTW

# ============================================================
#  1. SIMULATION CONFIGURATION
# ============================================================

"""
    Config

Immutable global parameter configuration. Constructed once before the
simulation loop and never modified. All derived quantities (Nt, dx, dy)
are computed automatically by the keyword constructor.

Physical parameters use their standard symbol names from the paper.
"""
struct Config
    # --- Physical parameters ---
    N             :: Int          # number of phase fields (N=1 for single SAV)
    epsilon       :: Float64
    M_phi         :: Float64
    M0_psi        :: Float64
    eta           :: Float64
    gamma_surf    :: Float64
    gamma_area    :: Int
    gamma_bend    :: Float64
    gamma_in      :: Int
    beta_in       :: Int
    psi_in_v      :: Vector{Float64}
    gamma_out     :: Int
    beta_out      :: Int
    psi_out_v     :: Vector{Float64}
    lamda         :: Int

    # --- Stabilization and SAV parameters ---
    S1 :: Float64
    S2 :: Float64
    S3 :: Float64
    S4 :: Float64
    C1 :: Float64
    C2 :: Float64
    C3 :: Float64

    # --- Time discretization ---
    dt :: Float64
    T  :: Float64
    Nt :: Int          # derived: round(T / dt)

    # --- Spatial grid (primary) ---
    Nx :: Int
    Ny :: Int
    Nz :: Int
    Lx :: Float64
    Ly :: Float64
    Lz :: Float64

    # --- Grid spacing (derived) ---
    dx :: Float64      # derived: Lx / Nx
    dy :: Float64      # derived: Ly / Ny
    dz :: Float64      # derived: Ly / Ny

    # --- Solver control ---
    tol  :: Float64
    goal :: Symbol

    spe_len :: Int
end

"""
    Config_3d(; epsilon, M_phi, ...) -> Config_3d

Keyword constructor that automatically computes derived quantities:
- `Nt  = round(Int, T / dt)`
- `dx  = Lx / Nx`
- `dy  = Ly / Ny`
- `dz  = Lz / Nz`

# Example
```julia
cfg = Config(
    N = 1,
    epsilon=0.05, M_phi=1.0, M0_psi=1.0, eta=1.0,
    gamma_surf=1.0, gamma_area=1.0, gamma_bend=1.0,
    gamma_in=1.0, beta_in=0.0, psi_in_v=[1.0],
    gamma_out=1.0, beta_out=0.0, psi_out_v=[1.0],
    lamda=1,
    S1=1.0, S2=1.0, S3=1.0, S4=1.0,
    C1=1.0, C2=1.0, C3=1.0,
    dt=1e-4, T=1.0, Nx=128, Ny=128, Lx=2π, Ly=2π,
    tol=1e-10, goal=:s
)
```
"""

function Config(;
        N,
        epsilon, M_phi, M0_psi, eta,
        gamma_surf, gamma_area, gamma_bend,
        gamma_in, beta_in, psi_in_v::Vector{Float64},
        gamma_out, beta_out, psi_out_v::Vector{Float64},
        lamda,
        S1, S2, S3, S4,
        C1, C2, C3,
        dt, T, Nx, Ny, Nz, Lx, Ly, Lz,
        tol, goal)

    Nt = round(Int, T / dt)
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    spe_len = (Nx ÷ 2 + 1) * Ny * Nz * N  # 频谱空间的总长度

    return Config(
        N,
        epsilon, M_phi, M0_psi, eta,
        gamma_surf, gamma_area, gamma_bend,
        gamma_in, beta_in, psi_in_v,
        gamma_out, beta_out, psi_out_v,
        lamda,
        S1, S2, S3, S4,
        C1, C2, C3,
        dt, T, Nt,
        Nx, Ny, Nz, Lx, Ly, Lz,
        dx, dy, dz, 
        tol, goal,
        spe_len
    )
end

# Allows broadcasting over a Config without expanding it:
# e.g., f.(field, cfg) works correctly
Base.broadcastable(c::Config) = Ref(c)


# ============================================================
#  2. FIELD STATE
# ============================================================

"""
    FieldState

Stores all physical fields at a single time level (either t=n or t=n-1).

# Array layout conventions
- `phi`:   Nx × Ny × Nφ  (third dim = number of phase fields; Nφ=1 default)
- `u`:     Nx × Ny × 2   (last dim = spatial components, column-major friendly)
- `p`:     Nx × Ny (pressure)
- `*_hat`: spectral counterparts from rfft; shape is (Nx÷2+1) × Ny × [Nφ or 2 or 1]
           rfft exploits real-valuedness: only stores non-redundant frequencies

# Mutability
Arrays are NOT individually reassigned — update them in-place with copyto!().
Only the SAV scalar fields (Q, R1, R2, R3) are plain mutable
fields that get scalar assignment each timestep. 
while A0 should be inmutable due to no changes during all process

# Updating state between timesteps
Do NOT do: `state_prev = state_curr`  (this is a pointer copy, not a deep copy)
DO use the provided: `update_state!(state_prev, state_curr)`
"""
mutable struct FieldState
    # --- Phase field and its spectrum ---
    phi     :: Array{Float64, 4}     # Nx × Ny × Nz × N
    phi_hat :: Array{ComplexF64, 4}  # (Nx÷2+1) × Ny × Nz × N

    # --- Membrane field and its spectrum ---
    psi     :: Array{Float64, 4}       # Nx × Ny × Nz × N
    psi_hat :: Array{ComplexF64, 4}    # (Nx÷2+1) × Ny × Nz × N

    # --- Velocity field and its spectrum ---
    u       :: Array{Float64, 4}     # Nx × Ny × Nz × 2
    u_hat   :: Array{ComplexF64, 4}  # (Nx÷2+1) × Ny × Nz × 2

    # --- Pressure field and its spectrum ---
    p       :: Array{Float64, 3}       # Nx × Ny × Nz
    p_hat   :: Array{ComplexF64, 3}    # (Nx÷2+1) × Ny × Nz

    # --- Chemical potential φ and its spectrum ---
    mu      :: Array{Float64, 4}       # Nx × Ny × Nz × N
    mu_hat  :: Array{ComplexF64, 4}    # (Nx÷2+1) × Ny × Nz × N

    # --- Chemical potential ψ and its spectrum ---
    nu      :: Array{Float64, 4}       # Nx × Ny × Nz × N
    nu_hat  :: Array{ComplexF64, 4}    # (Nx÷2+1) × Ny × Nz × N

    # --- SAV scalar variables (mutated every timestep) ---
    Q  :: Float64
    R1 :: Float64
    R2 :: Float64
    R3 :: Float64

    # --- Initial area constraint target (derived from A0) ---
    A0 :: Vector{Float64}
    dt :: Float64
end

# 定义构造函数进行预分配
function FieldState(Nx::Int, Ny::Int, Nz::Int, N::Int)
    # 谱空间维度 (假设使用了 RFFT，通常是 Nx/2 + 1)
    Nx_hat = Nx ÷ 2 + 1
    
    # 预分配内存
    phi     = zeros(Float64, Nx, Ny, Nz, N)
    phi_hat = zeros(ComplexF64, Nx_hat, Ny, Nz, N)
    
    psi     = zeros(Float64, Nx, Ny, Nz, N)
    psi_hat = zeros(ComplexF64, Nx_hat, Ny, Nz, N)
    
    u       = zeros(Float64, Nx, Ny, Nz, 2)
    u_hat   = zeros(ComplexF64, Nx_hat, Ny, Nz, 2)
    
    p       = zeros(Float64, Nx, Ny, Nz)
    p_hat   = zeros(ComplexF64, Nx_hat, Ny, Nz)
    
    mu      = zeros(Float64, Nx, Ny, Nz, N)
    mu_hat  = zeros(ComplexF64, Nx_hat, Ny, Nz, N)
    
    nu      = zeros(Float64, Nx, Ny, Nz, N)
    nu_hat  = zeros(ComplexF64, Nx_hat, Ny, Nz, N)
    
    # 初始化标量变量
    Q, R1, R2, R3, A0, dt = 0.0, 0.0, 0.0, 0.0, [0.0], 0.0
    
    # 返回构造好的实例
    return FieldState(phi, phi_hat, psi, psi_hat, u, u_hat, p, p_hat, mu, mu_hat, nu, nu_hat, Q, R1, R2, R3, A0, dt)
end

"""
    update_state!(dest::FieldState, src::FieldState)

In-place copy of all fields from `src` into `dest`.
Use this to perform `u_prev ← u_curr` at the end of each timestep.

Avoids allocation by writing into pre-existing arrays rather than
creating new ones. This is the ONLY correct way to advance the time level.
"""
function update_state!(dest::FieldState, src::FieldState)
    copyto!(dest.phi,     src.phi)
    copyto!(dest.phi_hat, src.phi_hat)
    copyto!(dest.psi,     src.psi)
    copyto!(dest.psi_hat, src.psi_hat)
    copyto!(dest.u,       src.u)
    copyto!(dest.u_hat,   src.u_hat)
    copyto!(dest.p,       src.p)
    copyto!(dest.p_hat,   src.p_hat)
    copyto!(dest.mu,      src.mu)
    copyto!(dest.mu_hat,  src.mu_hat)
    copyto!(dest.nu,      src.nu)
    copyto!(dest.nu_hat,  src.nu_hat)

    dest.Q           = src.Q
    dest.R1          = src.R1
    dest.R2          = src.R2
    dest.R3          = src.R3

    # 注意：A0 是初始值，不随时间更新，因此不复制

    return dest
end


# ============================================================
#  3. SPECTRAL OPERATORS
# ============================================================

"""
    Operators{P, IP, PP, IPP}

Spectral operators and FFTW plans

# Using rfft2 and irfft2 gobal, and some else for specific situation

# Sign convention
All derivative operators follow the standard spectral convention:
  D1[α]    =  i * k[α]              (first derivative in direction α)
  Laplacian = -(kx² + ky²)          (negative semi-definite)
  Biharmonic = (kx² + ky²)²         (positive semi-definite)

K is the raw wavenumber arrays (real-valued, before multiplication by i).
But will not be stored.

# rfft layout
For an Nx×Ny real field, rfft output has shape (Nx÷2+1) × Ny.
This is exactly half (+1) the storage of a full complex fft, and is
the correct shape for all `*_hat` fields in FieldState.

Type parameters are inferred by build_operators(); never set manually.
  P, IP:    rfft/irfft plan types for the Nx×Ny physical grid
"""
# src/Types.jl
# 修改 Operators struct，将 AbstractArray 改为 Array，并添加全尺寸 FFT plan 和 matvec 缓冲区

struct Operators{P, IP, PF, IPF}   # 新增 PF, IPF 两个类型参数
    D1             :: NTuple{3, Array{ComplexF64, 3}}
    D1_full        :: NTuple{3, Array{ComplexF64, 3}}
    Laplacian      :: Array{Float64, 3}
    Laplacian_full :: Array{Float64, 3}
    Biharmonic     :: Array{Float64, 3}
    fft_plan       :: P
    ifft_plan      :: IP
    fft_plan_full  :: PF           # 全尺寸 fft plan (ComplexF64 输入)
    ifft_plan_full :: IPF          # 全尺寸 ifft plan (ComplexF64 输入)
    fft_plan_1     :: Any          # 保持原来的 Any（标量场专用）
    ifft_plan_1    :: Any
    fft_plan_2     :: P
    ifft_plan_2    :: IP
    Nx             :: Int
    Ny             :: Int
    Nz             :: Int
    inv_s          :: Float64
    # ── 改为具体类型（消除 AbstractArray 传染）──
    temp_real1          :: Array{Float64, 4}
    temp_real2          :: Array{Float64, 4}
    temp_real3          :: Array{Float64, 4}
    temp_real4          :: Array{Float64, 4}
    temp_comp1          :: Array{ComplexF64, 4}
    temp_comp2          :: Array{ComplexF64, 4}
    temp_comp3          :: Array{ComplexF64, 4}
    temp_comp4          :: Array{ComplexF64, 4}
    temp_complex_irfft  :: Array{ComplexF64, 4}
    # ── Step 2/3/4/5 共用的谱空间工作缓冲区（半谱尺寸）──
    buf_rhat1 :: Array{ComplexF64, 4}   # (Nk × Ny × Nz × N)  通用谱缓冲
    buf_rhat2 :: Array{ComplexF64, 4}
    # ── Step 4/5 速度场工作缓冲区（半谱，2分量）──
    buf_uhat1 :: Array{ComplexF64, 4}   # (Nk × Ny × Nz × 2)
    buf_uhat2 :: Array{ComplexF64, 4}
    # ── Step 4/5 物理空间速度工作缓冲区 ──
    buf_uphys1 :: Array{Float64, 4}     # (Nx × Ny × Nz × 2)
    buf_uphys2 :: Array{Float64, 4}
    buf_deriv :: Array{Float64, 3}
    # ── 通用实空间 3D 工作缓冲区（标量，不带 N 维）──
    buf_phys2d :: Array{Float64, 3}     # (Nx × Ny × Nz)
    buf_hat2d  :: Array{ComplexF64, 3}  # (Nk × Ny × Nz)
end


# src/Types.jl
# 在 FieldState 和 Operators 定义之后，添加以下内容

# ============================================================
#  Step1Cache：替代 Dict{Symbol,Any}，完全预分配
# ============================================================
"""
    Step1Cache

预分配的 Step 1 结果容器，替代 Dict{Symbol,Any}。
所有字段类型确定，零装箱，支持原地写入 (.=)。
在仿真初始化时创建一次，整个时间循环复用。
"""
struct Step1Cache
    phi_11 :: Array{ComplexF64, 4}
    mu_11  :: Array{ComplexF64, 4}
    psi_11 :: Array{ComplexF64, 4}
    nu_11  :: Array{ComplexF64, 4}

    phi_12 :: Array{ComplexF64, 4}
    mu_12  :: Array{ComplexF64, 4}
    psi_12 :: Array{ComplexF64, 4}
    nu_12  :: Array{ComplexF64, 4}

    phi_13 :: Array{ComplexF64, 4}
    mu_13  :: Array{ComplexF64, 4}
    phi_14 :: Array{ComplexF64, 4}
    mu_14  :: Array{ComplexF64, 4}

    phi_21 :: Array{ComplexF64, 4}
    mu_21  :: Array{ComplexF64, 4}
    psi_21 :: Array{ComplexF64, 4}
    nu_21  :: Array{ComplexF64, 4}
end

function Step1Cache(Nx::Int, Ny::Int, Nz::Int, N::Int)
    Nk = Nx ÷ 2 + 1
    f() = zeros(ComplexF64, Nk, Ny, Nz, N)
    return Step1Cache(f(),f(),f(),f(), f(),f(),f(),f(),
                      f(),f(),f(),f(), f(),f(),f(),f())
end


# ============================================================
#  4. BDF TIME-STEPPING COEFFICIENTS
# ============================================================

"""
    BDFCoeff

Coefficients for BDF1 (first timestep) and BDF2 (all subsequent timesteps).

The discretization of the time derivative is:
  (a * u^n + b * u^(n-1) + c * u^(n-2)) / dt  ≈  du/dt

| Scheme | a    | b    | c   |
|--------|------|------|-----|
| BDF1   | 1.0  | -1.0 | 0.0 |
| BDF2   | 1.5  | -2.0 | 0.5 |

Use the provided constructors rather than setting fields manually.
"""
struct BDFCoeff
    a :: Float64
    b :: Float64
    c :: Float64
end

"""Return BDF1 coefficients (for the first timestep, n=1)."""
BDFCoeff(::Val{1}) = BDFCoeff(1.0, -1.0,  0.0)

"""Return BDF2 coefficients (for all timesteps n ≥ 2)."""
BDFCoeff(::Val{2}) = BDFCoeff(1.5, -2.0,  0.5)

"""
    bdf_coeff(n::Int) -> BDFCoeff

Convenience selector: returns BDF1 for n=1, BDF2 for n≥2.
Type-stable: the `Val` dispatch resolves at compile time when `n` is
a compile-time constant, and at runtime otherwise (still correct).
"""
bdf_coeff(n::Int) = n == 1 ? BDFCoeff(Val(1)) : BDFCoeff(Val(2))
