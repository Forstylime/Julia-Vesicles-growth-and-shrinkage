# File: src/Types.jl

using FFTW

"""
仿真全局参数配置
"""
struct Config
    # 物理参数
    epsilon::Float64
    M_phi::Float64
    M0_psi::Float64
    eta::Float64
    gamma_surf::Float64
    gamma_area::Float64
    gamma_bend::Float64
    gamma_in::Float64
    beta_in::Float64
    psi_in::Float64
    gamma_out::Float64
    beta_out::Float64
    psi_out::Float64
    lamda::Float64

    # 稳定化参数与SAV参数
    S1::Float64; S2::Float64; S3::Float64; S4::Float64
    C1::Float64; C2::Float64; C3::Float64

    # 数值参数
    dt::Float64
    T::Float64
    Nt::Int
    Nx::Int
    Ny::Int
    Lx::Float64
    Ly::Float64
    dx::Float64
    dy::Float64

    # 控制参数
    tol::Float64
    goal::Symbol

    # 物理约束辅助量
    A::Float64
end

"""自动计算派生数值参数的外部构造函数"""
function Config(; epsilon, M_phi, M0_psi, eta,
                  gamma_surf, gamma_area, gamma_bend,
                  gamma_in, beta_in, psi_in,
                  gamma_out, beta_out, psi_out,
                  lamda,
                  S1, S2, S3, S4, C1, C2, C3,
                  dt, T, Nx, Ny, Lx, Ly,
                  tol, goal, A)
    Nt = round(Int, T / dt)
    dx = Lx / Nx
    dy = Ly / Ny
    return Config(
        epsilon, M_phi, M0_psi, eta,
        gamma_surf, gamma_area, gamma_bend,
        gamma_in, beta_in, psi_in,
        gamma_out, beta_out, psi_out,
        lamda,
        S1, S2, S3, S4, C1, C2, C3,
        dt, T, Nt, Nx, Ny, Lx, Ly, dx, dy,
        tol, goal, A
    )
end
Base.broadcastable(c::Config) = Ref(c)
# ----------------------------------------------------------------

"""
存储物理场（实空间与谱空间）
"""
mutable struct FieldState
    # 主物理场
    phi::Matrix{Float64}
    phi_hat::Matrix{ComplexF64}

    psi::Matrix{Float64}
    psi_hat::Matrix{ComplexF64}

    u::Array{Float64, 3}        # Nx × Ny × 2
    u_hat::Array{ComplexF64, 3}

    p::Matrix{Float64}
    p_hat::Matrix{ComplexF64}

    # 化学势
    mu::Matrix{Float64}
    mu_hat::Matrix{ComplexF64}

    nu::Matrix{Float64}
    nu_hat::Matrix{ComplexF64}

    # SAV 标量变量
    Q::Float64
    R1::Float64
    R2::Float64
    R3::Float64

    # 物理约束辅助量
    A::Float64
end

# ----------------------------------------------------------------

"""
谱空间算子与 FFT 计划
类型参数 P, IP 由构造时自动推断，保留 FFTW plan 的具体类型以避免动态分发
"""
struct Operators{P, IP}
    K::NTuple{2, Matrix{Float64}}
    D1::NTuple{2, Matrix{ComplexF64}}
    Laplacian::Matrix{Float64}
    Biharmonic::Matrix{Float64}
    fft_plan::P
    ifft_plan::IP
end

"""
BDF 时间离散系数
n=1 时用 BDF1：a=1, b=-1, c=0
n>1 时用 BDF2：a=1.5, b=-2, c=0.5
"""
struct BDFCoeff
    a::Float64
    b::Float64
    c::Float64
end