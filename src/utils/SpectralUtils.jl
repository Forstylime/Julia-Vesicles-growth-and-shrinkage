# File: src/SpectralUtils.jl

using FFTW

"""
生成二维傅里叶谱方法微分算子
"""
function generate_operators(conf::Config)
    N   = (conf.Nx, conf.Ny)
    L   = (conf.Lx, conf.Ly)
    dim = length(N)

    # 1. 各维度波数向量
    # fftfreq(n, 1/dx) = fftfreq(n, N/L) 给出以 rad/unit 为单位的波数
    k_vecs = ntuple(d -> 2π * FFTW.fftfreq(N[d], N[d] / L[d]), dim)

    # 2. 广播到完整 Nx×Ny 网格（避免后续每次运算都触发广播）
    K = ntuple(dim) do d
        sz    = ntuple(i -> i == d ? N[d] : 1, dim)
        k_col = reshape(k_vecs[d], sz)
        return k_col .* ones(Float64, N...)   # 展开为完整矩阵
    end

    # 3. 一阶微分算子 D1 = (i·kx, i·ky)
    D1 = ntuple(d -> complex.(zeros(Float64, N...), K[d]), dim)

    # 4. 拉普拉斯与双谐波算子
    K2_sum    = zeros(Float64, N...)           # 修正：N... 展开元组
    for d in 1:dim
        @. K2_sum += K[d]^2
    end
    Laplacian  = .-K2_sum
    Biharmonic = K2_sum .^ 2

    # 5. 预规划 FFT（MEASURE 模式：启动慢，运行快）
    tmp    = zeros(ComplexF64, N...)
    p_fft  = plan_fft(tmp;  flags=FFTW.MEASURE)
    p_ifft = plan_ifft(tmp; flags=FFTW.MEASURE)

    # 类型参数由编译器自动推断，不手动指定 {dim}
    return Operators(K, D1, Laplacian, Biharmonic, p_fft, p_ifft)
end