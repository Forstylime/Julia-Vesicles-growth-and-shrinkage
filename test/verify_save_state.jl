using MAT
using Test
using FFTW
using LinearAlgebra

# Mock the FieldState and Config if they are not easily includable or to keep it fast
# But actually we can include them from the project
include(joinpath(@__DIR__, "..", "src", "Types.jl"))
include(joinpath(@__DIR__, "..", "scripts", "Main.jl"))

@testset "Verify MAT Saving" begin
    # Create a dummy state
    Nx, Ny, N = 16, 16, 1
    state = FieldState(Nx, Ny, N)
    state.phi .= rand(Nx, Ny, N)
    state.psi .= rand(Nx, Ny, N)
    state.mu .= rand(Nx, Ny, N)
    state.nu .= rand(Nx, Ny, N)
    state.p .= rand(Nx, Ny)
    state.u .= rand(Nx, Ny, 2)
    state.R1, state.R2, state.R3, state.Q = 1.1, 2.2, 3.3, 4.4

    save_path = mktempdir()
    mat_folder = joinpath(@__DIR__, "..", "MAT/case3")
    mkpath(mat_folder)
    
    # Define dummy Dt and T
    Dt = [0.0, 1e-6]
    T = 1e-6
    
    # Call _visualize
    # Note: _visualize calls set_para_base, which is in src/Init.jl (likely)
    # Since we included Main.jl, which includes everything, it should be fine.
    
    _visualize(state, [0.0], [0.0], Dt, T, save_path)
    
    mat_file = joinpath(mat_folder, "state_2d_final.mat")
    @test isfile(mat_file)
    
    data = matread(mat_file)
    @test haskey(data, "phi")
    @test haskey(data, "psi")
    @test haskey(data, "mu")
    @test haskey(data, "nu")
    @test haskey(data, "p")
    @test haskey(data, "u")
    @test haskey(data, "R1")
    @test haskey(data, "R2")
    @test haskey(data, "R3")
    @test haskey(data, "Q")
    
    @test data["R1"] == 1.1
    @test data["Q"] == 4.4
    @test size(data["phi"]) == (Nx, Ny, N)
    
    println("Verification successful!")
end
