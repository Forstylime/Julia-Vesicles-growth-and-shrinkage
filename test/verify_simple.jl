include("scripts/Main.jl")

# Dummy data for ndim=3 (state.phi is 3D, phi_sum is 2D)
struct DummyState3D
    phi::Array{Float64, 3}
end
state3d = DummyState3D(rand(64, 64, 1))
energy_hist = [1.0]
area_hist = [1.0]
Dt = [0.0, 1.0]
T = 1.0

println("Testing ndim=3...")
_visualize(state3d, energy_hist, area_hist, Dt, T, "results")

# Dummy data for ndim=4 (state.phi is 4D, phi_sum is 3D)
struct DummyState4D
    phi::Array{Float64, 4}
end
state4d = DummyState4D(rand(64, 64, 64, 1))

println("Testing ndim=4...")
_visualize(state4d, energy_hist, area_hist, Dt, T, "results")

println("Verification script finished.")
