include("scripts/Main.jl")
println("MAT loaded: ", isdefined(Main, :MAT))
# Run a very short main to check _visualize
main(T=2e-7, dt=1e-7)
println("Simulation finished and _visualize called.")
