
# Mock verification of the saving logic
save_interval = 0.001
Dt = [0.0, 1e-7]
# Simulate a sequence of time points that should hit multiples of 0.001
# even with floating point errors
dt = 1e-5
t = 1e-5
while t <= 0.015 + 1e-7
    push!(Dt, t)
    t += dt
end

# Introduce some noise to simulate floating point issues
Dt_noisy = Dt .+ (rand(length(Dt)) .- 0.5) .* 1e-18

function simulate_saving(Dt, save_interval)
    last_save_idx = -1
    saved_times = Float64[]
    for n in 1:length(Dt)
        curr_save_idx = floor(Int, (Dt[n] + 1e-12) / save_interval)
        if curr_save_idx > last_save_idx
            push!(saved_times, Dt[n])
            last_save_idx = curr_save_idx
        end
    end
    return saved_times
end

saved_times = simulate_saving(Dt_noisy, save_interval)

println("Saved times:")
for t in saved_times
    @printf("%.4f\n", t)
end

# Check if we have all expected intervals
expected = 0.0:0.001:0.015
for e in expected
    found = any(abs.(saved_times .- e) .< 1e-10)
    if !found
        println("MISSING: $e")
    else
        # println("Found: $e")
    end
end
