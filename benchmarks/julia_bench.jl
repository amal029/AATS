# benchmarks.jl
using DifferentialEquations
using DelayDiffEq    # <-- ADD THIS LINE
using BenchmarkTools
using Printf
using DataFrames, CSV

println("=======================================================")
println("  RUNNING SOTA JULIA (SciML) SYNCHRONOUS DDE SOLVER")
println("=======================================================\n")

# Global Tolerances
const RELTOL = 1e-10
const ABSTOL = 1e-10
const T_SPAN = (0.0, 5.0)

# ============================================================================
# 1. BENCHMARK DEFINITION: Advection-Reaction DPDE (N=2000)
# ============================================================================
const N_DPDE = 10000
const RHO_COEFF = 1.0
const TAU_DPDE_VAL = 0.5
const DX_INV_DPDE = N_DPDE + 1.0
const VELOCITY_DPDE = 1.0

function f_dpde!(du, u, h, p, t)
    for i in 1:N_DPDE
        left_idx = (i == 1) ? N_DPDE : (i - 1)
        dxdz = (u[i] - u[left_idx]) * DX_INV_DPDE
        
        u_delay = h(p, t - TAU_DPDE_VAL; idxs=i)
        reaction = u[i] * (1.0 - u_delay) * RHO_COEFF
        du[i] = -VELOCITY_DPDE * dxdz + reaction
    end
end

function history_dpde(p, t; idxs=nothing)
    if typeof(idxs) <: Number
        x_coord = (idxs) / DX_INV_DPDE
        # Squared sine wave!
        return (sin(pi * x_coord / 1.0))^2
    else
        # Squared sine wave inside the array!
        return [(sin(pi * (i) / DX_INV_DPDE))^2 for i in 1:N_DPDE]
    end
end

# ============================================================================
# 2. BENCHMARK DEFINITION: Sparse Multi-Rate Network (N=10000)
# ============================================================================
const N_MR = 10000
const TAU_MR_VAL = 0.5

function f_mr!(du, u, h, p, t)
    for i in 1:N_MR
        alpha = (i <= 100) ? 50.0 : 0.01
        decay = -alpha * u[i]
        coupling = 0.0
        
        i_cpp = i - 1
        rem = i_cpp % 100
        first_j_cpp = (rem == 0) ? 0 : (100 - rem)
        
        j_cpp = first_j_cpp
        while j_cpp < N_MR
            j = j_cpp + 1
            weight = 0.05 * cos(i_cpp + j_cpp)
            x_del = h(p, t - TAU_MR_VAL; idxs=j)
            coupling += (x_del / (1.0 + x_del^2)) * weight
            j_cpp += 100
        end
        du[i] = decay + coupling
    end
end

history_mr(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.1 : fill(0.1, N_MR)

# ============================================================================
# 3. BENCHMARK DEFINITION: State-Dependent Ring Network (N=1000)
# ============================================================================
const N_SD = 10000
const BETA_COEFF = 5.0
const GAMMA_COEFF = 1.0

function f_sd!(du, u, h, p, t)
    for i in 1:N_SD
        # Delay depends locally on u[i] instead of globally on u[1]
        x_local = u[i]
        tau_val = 1.0 + (0.5 * x_local) / (1.0 + x_local^2)
        
        prev_idx = (i == 1) ? N_SD : (i - 1)
        x_delayed_neighbor = h(p, t - tau_val; idxs=prev_idx)
        hill_denom = 1.0 + x_delayed_neighbor^3
        du[i] = (BETA_COEFF / hill_denom) - GAMMA_COEFF * u[i]
    end
end

history_sd(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.5 : fill(0.5, N_SD)

# ============================================================================
# 4. BENCHMARK DEFINITION: Propagating Pulse Chain (Dynamic Multi-Rate)
# ============================================================================
const N_PULSE = 10000
const TAU_PULSE_VAL = 0.5

function f_pulse!(du, u, h, p, t)
    du[1] = -u[1]
    for i in 2:N_PULSE
        x_prev = h(p, t - TAU_PULSE_VAL; idxs=i-1)
        activation = (2.0 * x_prev^2) / (0.25 + x_prev^2)
        du[i] = -u[i] + activation
    end
end

history_pulse(p, t; idxs=nothing) = begin
    if typeof(idxs) <: Number
        return (idxs == 1) ? 1.0 : 0.0
    else
        res = fill(0.0, N_PULSE)
        res[1] = 1.0
        return res
    end
end

# ============================================================================
# 5. BENCHMARK DEFINITION: Diffusion DPDE (N=10000)
# ============================================================================
const N_DIFF = 50 # Scale down to match C++ parity
const ALPHA_DIFF = 1.2
const BETA_DIFF = 0.4
const TAU_DIFF_VAL = 0.15
const DX_INV_DIFF = N_DIFF + 1.0

function f_diff!(du, u, h, p, t)
    for i in 1:N_DIFF
        # 1D Periodic boundary conditions matching your C++ topology
        left_idx  = (i == 1) ? N_DIFF : (i - 1)
        right_idx = (i == N_DIFF) ? 1 : (i + 1)
        
        # Second-order central space finite difference stencil
        # diffusion_stencil = ALPHA_DIFF * (u[right_idx] - 2.0 * u[i] + u[left_idx])
        diffusion_stencil = ALPHA_DIFF * (u[right_idx] - 2.0 * u[i] + u[left_idx]) *
        (DX_INV_DIFF * DX_INV_DIFF)
        
        # Scalar-sniped historical evaluation (bypasses vector allocations)
        u_delay = h(p, t - TAU_DIFF_VAL; idxs=i)
        delayed_loss = BETA_DIFF * u[i] * u_delay
        
        du[i] = diffusion_stencil - delayed_loss
    end
end

function history_diff(p, t; idxs=nothing)
    if typeof(idxs) <: Number
        return 0.0 # System was completely at rest prior to t = 0
    else
        return zeros(Float64, N_DIFF)
    end
end

# Call this function after each solve step.
# Example: export_julia_trajectories(sol_dpde, "dpde", 5.0)

function export_julia_trajectories(sol, prefix::String, T_max::Float64; 
                                   num_report_points::Int=1000, vars_to_log::Int=20)
    println(" -> Saving Julia diagnostic logs for [$prefix]...")
    
    # Create the exact same time grid used by the AATS C++ solver
    dt_report = T_max / num_report_points
    report_times = 0.0:dt_report:T_max

    # Initialize a DataFrame with the time column
    df = DataFrame(time = report_times)

    # Evaluate the continuous Julia solution at these exact times
    states = [sol(t) for t in report_times]

    # Map Julia's 1-based indexing to C++'s 0-based indexing for the headers
    # Ensure we don't try to log more variables than actually exist
    actual_vars_to_log = min(vars_to_log, length(states[1]))
    
    for j in 1:actual_vars_to_log
        # Create columns: x0, x1, x2...
        df[!, Symbol("x$(j-1)")] = [state[j] for state in states]
    end

    # Export to CSV
    filename = "$(prefix)_julia_trajectories.csv"
    CSV.write(filename, df)
    println("    * Saved $filename")
end

# ============================================================================
# CORE EXECUTION PIPELINE
# ============================================================================

# --- Benchmark 1 ---
println("Executing Benchmark 1 [Advection DPDE] (N=$N_DPDE)...")
prob1 = DDEProblem(f_dpde!, history_dpde(nothing, 0.0), history_dpde, T_SPAN; constant_lags=[TAU_DPDE_VAL])
# Compile run
solve(prob1, MethodOfSteps(Tsit5()), reltol=RELTOL, abstol=ABSTOL, save_everystep=false)
# Timed run
@time sol1 = solve(prob1, MethodOfSteps(Tsit5()), reltol=RELTOL, abstol=ABSTOL)
export_julia_trajectories(sol1, "dpde", 5.0)
println(" -> Benchmark 1 Complete.\n")

# --- Benchmark 2 ---
println("Executing Benchmark 2 [Multi-Rate] (N=$N_MR)...")
prob2 = DDEProblem(f_mr!, history_mr(nothing, 0.0), history_mr, T_SPAN; constant_lags=[TAU_MR_VAL])
solve(prob2, MethodOfSteps(Tsit5()), reltol=RELTOL, abstol=ABSTOL, save_everystep=false)
@time sol2 = solve(prob2, MethodOfSteps(Tsit5()), reltol=RELTOL, abstol=ABSTOL)
export_julia_trajectories(sol2, "multirate", 5.0)
println(" -> Benchmark 2 Complete.\n")

# --- Benchmark 3 ---
println("Executing Benchmark 3 [State-Dependent] (N=$N_SD)...")
# Note: State-dependent delays cannot use constant_lags.
prob3 = DDEProblem(f_sd!, history_sd(nothing, 0.0), history_sd, T_SPAN)
solve(prob3, MethodOfSteps(Tsit5()), reltol=RELTOL, abstol=ABSTOL, save_everystep=false)
@time sol3 = solve(prob3, MethodOfSteps(Tsit5()), reltol=RELTOL, abstol=ABSTOL)
export_julia_trajectories(sol3, "statedep", 5.0)
println(" -> Benchmark 3 Complete.\n")

# --- Benchmark 4 ---
println("Executing Benchmark 4 [Propagating Pulse] (N=$N_PULSE)...")
prob4 = DDEProblem(f_pulse!, history_pulse(nothing, 0.0), history_pulse, T_SPAN; constant_lags=[TAU_PULSE_VAL])
solve(prob4, MethodOfSteps(Tsit5()), reltol=RELTOL, abstol=ABSTOL, save_everystep=false)
@time sol4 = solve(prob4, MethodOfSteps(Tsit5()), reltol=RELTOL, abstol=ABSTOL)
export_julia_trajectories(sol4, "prop_pulse", 5.0)
println(" -> Benchmark 4 Complete.\n")

# --- Benchmark 5 ---
# 1. Generate the sharp localized Gaussian initial state
u0_diff = zeros(Float64, N_DIFF)
const center_diff = N_DIFF ÷ 2
const width_diff = 25.0

for i in 1:N_DIFF
    # Calculate 0-indexed distance equivalent to the C++ logic
    dist = Float64(i - 1) - Float64(center_diff)
    u0_diff[i] = exp(-(dist * dist) / (2.0 * width_diff * width_diff))
end

# 2. Problem Construction
tspan = (0.0, 5.0)
prob5 = DDEProblem(f_diff!, u0_diff, history_diff, tspan; constant_lags=[TAU_DIFF_VAL])

# 3. Execution (Using explicit Tsit5 to guarantee clean algorithmic comparison)
println("Executing Julia Benchmark 5 [Diffusion DPDE] (N=$N_DIFF)...")
solve(prob5, MethodOfSteps(Tsit5()), reltol=1e-6, abstol=1e-6)
@time sol5 = solve(prob5, MethodOfSteps(Tsit5()), reltol=1e-6, abstol=1e-6)
export_julia_trajectories(sol5, "diffusion", 5.0)
println("-> Benchmark 5 Complete.")

println("All Julia validations executed successfully.")