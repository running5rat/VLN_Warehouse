# Official Altro.jl solver, isolated from Python/Isaac dependencies.
# Input contains only source-time observations and constant-velocity predictions.
using Altro, TrajectoryOptimization, RobotDynamics, StaticArrays, LinearAlgebra, JSON3
using ForwardDiff, FiniteDiff
const RD = RobotDynamics

RD.@autodiff struct WarehouseDrive <: RD.ContinuousDynamics end
RD.state_dim(::WarehouseDrive) = 5
RD.control_dim(::WarehouseDrive) = 2
RD.dynamics(::WarehouseDrive, x, u) = SA[x[4]*cos(x[3]), x[4]*sin(x[3]), x[5], u[1], u[2]]

request = JSON3.read(read(ARGS[1], String))
N = Int(request.horizon)+1
dt = Float64(request.dt)
x0 = SVector{5}(Float64.(request.state))
limits = request.limits
model = WarehouseDrive()
candidates = Any[]
for seed in request.seeds
    started = time()
    try
        xf = SVector{5}(Float64.(seed.goal))
        obj = LQRObjective(Diagonal(SA[1.,1.,.2,.5,.2]), Diagonal(SA[.1,.1]),
                           Diagonal(SA[50.,50.,2.,5.,2.]), xf, N)
        constraints = ConstraintList(5,2,N)
        bound = BoundConstraint(5,2,
            x_min=SA[-Inf,-Inf,-Inf,0.,-Float64(limits.w_max)],
            x_max=SA[Inf,Inf,Inf,Float64(limits.v_max),Float64(limits.w_max)],
            u_min=SA[-Float64(limits.a_max),-Float64(limits.alpha_max)],
            u_max=SA[Float64(limits.a_max),Float64(limits.alpha_max)])
        add_constraint!(constraints,bound,1:N-1)
        terminal_bound = BoundConstraint(5,2,
            x_min=SA[-Inf,-Inf,-Inf,0.,-Float64(limits.w_max)],
            x_max=SA[Inf,Inf,Inf,Float64(limits.v_max),Float64(limits.w_max)])
        add_constraint!(constraints,terminal_bound,N)
        for k in 1:N
            circles = [Float64.(c) for c in request.static_circles]
            for tr in request.predictions
                push!(circles, [Float64(tr.xy[k][1]),Float64(tr.xy[k][2]),Float64(tr.radius[k])])
            end
            if !isempty(circles)
                con = CircleConstraint(5,[c[1] for c in circles],[c[2] for c in circles],
                    [c[3]+Float64(limits.radius) for c in circles])
                add_constraint!(constraints,con,k:k)
            end
        end
        prob = Problem(model,obj,x0,(N-1)*dt,xf=xf,constraints=constraints)
        initial_controls!(prob,[SVector{2}(Float64.(u)) for u in seed.controls])
        opts = SolverOptions(projected_newton=false, constraint_tolerance=1e-5,
                             cost_tolerance=1e-5, iterations=300, iterations_outer=20,
                             penalty_initial=1., penalty_scaling=10., verbose=0)
        solver = ALTROSolver(prob,opts)
        solve!(solver)
        # Projected Newton is disabled. ALTRO 0.5.0's aggregate accessor also
        # inspects its unused PN trajectory, whose initial values are stale.
        # Check the AL-iLQR trajectory that actually produced these candidates.
        violation = max_violation(solver.solver_al)
        push!(candidates,Dict("mode_index"=>Int(seed.mode_index),
            "solver_status"=>isfinite(violation) && violation<=1e-4 ? "solved" : "unconverged",
            "max_violation"=>isfinite(violation) ? violation : nothing,
            "states"=>[collect(x) for x in states(solver)],
            "controls"=>[collect(u) for u in controls(solver)],
            "elapsed_seconds"=>time()-started))
    catch err
        push!(candidates,Dict("mode_index"=>Int(seed.mode_index), "solver_status"=>"failed",
                              "error"=>sprint(showerror,err),"elapsed_seconds"=>time()-started))
    end
end
open(ARGS[2],"w") do io
    JSON3.write(io,Dict("source_binding"=>request.source_binding,
        "solver"=>"Altro.jl 0.5.0 / AL-iLQR; projected Newton disabled",
        "candidates"=>candidates))
end
