using OrdinaryDiffEq
using GLMakie

# Spezifikationen des Problems
L = 1.0 # [m]
heat_capacity_ratio = 1.4

plot_size = (1000, 1000) # Auflösung der Plots
fps = 25 # Bildwiederholrate der Animationen

# Anfangsbedingung
u0(x) = x <= 0.5 ? [1.0, 0.0, 2.5] : [0.125, 0.0, 0.25] # Sod-Problem
# u0(x) = x <= 0.5 ? [0.445, 0.311, 8.928] : [0.5, 0.0, 1.4275] # Lax-Problem

function u0!(u, n, dx)
    for i in 1:n
        u[i, :] .= u0((i-0.5)*dx)
    end
end

# Zeitintervall
tspan = (0.0, 0.16)
# tspan = (0.0, 5.0)
n_frames = ceil(Int, tspan[2]-tspan[1])*fps

# Ortsdiskretisierung
n = 500
dx = L/n
xs = range(dx/2, L-(dx/2); length = n)

param_1 = (L, heat_capacity_ratio, n, dx)
fluxes = zeros(n-3, 3)
param_2 = (L, heat_capacity_ratio, n, dx, fluxes)

get_vars(u) = u[1], u[2]/u[1], u[3]
get_pressure(rho, v, E, heat_capacity_ratio) = (heat_capacity_ratio - 1) * (E - 0.5*rho*v^2)
get_abs_lam(rho, v, p, heat_capacity_ratio) = abs(v) + sqrt(max(0.0, heat_capacity_ratio*p / rho))
get_flux(rho, v, E, p) = [rho*v, rho*v^2 + p, v*(E + p)]

function recover_cweno(x, x_i, dx, u) #-- size(u) = [3, 3]
    return [cweno(x, x_i, dx, u[:, 1]), 
            cweno(x, x_i, dx, u[:, 2]),
            cweno(x, x_i, dx, u[:, 3])]
end

function cweno(x, x_i, dx, u)
    ep = 1.0e-6; p = 0.6;
    uL = u[2]-u[1]; uC = u[3]-2*u[2]+u[1]; uR = u[3]-u[2]; uCC = u[3]-u[1];
    ISL = uL^2; ISC = 13/3*uC^2 + 0.25*uCC^2; ISR = uR^2;
    aL = 0.25*(1/(ep+ISL))^p; aC = 0.5*(1/(ep+ISC))^p; aR = 0.25*(1/(ep+ISR))^p;
    suma = max(aL+aC+aR,eps(1.0)); 
    wL = aL/suma; wC = aC/suma; wR = aR/suma;
    pL = u[2] + uL/dx*(x-x_i);
    pC = u[2] - uC/12 + uCC/(2*dx)*(x-x_i) + uC/dx^2*(x-x_i)^2;
    pR = u[2] + uR/dx*(x-x_i);
    return wL*pL + wC*pC + wR*pR
end

function local_lax_friedrichs(u_l_recover, u_r_recover, heat_capacity_ratio)
    rho_l_recover, v_l_recover, E_l_recover = get_vars(u_l_recover)
    p_l_recover = get_pressure(rho_l_recover, v_l_recover, E_l_recover, heat_capacity_ratio)

    rho_r_recover, v_r_recover, E_r_recover = get_vars(u_r_recover)
    p_r_recover = get_pressure(rho_r_recover, v_r_recover, E_r_recover, heat_capacity_ratio)

    lam = max(get_abs_lam(rho_l_recover, v_l_recover, p_l_recover, heat_capacity_ratio), 
                get_abs_lam(rho_r_recover, v_r_recover, p_r_recover, heat_capacity_ratio))
    flux_r = 0.5*(get_flux(rho_l_recover, v_l_recover, E_l_recover, p_l_recover) .+ get_flux(rho_r_recover, v_r_recover, E_r_recover, p_r_recover)
                .- lam*(u_r_recover .- u_l_recover))
    
    return flux_r
end

function euler_equations_fvm_1!(du, u, param, t)
    L, heat_capacity_ratio, n, dx = param

    flux_r = zeros(3)
    
    for i in 2:n-2
        if i == 2
            u_l_recover = recover_cweno(i*dx, (i-0.5)*dx, dx, view(u, i-1:i+1, :))
            u_r_recover = recover_cweno(i*dx, (i+0.5)*dx, dx, view(u, i:i+2, :))
            flux_r = local_lax_friedrichs(u_l_recover, u_r_recover, heat_capacity_ratio)
        end

        if 3 <= i <= n-2
            flux_l = flux_r

            u_l_recover = recover_cweno(i*dx, (i-0.5)*dx, dx, view(u, i-1:i+1, :))
            u_r_recover = recover_cweno(i*dx, (i+0.5)*dx, dx, view(u, i:i+2, :))
            flux_r = local_lax_friedrichs(u_l_recover, u_r_recover, heat_capacity_ratio)

            @. du[i, :] = -(flux_r - flux_l) / dx
        end
    end
end

function euler_equations_fvm_2!(du, u, param, t)
    L, heat_capacity_ratio, n, dx, fluxes = param
    
    Threads.@threads for i in 2:n-2
        u_l_recover = recover_cweno(i*dx, (i-0.5)*dx, dx, view(u, i-1:i+1, :))
        u_r_recover = recover_cweno(i*dx, (i+0.5)*dx, dx, view(u, i:i+2, :))
        flux_r = local_lax_friedrichs(u_l_recover, u_r_recover, heat_capacity_ratio)
        fluxes[i-1, :] .= flux_r
    end

    Threads.@threads for i in 3:n-2
        flux_l = view(fluxes, i-2, :)
        flux_r = view(fluxes, i-1, :)
        @. du[i, :] = -(flux_r - flux_l) / dx
    end
end

function apply_boundary_conditions!(u)
    # Neumann
    u[2, :] = u[3, :]
    u[1, :] = u[2, :]
    u[end-1, :] = u[end-2, :]
    u[end, :] = u[end-1, :]
end

function callback!(integrator)
    apply_boundary_conditions!(integrator.u)
    # println(integrator.t)
end

u0s = zeros(n, 3)
u0!(u0s, n, dx)
apply_boundary_conditions!(u0s)

#=
using InteractiveUtils
using BenchmarkTools

@info Threads.nthreads()

u = u0s
du_1 = zeros(size(u))
du_2 = copy(du_1)
t = tspan[1]

# @code_warntype euler_equations_fvm_1!(du_1, u, param_1, t)
@benchmark euler_equations_fvm_1!($du_1, $u, $param_1, $t)
euler_equations_fvm_1!(du_1, u, param_1, t)

# @code_warntype euler_equations_fvm_2!(du_2, u, param_2, t)
@benchmark euler_equations_fvm_2!($du_2, $u, $param_2, $t)
euler_equations_fvm_2!(du_2, u, param_2, t)

@info isapprox(du_1, du_2)
=#

solver = SSPRK432()
# tol = 1.0e-8
tol = 1.0e-5

f = ODEFunction(euler_equations_fvm_2!)
prob = ODEProblem(f, u0s, tspan, param_2)
condition(u, t, integrator) = true
cb = DiscreteCallback(condition, callback!)
sol = @time solve(prob, solver; reltol = tol, abstol = tol, callback = cb)

display(sol.stats)

# Interface für Lösung

function get_vars_from_solution(t)
    u = sol(t)
    rhos = u[:, 1]
    vs = u[:, 2] ./ rhos
    Es = u[:, 3]
    ps = get_pressure.(rhos, vs, Es, Ref(heat_capacity_ratio))
    return rhos, vs, Es, ps
end

# Plotting

time = tspan[2]
rhos, vs, Es, ps = Observable.(get_vars_from_solution(time))

fig = Figure(size = plot_size)
ax = Axis(fig[1, 1], title = "Numerische Lösung der 1D-Eulergleichungen mit FVM")
# ax = Axis(fig[1, 1], title = "Numerische Lösung der 1D-Eulergleichungen mit FVM", limits = (0.0, L, -0.2, 4.2))
lin_rho = lines!(xs, rhos)
lin_v = lines!(xs, vs)
lin_p = lines!(xs, ps)
Legend(fig[1, 1], 
        [lin_rho, lin_v, lin_p], ["Dichte", "Geschwindigkeit", "Druck"], 
        tellheight = false,
        tellwidth = false,
        margin = (10, 10, 10, 10),
        halign = :right, valign = :top, orientation = :horizontal)
save("eulergleichungen_1d.png", fig)
display(fig)

rhos[], vs[], Es[], ps[] = get_vars_from_solution(0.0)

GLMakie.record(fig, "eulergleichungen_1d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do t
    rhos[], vs[], Es[], ps[] = get_vars_from_solution(t)
end