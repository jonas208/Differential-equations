using OrdinaryDiffEq
using CUDA
using GLMakie

# Spezifikationen des Problems
type = Float32
L = type(1.0) # [m]
heat_capacity_ratio = type(1.4)

plot_size = (1000, 1000) # Auflösung der Plots
fps = 25 # Bildwiederholrate der Animationen

# Anfangsbedingung
u0(x) = x <= type(0.5) ? [type(1.0), type(0.0), type(2.5)] : [type(0.125), type(0.0), type(0.25)] # Sod-Problem
# u0(x) = x <= type(0.5) ? [type(0.445), type(0.311), type(8.928)] : [type(0.5), type(0.0), type(1.4275)] # Lax-Problem

function u0!(u::AbstractMatrix{T}, n::Int64, dx::T) where T
    for i in 1:n
        u[i, :] .= u0((i-T(0.5))*dx)
    end
end

# Zeitintervall
tspan = (type(0.0), type(0.16))
# tspan = (type(0.0), type(5.0))
n_frames = ceil(Int, tspan[2]-tspan[1])*fps

# Ortsdiskretisierung
# n = 500
n = 5000
dx = L/n
xs = range(dx/2, L-(dx/2); length = n)

fluxes_cpu = zeros(type, n-3, 3)
param_cpu = (L, heat_capacity_ratio, n, dx, fluxes_cpu)

fluxes_gpu = CuArray(fluxes_cpu)
param_gpu = (L, heat_capacity_ratio, n, dx, fluxes_gpu)

get_vars(u::AbstractVector{T}) where T = u[1], u[2]/u[1], u[3]
get_pressure(rho::T, v::T, E::T, heat_capacity_ratio::T) where T = (heat_capacity_ratio - 1) * (E - T(0.5)*rho*v^2)
get_abs_lam(rho::T, v::T, p::T, heat_capacity_ratio::T) where T = abs(v) + sqrt(max(T(0.0), heat_capacity_ratio*p / rho))
get_flux(rho::T, v::T, E::T, p::T) where T = [rho*v, rho*v^2 + p, v*(E + p)]

get_vars_gpu(rho::T, rho_v::T, E::T) where T = rho, rho_v/rho, E
get_flux_1_gpu(rho::T, v::T, E::T, p::T) where T = rho*v
get_flux_2_gpu(rho::T, v::T, E::T, p::T) where T = rho*v^2 + p
get_flux_3_gpu(rho::T, v::T, E::T, p::T) where T = v*(E + p)

function recover_cweno(x, x_i, dx, u) #-- size(u) = [3, 3]
    return [cweno(x, x_i, dx, u[:, 1]), 
            cweno(x, x_i, dx, u[:, 2]),
            cweno(x, x_i, dx, u[:, 3])]
end

function cweno(x::T, x_i::T, dx::T, u::AbstractVector{T}) where T <: AbstractFloat
    ep = T(1.0e-6); p = T(0.6);
    uL = u[2]-u[1]; uC = u[3]-2*u[2]+u[1]; uR = u[3]-u[2]; uCC = u[3]-u[1];
    ISL = uL^2; ISC = T(13/3)*uC^2 + T(0.25)*uCC^2; ISR = uR^2;
    aL = T(0.25)*(1/(ep+ISL))^p; aC = T(0.5)*(1/(ep+ISC))^p; aR = T(0.25)*(1/(ep+ISR))^p;
    suma = max(aL+aC+aR,eps(T(1.0))); 
    wL = aL/suma; wC = aC/suma; wR = aR/suma;
    pL = u[2] + uL/dx*(x-x_i);
    pC = u[2] - uC/12 + uCC/(2*dx)*(x-x_i) + uC/dx^2*(x-x_i)^2;
    pR = u[2] + uR/dx*(x-x_i);
    return wL*pL + wC*pC + wR*pR
end

function cweno_gpu(x::T, x_i::T, dx::T, u1::T, u2::T, u3::T) where T <: AbstractFloat
    ep = T(1.0e-6); p = T(0.6);
    uL = u2-u1; uC = u3-2*u2+u1; uR = u3-u2; uCC = u3-u1;
    ISL = uL^2; ISC = T(13/3)*uC^2 + T(0.25)*uCC^2; ISR = uR^2;
    aL = T(0.25)*(1/(ep+ISL))^p; aC = T(0.5)*(1/(ep+ISC))^p; aR = T(0.25)*(1/(ep+ISR))^p;
    suma = max(aL+aC+aR,eps(T(1.0))); 
    wL = aL/suma; wC = aC/suma; wR = aR/suma;
    pL = u2 + uL/dx*(x-x_i);
    pC = u2 - uC/12 + uCC/(2*dx)*(x-x_i) + uC/dx^2*(x-x_i)^2;
    pR = u2 + uR/dx*(x-x_i);
    return wL*pL + wC*pC + wR*pR
end

function local_lax_friedrichs(u_l_recover::AbstractVector{T}, u_r_recover::AbstractVector{T}, heat_capacity_ratio::T) where T <: AbstractFloat
    rho_l_recover, v_l_recover, E_l_recover = get_vars(u_l_recover)
    p_l_recover = get_pressure(rho_l_recover, v_l_recover, E_l_recover, heat_capacity_ratio)

    rho_r_recover, v_r_recover, E_r_recover = get_vars(u_r_recover)
    p_r_recover = get_pressure(rho_r_recover, v_r_recover, E_r_recover, heat_capacity_ratio)

    lam = max(get_abs_lam(rho_l_recover, v_l_recover, p_l_recover, heat_capacity_ratio), 
                get_abs_lam(rho_r_recover, v_r_recover, p_r_recover, heat_capacity_ratio))
    flux_r = T(0.5)*(get_flux(rho_l_recover, v_l_recover, E_l_recover, p_l_recover) .+ get_flux(rho_r_recover, v_r_recover, E_r_recover, p_r_recover)
                .- lam*(u_r_recover .- u_l_recover))
    
    return flux_r
end

function euler_equations_fvm_cpu!(du::AbstractMatrix{T}, u::AbstractMatrix{T}, param::Tuple{T, T, Int64, T, AbstractMatrix{T}}, t::T) where T
    L, heat_capacity_ratio, n, dx, fluxes = param
    
    Threads.@threads for i in 2:n-2
        u_l_recover = recover_cweno(i*dx, (i-T(0.5))*dx, dx, view(u, i-1:i+1, :))
        u_r_recover = recover_cweno(i*dx, (i+T(0.5))*dx, dx, view(u, i:i+2, :))
        flux_r = local_lax_friedrichs(u_l_recover, u_r_recover, heat_capacity_ratio)
        fluxes[i-1, :] .= flux_r
    end

    Threads.@threads for i in 3:n-2
        flux_l = view(fluxes, i-2, :)
        flux_r = view(fluxes, i-1, :)
        @. du[i, :] = -(flux_r - flux_l) / dx
    end
end

function calculate_fluxes_kernel!(fluxes::AbstractMatrix{T}, u::AbstractMatrix{T}, heat_capacity_ratio::T, n::Int64, dx::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if 2 <= i <= n-2
        x, x_i = i*dx, (i-T(0.5))*dx

        rho1, rho2, rho3 = u[i-1, 1], u[i, 1], u[i+1, 1]
        rho_l_recover = cweno_gpu(x, x_i, dx, rho1, rho2, rho3)

        rho_v1, rho_v2, rho_v3 = u[i-1, 2], u[i, 2], u[i+1, 2]
        rho_v_l_recover = cweno_gpu(x, x_i, dx, rho_v1, rho_v2, rho_v3)

        E1, E2, E3 = u[i-1, 3], u[i, 3], u[i+1, 3]
        E_l_recover = cweno_gpu(x, x_i, dx, E1, E2, E3)


        x, x_i = i*dx, (i+T(0.5))*dx

        rho1, rho2, rho3 = u[i, 1], u[i+1, 1], u[i+2, 1]
        rho_r_recover = cweno_gpu(x, x_i, dx, rho1, rho2, rho3)

        rho_v1, rho_v2, rho_v3 = u[i, 2], u[i+1, 2], u[i+2, 2]
        rho_v_r_recover = cweno_gpu(x, x_i, dx, rho_v1, rho_v2, rho_v3)

        E1, E2, E3 = u[i, 3], u[i+1, 3], u[i+2, 3]
        E_r_recover = cweno_gpu(x, x_i, dx, E1, E2, E3)


        rho_l_recover, v_l_recover, E_l_recover = get_vars_gpu(rho_l_recover, rho_v_l_recover, E_l_recover)
        p_l_recover = get_pressure(rho_l_recover, v_l_recover, E_l_recover, heat_capacity_ratio)

        rho_r_recover, v_r_recover, E_r_recover = get_vars_gpu(rho_r_recover, rho_v_r_recover, E_r_recover)
        p_r_recover = get_pressure(rho_r_recover, v_r_recover, E_r_recover, heat_capacity_ratio)


        lam = max(get_abs_lam(rho_l_recover, v_l_recover, p_l_recover, heat_capacity_ratio), 
                get_abs_lam(rho_r_recover, v_r_recover, p_r_recover, heat_capacity_ratio))
        flux_r_1 = T(0.5)*(get_flux_1_gpu(rho_l_recover, v_l_recover, E_l_recover, p_l_recover) + get_flux_1_gpu(rho_r_recover, v_r_recover, E_r_recover, p_r_recover)
                - lam*(rho_r_recover - rho_l_recover))
        flux_r_2 = T(0.5)*(get_flux_2_gpu(rho_l_recover, v_l_recover, E_l_recover, p_l_recover) + get_flux_2_gpu(rho_r_recover, v_r_recover, E_r_recover, p_r_recover)
                - lam*(rho_v_r_recover - rho_v_l_recover))
        flux_r_3 = T(0.5)*(get_flux_3_gpu(rho_l_recover, v_l_recover, E_l_recover, p_l_recover) + get_flux_3_gpu(rho_r_recover, v_r_recover, E_r_recover, p_r_recover)
                - lam*(E_r_recover - E_l_recover))


        fluxes[i-1, 1] = flux_r_1
        fluxes[i-1, 2] = flux_r_2
        fluxes[i-1, 3] = flux_r_3
    end

    return nothing
end

function fvm_kernel!(du::AbstractMatrix{T}, fluxes::AbstractMatrix{T}, n::Int64, dx::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if 3 <= i <= n-2
        for j in 1:3
            flux_l = fluxes[i-2, j]
            flux_r = fluxes[i-1, j]
            du[i, j] = -(flux_r - flux_l) / dx
        end
    end

    return nothing
end

function euler_equations_fvm_gpu!(du::AbstractMatrix{T}, u::AbstractMatrix{T}, param::Tuple{T, T, Int64, T, AbstractMatrix{T}}, t::T) where T
    L, heat_capacity_ratio, n, dx, fluxes = param

    fluxes_kernel = @cuda launch=false calculate_fluxes_kernel!(fluxes, u, heat_capacity_ratio, n, dx)
    config = CUDA.launch_configuration(fluxes_kernel.fun)

    threads = config.threads

    x_threads = min(n, threads)
    x_blocks = cld(n, x_threads)

    fluxes_kernel(fluxes, u, heat_capacity_ratio, n, dx, threads = x_threads, blocks = x_blocks)


    fvm_kernel = @cuda launch=false fvm_kernel!(du, fluxes, n, dx)
    config = CUDA.launch_configuration(fvm_kernel.fun)

    threads = config.threads

    x_threads = min(n, threads)
    x_blocks = cld(n, x_threads)

    fvm_kernel(du, fluxes, n, dx, threads = x_threads, blocks = x_blocks)
end

function apply_boundary_conditions!(u::AbstractMatrix{T}) where T
    # Neumann
    u[2, :] = u[3, :]
    u[1, :] = u[2, :]
    u[end-1, :] = u[end-2, :]
    u[end, :] = u[end-1, :]
end

function apply_boundary_conditions_kernel!(u::AbstractMatrix{T}) where T
    j = threadIdx().x

    # Neumann
    u[2, j] = u[3, j]
    u[1, j] = u[2, j]
    u[end-1, j] = u[end-2, j]
    u[end, j] = u[end-1, j]

    return nothing
end

function apply_boundary_conditions!(u::CuMatrix{T}) where T
    @cuda threads=3 apply_boundary_conditions_kernel!(u)
    return nothing
end

function callback!(integrator)
    apply_boundary_conditions!(integrator.u)
    # println(integrator.t)
end

u0s_cpu = zeros(type, n, 3)
u0!(u0s_cpu, n, dx)
apply_boundary_conditions!(u0s_cpu)

u0s_gpu = CuArray(u0s_cpu)

#=
using InteractiveUtils
using BenchmarkTools

@info Threads.nthreads()

u_cpu = u0s_cpu
u_gpu = u0s_gpu
du_cpu = zeros(type, size(u_cpu))
du_gpu = CuArray(du_cpu)
t = tspan[1]

# @code_warntype euler_equations_fvm_cpu!(du_cpu, u_cpu, param_cpu, t)
@benchmark euler_equations_fvm_cpu!($du_cpu, $u_cpu, $param_cpu, $t)
euler_equations_fvm_cpu!(du_cpu, u_cpu, param_cpu, t)

# @code_warntype euler_equations_fvm_gpu!(du_gpu, u_gpu, param_gpu, t)
@benchmark CUDA.@sync euler_equations_fvm_gpu!($du_gpu, $u_gpu, $param_gpu, $t)
euler_equations_fvm_gpu!(du_gpu, u_gpu, param_gpu, t)

@info isapprox(du_cpu, Array(du_gpu))
=#

solver = SSPRK432()
# tol = 1.0e-8
tol = 1.0e-5

condition(u, t, integrator) = true
cb = DiscreteCallback(condition, callback!; save_positions = (false, false))

f_cpu = ODEFunction(euler_equations_fvm_cpu!)
prob_cpu = ODEProblem(f_cpu, u0s_cpu, tspan, param_cpu)
sol_cpu = @time solve(prob_cpu, solver; reltol = tol, abstol = tol, callback = cb)

f_gpu = ODEFunction(euler_equations_fvm_gpu!)
prob_gpu = ODEProblem(f_gpu, u0s_gpu, tspan, param_gpu)
sol_gpu = @time solve(prob_gpu, solver; reltol = tol, abstol = tol, callback = cb)

display(sol_cpu.stats)
display(sol_gpu.stats)

# Interface für Lösung

function get_vars_from_solution(t)
    # u = sol_cpu(t)
    u = Array(sol_gpu(t))
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

rhos[], vs[], Es[], ps[] = get_vars_from_solution(type(0.0))

GLMakie.record(fig, "eulergleichungen_1d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do t
    rhos[], vs[], Es[], ps[] = get_vars_from_solution(t)
end