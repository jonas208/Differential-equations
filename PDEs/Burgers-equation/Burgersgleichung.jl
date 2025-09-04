using SparseArrays
using OrdinaryDiffEq
using GLMakie

# Spezifikationen des Problems
L = 20.0 # [m]
kinematic_viscosity = 1.0e-5

plot_size = (1000, 1000) # Auflösung der Plots
fps = 25 # Bildwiederholrate der Animationen

# analytische Lösung
u_exact_(x, t, kinematic_viscosity) = (1 + exp((x - 0.5*t) / (2*kinematic_viscosity)))^-1
u_exact(x, t, kinematic_viscosity) = u_exact_(x-2.5, t, kinematic_viscosity)

# Anfangsbedingung
# u0(x) = exp(-(x-5)^2/2)
u0(x) = u_exact(x, 0.0, kinematic_viscosity)

# Zeitintervall
tspan = (0.0, 1.0)
n_frames = ceil(Int, tspan[2]-tspan[1])*fps

# Ortsdiskretisierung
# n = 1000
n = 10^6
dx = L/n
xs = range(dx/2, L-(dx/2); length = n)

param = (n, dx, kinematic_viscosity)

function burgers_equation_fdm!(du, u, param, t)
    n, dx, kinematic_viscosity = param

    # u[1] = u_exact(dx/2, t, kinematic_viscosity)
    # u[n] = u_exact(L-(dx/2), t, kinematic_viscosity)
    
    for i in 2:n-1
        du[i] = -u[i] * ((u[i+1] - u[i-1]) / (2*dx))
        du[i] += kinematic_viscosity * (u[i+1] - 2*u[i] + u[i-1]) / dx^2
    end

    # println(t)
end

function burgers_equation_fvm!(du, u, param, t)
    n, dx, kinematic_viscosity = param

    # u[1] = u_exact(dx/2, t, kinematic_viscosity)
    # u[n] = u_exact(L-(dx/2), t, kinematic_viscosity)

    flux(u) = u^2 / 2

    flux_r = 0.0

    for i in 1:n
        if i == 1
            lam = max(abs(u[1]), abs(u[2]))
            flux_r = 0.5*(flux(u[1]) + flux(u[2]) - lam*(u[2] - u[1]))
        end

        if 1 < i < n
            flux_l = flux_r

            lam = max(abs(u[i]), abs(u[i+1]))
            flux_r = 0.5*(flux(u[i]) + flux(u[i+1]) - lam*(u[i+1] - u[i]))

            du[i] = -(flux_r - flux_l) / dx
            du[i] += kinematic_viscosity * (u[i+1] - 2*u[i] + u[i-1]) / dx^2
        end
    end

    println(t)
end

function jac_structure_fdm(n)
    jac = spzeros(n, n)
    for i in 2:n-1
        jac[i, i-1] = 1.0
        jac[i, i] = 1.0
        jac[i, i+1] = 1.0
    end
    return jac
end

function jac_structure_fvm(n)
    jac = spzeros(n, n)
    for i in 1:n
        if i == 1
            jac[i, 1] = 1.0
            jac[i, 2] = 1.0
        end

        if 1 < i < n
            jac[i, i-1] = 1.0
            jac[i, i] = 1.0
            jac[i, i+1] = 1.0
        end
    end
    return jac
end

u0s = u0.(xs)
solver = kinematic_viscosity <= 1.0e-2 ? Tsit5() : Rodas5P(autodiff=false)
tol = 1.0e-5

jac_prototype_fdm = jac_structure_fdm(n)
fdm = ODEFunction(burgers_equation_fdm!; jac_prototype = jac_prototype_fdm)
prob_fdm = ODEProblem(fdm, u0s, tspan, param)
sol_fdm = @time solve(prob_fdm, solver; reltol = tol, abstol = tol, saveat = tspan[1]:1/fps:tspan[2])

jac_prototype_fvm = jac_structure_fvm(n)
fvm = ODEFunction(burgers_equation_fvm!; jac_prototype = jac_prototype_fvm)
prob_fvm = ODEProblem(fvm, u0s, tspan, param)
sol_fvm = @time solve(prob_fvm, solver; reltol = tol, abstol = tol, saveat = tspan[1]:1/fps:tspan[2])

# Plotting

time = Observable(tspan[2])
us_fdm = @lift sol_fdm($time)
us_fvm = @lift sol_fvm($time)
us_exact = @lift u_exact.(xs, $time, kinematic_viscosity)

fig = Figure(size = plot_size)
ax = Axis(fig[1, 1], title = @lift("u(x, t); t = $(round($time, digits = 2))"), limits = (0, L, minimum(u0s), 1.5*maximum(u0s)))
lin_fdm = lines!(xs, us_fdm)
lin_fvm = lines!(xs, us_fvm)
lin_exact = lines!(xs, us_exact)
Legend(fig[1, 1], 
        # [lin_fdm, lin_fvm], ["FDM", "FVM"], 
        [lin_fdm, lin_fvm, lin_exact], ["FDM", "FVM", "analytische Lösung"], 
        tellheight = false,
        tellwidth = false,
        margin = (10, 10, 10, 10),
        halign = :right, valign = :top, orientation = :horizontal)
display(fig)

time[] = 2.5

record(fig, "burgers_equation.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do t
    time[] = t
end







using CUDA, BenchmarkTools

function fdm_kernel!(du, u, param, t)
    n, dx, kinematic_viscosity = param

    i = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    # @cuprintln("$i")

    if Int32(1) < i < n
        du[i] = -u[i] * ((u[i+1] - u[i-1]) / (2*dx))
        du[i] += kinematic_viscosity * (u[i+1] - 2*u[i] + u[i-1]) / dx^2
    end

    return nothing
end

function burgers_equation_fdm_gpu!(du, u, param, t)
    n, dx, kinematic_viscosity = param

    kernel = @cuda launch=false fdm_kernel!(du, u, param, t)
    config = CUDA.launch_configuration(kernel.fun)

    threads = config.threads

    x_threads = min(n, threads)
    x_blocks = cld(n, x_threads)

    kernel(du, u, param, t, threads = x_threads, blocks = x_blocks)

    # println(t)
end

param_gpu = (Int32(n), Float32(dx), Float32(kinematic_viscosity))

du_gpu = CUDA.zeros(size(u0s)...)
u_gpu = CUDA.rand(size(u0s)...)

t_gpu = 0.5f0

@btime CUDA.@sync burgers_equation_fdm_gpu!(du_gpu, u_gpu, param_gpu, t_gpu)

param_cpu = (Int32(n), Float32(dx), Float32(kinematic_viscosity))

du_cpu = zeros(Float32, size(u0s)...)
u_cpu = Array(u_gpu)

t_cpu = 0.5f0

@btime burgers_equation_fdm!(du_cpu, u_cpu, param_cpu, t_cpu)

@info isapprox(Array(du_gpu), du_cpu)

prob_fdm_gpu = ODEProblem(burgers_equation_fdm_gpu!, cu(u0s), Float32.(tspan), param_gpu)
sol_fdm_gpu = CUDA.@time solve(prob_fdm_gpu, Tsit5(); reltol = tol, abstol = tol, saveat = tspan[1]:1/fps:tspan[2])

prob_fdm_cpu = ODEProblem(burgers_equation_fdm!, u0s, tspan, param)
sol_fdm_cpu = @time solve(prob_fdm_cpu, Tsit5(); reltol = tol, abstol = tol, saveat = tspan[1]:1/fps:tspan[2])

#=
n, dx, kinematic_viscosity = param_gpu

kernel = @cuda launch=false fdm_kernel!(du_gpu, u_gpu, param_gpu, t_gpu)
config = CUDA.launch_configuration(kernel.fun)

threads = config.threads

x_threads = min(n, threads)
x_blocks = cld(n, x_threads)

kernel(du, u, param, t, threads = x_threads, blocks = x_blocks)
=#