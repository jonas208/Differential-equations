using LinearAlgebra
using SparseArrays
using LinearSolve
using Sparspak
using CUDA
using CUDA.CUSPARSE
using CUDSS
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks
using Interpolations
using GLMakie

include("obstacles.jl")
include("pressure.jl")
include("velocity.jl")
include("saving.jl")

# Spezifikationen des Problems
type = Float32
Lx = type(15.0) # Seitenlänge in x-Richtung der rechteckigen Oberfläche [m]
Ly = type(5.0) # Seitenlänge in y-Richtung der rechteckigen Oberfläche [m]
kinematic_viscosity = type(1.0e-6)
density = type(1.0)
# horizontal_velocity = type(0.75)
horizontal_velocity = type(4.0)

max_plot_size = (1920, 1080) # maximale Auflösung der Plots
fps = 25 # Bildwiederholrate der Animationen

# Zeitintervall
tspan = (type(0.0), type(5.0))
# tspan = (type(0.0), type(15.0))
n_frames = ceil(Int, tspan[2]-tspan[1])*fps

# Anfangsbedingungen
vx0(x::T, y::T) where T = T(0.0)
vy0(x::T, y::T) where T = T(0.0)

# Randbedingungen 
# Wände oben und unten (Dirichlet für Geschwindigkeit und Neumann für Druck)
# Zufluss links (Dirichlet für Geschwindigkeit und Neumann für Druck)
# Abfluss rechts (Neumann für Geschwindigkeit und Dirichlet für Druck)

# Ortsdiskretisierung
# nx = 160
# ny = 160
nx = 3750
ny = 750
dx = Lx/nx
dy = Ly/ny
xs = range(dx/2, Lx-dx/2; length = nx)
ys = range(dy/2, Ly-dy/2; length = ny)

factor_plot_size = min(max_plot_size[1]/nx, max_plot_size[2]/ny)
plot_size = (factor_plot_size*nx, factor_plot_size*ny)

#=
obstacles = [
    Rectangle((0.3, 0.3), 0.35, 0.15, dx, dy),
    Rectangle((0.3, 0.7), 0.35, 0.15, dx, dy)
]
=#
obstacles = [Rectangle((2.0, 2.5), 2.5, 1.0, dx, dy)]
# obstacles = []

# Koeffizientenmatrix für die Poisson-Gleichung für den Druck
pressure_vec_cpu = zeros(type, nx*ny)
divergence_mat_cpu = zeros(type, nx, ny)

coefficient_mat_cpu = get_coefficient_mat_cpu(divergence_mat_cpu, nx, ny, dx, dy, obstacles)

coefficient_mat_gpu = CuSparseMatrixCSR(coefficient_mat_cpu)
pressure_vec_gpu = CUDA.zeros(type, nx*ny)
divergence_mat_gpu = CuArray(divergence_mat_cpu)

# Definiere ein lineares Gleichungssystem der Form Ax = b
A_cpu = coefficient_mat_cpu
b_cpu = vec(divergence_mat_cpu)
lin_prob_cpu = LinearProblem(A_cpu, b_cpu)
lin_solver_cpu = type == Float64 ? KLUFactorization() : SparspakFactorization()
lin_solve_cpu = init(lin_prob_cpu, lin_solver_cpu)
solve!(lin_solve_cpu)

# Definiere ein lineares Gleichungssystem der Form Ax = b
A_gpu = coefficient_mat_gpu
x_gpu = pressure_vec_gpu
b_gpu = vec(divergence_mat_gpu)
lin_solve_gpu = CudssSolver(A_gpu, "G", 'F')
cudss_set(lin_solve_gpu, "ir_n_steps", 1)
cudss("analysis", lin_solve_gpu, x_gpu, b_gpu)
cudss("factorization", lin_solve_gpu, x_gpu, b_gpu)
cudss("solve", lin_solve_gpu, x_gpu, b_gpu)

# WENO-Parameter
use_weno = true
ep = type(1.0e-6); p = type(0.6);

# Parameter
fluxes_x_cpu = Array{type, 3}(undef, 2, nx-3, ny-3)
fluxes_y_cpu = Array{type, 3}(undef, 2, nx-3, ny-3)
param_cpu = (Lx, Ly, kinematic_viscosity, density, horizontal_velocity, obstacles, nx, ny, dx, dy, coefficient_mat_cpu, pressure_vec_cpu, divergence_mat_cpu, lin_solve_cpu, use_weno, ep, p, fluxes_x_cpu, fluxes_y_cpu)

fluxes_x_gpu = CuArray{type, 3}(undef, 2, nx-3, ny-3)
fluxes_y_gpu = CuArray{type, 3}(undef, 2, nx-3, ny-3)
param_gpu = (Lx, Ly, kinematic_viscosity, density, horizontal_velocity, obstacles, nx, ny, dx, dy, coefficient_mat_gpu, pressure_vec_gpu, divergence_mat_gpu, lin_solve_gpu, use_weno, ep, p, fluxes_x_gpu, fluxes_y_gpu)

vx0s = [vx0(x, y) for x in xs, y in ys]
vy0s = [vy0(x, y) for x in xs, y in ys]

apply_boundary_conditions!(vx0s, vy0s, horizontal_velocity, obstacles)

u0_cpu = vcat(vec(vx0s), vec(vy0s))
u0_gpu = CuArray(u0_cpu)

# DGL-System erster Ordnung (Linienmethode mit finiten Volumen)
function navier_stokes_fvm_cpu!(du::AbstractVector{T}, u::AbstractVector{T}, param, t::T) where T
    Lx, Ly, kinematic_viscosity, density, horizontal_velocity, obstacles, nx, ny, dx, dy, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param

    vx = reshape(view(u, 1:nx*ny), nx, ny)
    vy = reshape(view(u, nx*ny+1:2*nx*ny), nx, ny)

    dvx = reshape(view(du, 1:nx*ny), nx, ny)
    dvy = reshape(view(du, nx*ny+1:2*nx*ny), nx, ny)

    Threads.@threads for j in 2:ny-2
        for i in 2:nx-2
            vx_l, vx_r, vy_l, vy_r = recover_x(i, j, dx, vx, vy, use_weno, ep, p)
            flux_x_1_r, flux_x_2_r = local_lax_friedrichs_x(vx_l, vx_r, vy_l, vy_r)

            vx_l, vx_r, vy_l, vy_r = recover_y(i, j, dy, vx, vy, use_weno, ep, p)
            flux_y_1_r, flux_y_2_r = local_lax_friedrichs_y(vx_l, vx_r, vy_l, vy_r)

            fluxes_x[1, i-1, j-1] = flux_x_1_r
            fluxes_x[2, i-1, j-1] = flux_x_2_r

            fluxes_y[1, i-1, j-1] = flux_y_1_r
            fluxes_y[2, i-1, j-1] = flux_y_2_r
        end
    end

    Threads.@threads for j in 3:ny-2
        for i in 3:nx-2
            if !position_in_obstacle(obstacles, i, j)
                laplacian_vx = (vx[i+1, j] - 2*vx[i, j] + vx[i-1, j])/dx^2 + (vx[i, j+1] - 2*vx[i, j] + vx[i, j-1])/dy^2
                laplacian_vy = (vy[i+1, j] - 2*vy[i, j] + vy[i-1, j])/dx^2 + (vy[i, j+1] - 2*vy[i, j] + vy[i, j-1])/dy^2

                flux_x_1_l = fluxes_x[1, i-2, j-1]
                flux_x_2_l = fluxes_x[2, i-2, j-1]
                flux_x_1_r = fluxes_x[1, i-1, j-1]
                flux_x_2_r = fluxes_x[2, i-1, j-1]

                flux_y_1_l = fluxes_y[1, i-1, j-2]
                flux_y_2_l = fluxes_y[2, i-1, j-2]
                flux_y_1_r = fluxes_y[1, i-1, j-1]
                flux_y_2_r = fluxes_y[2, i-1, j-1]

                dvx[i, j] = -(flux_x_1_r - flux_x_1_l)/dx - (flux_y_1_r - flux_y_1_l)/dy + kinematic_viscosity*laplacian_vx
                dvy[i, j] = -(flux_x_2_r - flux_x_2_l)/dx - (flux_y_2_r - flux_y_2_l)/dy + kinematic_viscosity*laplacian_vy
            end
        end
    end
end

function fluxes_kernel!(fluxes_x::CuDeviceArray{T, 3}, fluxes_y::CuDeviceArray{T, 3}, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, nx, ny, dx::T, dy::T, use_weno, ep::T, p::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if 2 <= i <= nx-2 && 2 <= j <= ny-2
        vx_l, vx_r, vy_l, vy_r = recover_x(i, j, dx, vx, vy, use_weno, ep, p)
        flux_x_1_r, flux_x_2_r = local_lax_friedrichs_x(vx_l, vx_r, vy_l, vy_r)

        vx_l, vx_r, vy_l, vy_r = recover_y(i, j, dy, vx, vy, use_weno, ep, p)
        flux_y_1_r, flux_y_2_r = local_lax_friedrichs_y(vx_l, vx_r, vy_l, vy_r)

        fluxes_x[1, i-1, j-1] = flux_x_1_r
        fluxes_x[2, i-1, j-1] = flux_x_2_r

        fluxes_y[1, i-1, j-1] = flux_y_1_r
        fluxes_y[2, i-1, j-1] = flux_y_2_r
    end

    return nothing
end

function fvm_kernel!(dvx::CuDeviceMatrix{T}, dvy::CuDeviceMatrix{T}, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, fluxes_x::CuDeviceArray{T, 3}, fluxes_y::CuDeviceArray{T, 3}, obstacle_indices::CuDeviceVector, kinematic_viscosity::T, nx, ny, dx::T, dy::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if 3 <= i <= nx-2 && 3 <= j <= ny-2
        in_obstacle = false
        k = 1
        while !in_obstacle && k <= length(obstacle_indices)-3
            i0 = obstacle_indices[k]
            i1 = obstacle_indices[k+1]
            j0 = obstacle_indices[k+2]
            j1 = obstacle_indices[k+3]
            if i0 <= i <= i1 && j0 <= j <= j1
                in_obstacle = true
            end
            k += 4
        end
        
        if !in_obstacle
            laplacian_vx = (vx[i+1, j] - 2*vx[i, j] + vx[i-1, j])/dx^2 + (vx[i, j+1] - 2*vx[i, j] + vx[i, j-1])/dy^2
            laplacian_vy = (vy[i+1, j] - 2*vy[i, j] + vy[i-1, j])/dx^2 + (vy[i, j+1] - 2*vy[i, j] + vy[i, j-1])/dy^2

            flux_x_1_l = fluxes_x[1, i-2, j-1]
            flux_x_2_l = fluxes_x[2, i-2, j-1]
            flux_x_1_r = fluxes_x[1, i-1, j-1]
            flux_x_2_r = fluxes_x[2, i-1, j-1]

            flux_y_1_l = fluxes_y[1, i-1, j-2]
            flux_y_2_l = fluxes_y[2, i-1, j-2]
            flux_y_1_r = fluxes_y[1, i-1, j-1]
            flux_y_2_r = fluxes_y[2, i-1, j-1]

            dvx[i, j] = -(flux_x_1_r - flux_x_1_l)/dx - (flux_y_1_r - flux_y_1_l)/dy + kinematic_viscosity*laplacian_vx
            dvy[i, j] = -(flux_x_2_r - flux_x_2_l)/dx - (flux_y_2_r - flux_y_2_l)/dy + kinematic_viscosity*laplacian_vy
        end
    end

    return nothing
end

function navier_stokes_fvm_gpu!(du::CuVector{T}, u::CuVector{T}, param, t::T) where T
    Lx, Ly, kinematic_viscosity, density, horizontal_velocity, obstacles, nx, ny, dx, dy, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param

    vx = reshape(view(u, 1:nx*ny), nx, ny)
    vy = reshape(view(u, nx*ny+1:2*nx*ny), nx, ny)

    dvx = reshape(view(du, 1:nx*ny), nx, ny)
    dvy = reshape(view(du, nx*ny+1:2*nx*ny), nx, ny)

    fluxes_kernel = @cuda launch=false fluxes_kernel!(fluxes_x, fluxes_y, vx, vy, nx, ny, dx, dy, use_weno, ep, p)
    config = CUDA.launch_configuration(fluxes_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(nx, threads_per_dim)
    x_blocks = cld(nx, x_threads)

    y_threads = min(ny, threads_per_dim)
    y_blocks = cld(ny, y_threads)

    fluxes_kernel(fluxes_x, fluxes_y, vx, vy, nx, ny, dx, dy, use_weno, ep, p, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))


    obstacle_indices = CuArray(get_obstacle_indices(obstacles))

    fvm_kernel = @cuda launch=false fvm_kernel!(dvx, dvy, vx, vy, fluxes_x, fluxes_y, obstacle_indices, kinematic_viscosity, nx, ny, dx, dy)
    config = CUDA.launch_configuration(fvm_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(nx, threads_per_dim)
    x_blocks = cld(nx, x_threads)

    y_threads = min(ny, threads_per_dim)
    y_blocks = cld(ny, y_threads)

    fvm_kernel(dvx, dvy, vx, vy, fluxes_x, fluxes_y, obstacle_indices, kinematic_viscosity, nx, ny, dx, dy, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

condition(u, t, integrator) = true
solver = SSPRK432()
# tol = 1.0e-5
tol = 1.0e-4
# tol = 1.0e-3
saveat = tspan[1]:type(1/fps):tspan[2]

#=
f_cpu = ODEFunction(navier_stokes_fvm_cpu!)
prob_cpu = ODEProblem(f_cpu, u0_cpu, tspan, param_cpu)
pressure_cb_cpu = DiscreteCallback(condition, pressure_correction_cpu!; save_positions = (false, false))
saved_values_cpu = SavedValues(type, NTuple{3, Matrix{type}})
saving_cb_cpu = SavingCallback(save_func, saved_values_cpu; saveat = saveat, save_everystep = false, save_start = false, save_end = false)
cb_cpu = CallbackSet(pressure_cb_cpu, saving_cb_cpu)
sol_cpu = @time solve(prob_cpu, solver; reltol = tol, abstol = tol, callback = cb_cpu, save_everystep = false, save_start = false, save_end = false)

display(sol_cpu.stats)
=#

f_gpu = ODEFunction(navier_stokes_fvm_gpu!)
prob_gpu = ODEProblem(f_gpu, u0_gpu, tspan, param_gpu)
pressure_cb_gpu = DiscreteCallback(condition, pressure_correction_gpu!; save_positions = (false, false))
saved_values_gpu = SavedValues(type, NTuple{3, CuMatrix{type}})
saving_cb_gpu = SavingCallback(save_func, saved_values_gpu; saveat = saveat, save_everystep = false, save_start = false, save_end = false)
cb_gpu = CallbackSet(pressure_cb_gpu, saving_cb_gpu)
sol_gpu = CUDA.@time solve(prob_gpu, solver; reltol = tol, abstol = tol, callback = cb_gpu, save_everystep = false, save_start = false, save_end = false)

display(sol_gpu.stats)

# Interface für Lösung

# saved_values = saved_values_cpu
saved_values = saved_values_gpu
saveval = saved_values.saveval

function get_index(t)
    i = Int(round(t*fps))
    if i < 1
        i = 1
    end
    if i > length(saveval)
        i = length(saveval)
    end
    return i
end

function get_vx(t)
    i = get_index(t)
    vx = Array(saveval[i][1])
    return vx
end

function get_vy(t)
    i = get_index(t)
    vy = Array(saveval[i][2])
    return vy
end

function get_pressure(t)
    i = get_index(t)
    pressure = Array(saveval[i][3])
    return pressure
end

# Plotting the velocity field

t = Observable(tspan[2])
us = @lift get_vx($t)
vs = @lift get_vy($t)
strength = @lift vec(sqrt.($us .^ 2 .+ $vs .^ 2))

pressure = @lift get_pressure($t)
itp = @lift interpolate($pressure, BSpline(Linear()))
etp = @lift extrapolate($itp, Flat())
pressure_interp = @lift [$etp(x/dx, y/dy) for x in xs, y in ys]

#=
fig = Figure(size = plot_size)
ax = Axis(fig[1, 1], title = "Vektorfeld der Fließgeschwindigkeiten")
heatmap!(xs, ys, pressure_interp, colormap = :balance) # colormap = :balance
#=
arrows!(xs, ys, us, vs; 
        lengthscale = 0.02, normalize = true, 
        arrowsize = 10, linewidth = 2.5,
        arrowcolor = strength, linecolor = strength,
        colormap = :viridis)
=#
arrows2d!(xs, ys, us, vs; 
        lengthscale = 0.02, normalize = true, 
        # arrowsize = 10, linewidth = 2.5,
        color = strength, colormap = :viridis)
save("fluid_2d.png", fig)
display(fig)

t[] = type(2.5)

GLMakie.record(fig, "fluid_2d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do time
    t[] = time
end

# Plotting the streamlines

us_itp = @lift interpolate($us, BSpline(Linear()))
vs_itp = @lift interpolate($vs, BSpline(Linear()))
us_etp = @lift extrapolate($us_itp, Flat())
vs_etp = @lift extrapolate($vs_itp, Flat())
vel_interp(x, y; field, dx, dy) = Point2(field[1](x/dx, y/dy), field[2](x/dx, y/dy))
sf = @lift (x, y) -> vel_interp(x, y; field = ($us_etp, $vs_etp), dx = dx, dy = dy)

fig = Figure(size = plot_size)
ax = Axis(fig[1, 1], title = "Strömungslinien")
streamplot!(sf, 0..L, 0..L, colorscale = identity, colormap = :viridis)
save("fluid_stream_2d.png", fig)
display(fig)

t[] = type(tspan[2])

GLMakie.record(fig, "fluid_stream_2d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do time
    t[] = time
end
=#

# Plotting the norm of the velocity field

velocity_norm = @lift sqrt.($us .^ 2 .+ $vs .^ 2)

fig = Figure(size = plot_size)
ax = Axis(fig[1, 1], title = "Betrag der Geschwindigkeit")
heatmap!(velocity_norm)
save("fluid_norm_2d.png", fig)

t[] = type(2.5)

GLMakie.record(fig, "fluid_norm_2d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do time
    t[] = time
end