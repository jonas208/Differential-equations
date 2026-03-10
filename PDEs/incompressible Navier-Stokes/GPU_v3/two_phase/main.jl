using LinearAlgebra
using CUDA
using CUDA.CUSPARSE
using CUDSS
using OrdinaryDiffEq
using DiffEqCallbacks
using Interpolations
using GLMakie

include("pressure.jl")
include("velocity.jl")
include("saving.jl")

# Spezifikationen des Problems
type = Float32
Lx = type(12.5) # Seitenlänge in x-Richtung der rechteckigen Oberfläche [m]
Ly = type(3.0) # Seitenlänge in y-Richtung der rechteckigen Oberfläche [m]
kinematic_viscosity = type(1.5e-5)
density = type(1.0)
horizontal_velocity = type(2.5)
simulate_smoke = true

max_plot_size = (1920, 1080) # maximale Auflösung der Plots
fps = 25 # Bildwiederholrate der Animationen

# Zeitintervall
tspan = (type(0.0), type(5.0))
n_frames = ceil(Int, tspan[2]-tspan[1])*fps

# Anfangsbedingungen
vx0(x::T, y::T) where T = T(0.0)
vy0(x::T, y::T) where T = T(0.0)
c0(x::T, y::T) where T = T(0.0)

# Randbedingungen 
# Wände oben und unten (Dirichlet für Geschwindigkeit und Neumann für Druck)
# Zufluss links (Dirichlet für Geschwindigkeit und Neumann für Druck)
# Abfluss rechts (Neumann für Geschwindigkeit und Dirichlet für Druck)

# Ortsdiskretisierung
nx = 2500
ny = 600
dx = Lx/nx
dy = Ly/ny
xs = range(dx/2, Lx-dx/2; length = nx)
ys = range(dy/2, Ly-dy/2; length = ny)

factor_plot_size = min(max_plot_size[1]/nx, max_plot_size[2]/ny)
plot_size = (factor_plot_size*nx, factor_plot_size*ny)

using Images
img = load("obstacles/A5_mask_medium_long.png")
img = imrotate(img, π/2)
img = collect(img)
mask = BitMatrix(alpha.(img) .== 1)
# heatmap(mask)

is_solid_cpu = mask
is_solid_gpu = CuArray(is_solid_cpu)

# Koeffizientenmatrix für die Poisson-Gleichung für den Druck
rowPtr_cpu, colVal_cpu, nzVal_cpu = get_coefficient_mat_cpu(nx, ny, dx, dy, is_solid_cpu, ones(type, nx, ny))
rowPtr_gpu, colVal_gpu, nzVal_gpu = CuVector(rowPtr_cpu), CuVector(colVal_cpu), CuVector(nzVal_cpu)

coefficient_mat_gpu = CuSparseMatrixCSR(rowPtr_gpu, colVal_gpu, nzVal_gpu, (nx * ny, nx * ny))
pressure_vec_gpu = CUDA.zeros(type, nx*ny)
divergence_mat_gpu = CUDA.zeros(type, nx, ny)

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
fluxes_x_gpu = CuArray{type, 3}(undef, 3, nx-3, ny-3)
fluxes_y_gpu = CuArray{type, 3}(undef, 3, nx-3, ny-3)
param_gpu = (Lx, Ly, kinematic_viscosity, density, horizontal_velocity, simulate_smoke, is_solid_gpu, nx, ny, dx, dy, coefficient_mat_gpu, pressure_vec_gpu, divergence_mat_gpu, lin_solve_gpu, use_weno, ep, p, fluxes_x_gpu, fluxes_y_gpu)

vx0s_gpu = CuMatrix([vx0(x, y) for x in xs, y in ys])
vy0s_gpu = CuMatrix([vy0(x, y) for x in xs, y in ys])
c0s_gpu = CuMatrix([c0(x, y) for x in xs, y in ys])

apply_boundary_conditions!(vx0s_gpu, vy0s_gpu, c0s_gpu, horizontal_velocity, is_solid_gpu)

u0_gpu = vcat(vec(vx0s_gpu), vec(vy0s_gpu), vec(c0s_gpu))

function fluxes_kernel!(fluxes_x::CuDeviceArray{T, 3}, fluxes_y::CuDeviceArray{T, 3}, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, c::CuDeviceMatrix{T}, kinematic_viscosity::T, simulate_smoke, nx, ny, dx::T, dy::T, use_weno, ep::T, p::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if 2 <= i <= nx-2 && 2 <= j <= ny-2
        vx_l, vx_r, vy_l, vy_r, c_l, c_r = recover_x(i, j, dx, vx, vy, c, simulate_smoke, use_weno, ep, p)
        flux_x_1_r, flux_x_2_r, flux_x_smoke_r = local_lax_friedrichs_x(vx_l, vx_r, vy_l, vy_r, c_l, c_r)
        # flux_x_1_r, flux_x_2_r, flux_x_smoke_r = hybrid_llf_upwind_x(vx_l, vx_r, vy_l, vy_r, c_l, c_r)

        vx_l, vx_r, vy_l, vy_r, c_l, c_r = recover_y(i, j, dy, vx, vy, c, simulate_smoke, use_weno, ep, p)
        flux_y_1_r, flux_y_2_r, flux_y_smoke_r = local_lax_friedrichs_y(vx_l, vx_r, vy_l, vy_r, c_l, c_r)
        # flux_y_1_r, flux_y_2_r, flux_y_smoke_r = hybrid_llf_upwind_y(vx_l, vx_r, vy_l, vy_r, c_l, c_r)

        fluxes_x[1, i-1, j-1] = flux_x_1_r - kinematic_viscosity*(vx[i+1, j]-vx[i, j])/dx
        fluxes_x[2, i-1, j-1] = flux_x_2_r - kinematic_viscosity*(vy[i+1, j]-vy[i, j])/dx
        fluxes_x[3, i-1, j-1] = flux_x_smoke_r

        fluxes_y[1, i-1, j-1] = flux_y_1_r - kinematic_viscosity*(vx[i, j+1]-vx[i, j])/dy
        fluxes_y[2, i-1, j-1] = flux_y_2_r - kinematic_viscosity*(vy[i, j+1]-vy[i, j])/dy
        fluxes_y[3, i-1, j-1] = flux_y_smoke_r
    end

    return nothing
end

function fvm_kernel!(dvx::CuDeviceMatrix{T}, dvy::CuDeviceMatrix{T}, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, dc::CuDeviceMatrix{T}, fluxes_x::CuDeviceArray{T, 3}, fluxes_y::CuDeviceArray{T, 3}, is_solid::CuDeviceMatrix{Bool}, kinematic_viscosity::T, simulate_smoke, nx, ny, dx::T, dy::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if 3 <= i <= nx-2 && 3 <= j <= ny-2
        if !is_solid[i, j]
            # laplacian_vx = (vx[i+1, j] - 2*vx[i, j] + vx[i-1, j])/dx^2 + (vx[i, j+1] - 2*vx[i, j] + vx[i, j-1])/dy^2
            # laplacian_vy = (vy[i+1, j] - 2*vy[i, j] + vy[i-1, j])/dx^2 + (vy[i, j+1] - 2*vy[i, j] + vy[i, j-1])/dy^2

            flux_x_1_l = fluxes_x[1, i-2, j-1]
            flux_x_2_l = fluxes_x[2, i-2, j-1]
            flux_x_smoke_l = fluxes_x[3, i-2, j-1]
            flux_x_1_r = fluxes_x[1, i-1, j-1]
            flux_x_2_r = fluxes_x[2, i-1, j-1]
            flux_x_smoke_r = fluxes_x[3, i-1, j-1]

            flux_y_1_l = fluxes_y[1, i-1, j-2]
            flux_y_2_l = fluxes_y[2, i-1, j-2]
            flux_y_smoke_l = fluxes_y[3, i-1, j-2]
            flux_y_1_r = fluxes_y[1, i-1, j-1]
            flux_y_2_r = fluxes_y[2, i-1, j-1]
            flux_y_smoke_r = fluxes_y[3, i-1, j-1]

            dvx[i, j] = -(flux_x_1_r - flux_x_1_l)/dx - (flux_y_1_r - flux_y_1_l)/dy # + kinematic_viscosity*laplacian_vx
            dvy[i, j] = -(flux_x_2_r - flux_x_2_l)/dx - (flux_y_2_r - flux_y_2_l)/dy # + kinematic_viscosity*laplacian_vy
            dc[i, j] = -(flux_x_smoke_r - flux_x_smoke_l)/dx - (flux_y_smoke_r - flux_y_smoke_l)/dy
        end
    end

    return nothing
end

function navier_stokes_fvm_gpu!(du::CuVector{T}, u::CuVector{T}, param, t::T) where T
    Lx, Ly, kinematic_viscosity, density, horizontal_velocity, simulate_smoke, is_solid, nx, ny, dx, dy, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param

    vx = reshape(view(u, 1:nx*ny), nx, ny)
    vy = reshape(view(u, nx*ny+1:2*nx*ny), nx, ny)
    c = reshape(view(u, 2*nx*ny+1:3*nx*ny), nx, ny)

    dvx = reshape(view(du, 1:nx*ny), nx, ny)
    dvy = reshape(view(du, nx*ny+1:2*nx*ny), nx, ny)
    dc = reshape(view(du, 2*nx*ny+1:3*nx*ny), nx, ny)

    fluxes_kernel = @cuda launch=false fluxes_kernel!(fluxes_x, fluxes_y, vx, vy, c, kinematic_viscosity, simulate_smoke, nx, ny, dx, dy, use_weno, ep, p)
    config = CUDA.launch_configuration(fluxes_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(nx, threads_per_dim)
    x_blocks = cld(nx, x_threads)

    y_threads = min(ny, threads_per_dim)
    y_blocks = cld(ny, y_threads)

    fluxes_kernel(fluxes_x, fluxes_y, vx, vy, c, kinematic_viscosity, simulate_smoke, nx, ny, dx, dy, use_weno, ep, p, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))


    fvm_kernel = @cuda launch=false fvm_kernel!(dvx, dvy, vx, vy, dc, fluxes_x, fluxes_y, is_solid, kinematic_viscosity, simulate_smoke, nx, ny, dx, dy)
    config = CUDA.launch_configuration(fvm_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(nx, threads_per_dim)
    x_blocks = cld(nx, x_threads)

    y_threads = min(ny, threads_per_dim)
    y_blocks = cld(ny, y_threads)

    fvm_kernel(dvx, dvy, vx, vy, dc, fluxes_x, fluxes_y, is_solid, kinematic_viscosity, simulate_smoke, nx, ny, dx, dy, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

condition(u, t, integrator) = true
# solver = SSPRK432()
solver = Tsit5()
# tol = 1.0e-5
tol = 1.0e-4
# tol = 1.0e-3
saveat = tspan[1]:type(1/fps):tspan[2]

f_gpu = ODEFunction(navier_stokes_fvm_gpu!)
prob_gpu = ODEProblem(f_gpu, u0_gpu, tspan, param_gpu)
pressure_cb_gpu = DiscreteCallback(condition, pressure_correction_gpu!; save_positions = (false, false))
saved_values_gpu = SavedValues(type, NTuple{4, CuMatrix{type}})
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

function get_smoke(t)
    i = get_index(t)
    smoke = Array(saveval[i][4])
    return smoke
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

cs = @lift get_smoke($t)

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
save("plots/fluid_2d.png", fig)
display(fig)

t[] = type(2.5)

GLMakie.record(fig, "plots/fluid_2d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do time
    t[] = time
end
=#

# Plotting the streamlines

us_itp = @lift interpolate($us, BSpline(Linear()))
vs_itp = @lift interpolate($vs, BSpline(Linear()))
us_etp = @lift extrapolate($us_itp, Flat())
vs_etp = @lift extrapolate($vs_itp, Flat())
vel_interp(x, y; field, dx, dy) = Point2(field[1](x/dx, y/dy), field[2](x/dx, y/dy))
sf = @lift (x, y) -> vel_interp(x, y; field = ($us_etp, $vs_etp), dx = dx, dy = dy)

fig = Figure(size = plot_size)
ax = GLMakie.Axis(fig[1, 1], title = "Strömungslinien")
streamplot!(sf, 0..Lx, 0..Ly, colorscale = identity, colormap = :viridis)
save("plots/fluid_stream_2d.png", fig)
display(fig)

t[] = type(tspan[2])

GLMakie.record(fig, "plots/fluid_stream_2d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do time
    t[] = time
end

# Plotting the norm of the velocity field

velocity_norm = @lift sqrt.($us .^ 2 .+ $vs .^ 2)

fig = Figure(size = plot_size)
ax = GLMakie.Axis(fig[1, 1], title = "Betrag der Geschwindigkeit")
heatmap!(velocity_norm)
save("plots/fluid_norm_2d.png", fig)

t[] = type(2.5)

GLMakie.record(fig, "plots/fluid_norm_2d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do time
    t[] = time
end

# Plotting the get_smoke

fig = Figure(size = plot_size)
ax = GLMakie.Axis(fig[1, 1], title = "Rauch")
heatmap!(cs)
image!(rotr90(load("obstacles/A5_mask_medium_long.png")))
save("plots/smoke_2d.png", fig)

t[] = type(2.5)

GLMakie.record(fig, "plots/smoke_2d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do time
    t[] = time
end