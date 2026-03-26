using CUDA
using CUDA.CUSPARSE
using CUDSS
using OrdinaryDiffEq
using DiffEqCallbacks
using Interpolations
using GLMakie

include("pressure.jl")
include("velocities.jl")
include("saving.jl")

#=  Staggered Grid inklusive Rand- bzw. Ghost-Zellen

    →       →       →       →       →       →       →       →       →

↑       ↑       ↑       ↑       ↑       ↑       ↑       ↑       ↑       ↑

    →       →       →       →       →       →       →       →       →    

↑       ↑   + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +   ↑       ↑
            |       |       |       |       |       |       |
    →       →   •   →   •   →   •   →   •   →   •   →   •   →       →
            |       |       |       |       |       |       |
↑       ↑   + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +   ↑       ↑
            |       |       |       |       |       |       |
    →       →   •   →   •   →   •   →   •   →   •   →   •   →       →
            |       |       |       |       |       |       |
↑       ↑   + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +   ↑       ↑
            |       |       |       |       |       |       |
    →       →   •   →   •   →   •   →   •   →   •   →   •   →       →
            |       |       |       |       |       |       |
↑       ↑   0 - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +   ↑       ↑

    →       →       →       →       →       →       →       →       →

↑       ↑       ↑       ↑       ↑       ↑       ↑       ↑       ↑       ↑

    →       →       →       →       →       →       →       →       →



nx = nx_pressure = 6
ny = ny_pressure = 3

nx_vx = nx + 3 = 9
ny_vx = ny + 4 = 7

nx_vy = nx + 4 = 10
ny_vy = ny + 3 = 6

"•" sind die Mittelpunkte der Druckzellen (nicht versetzt)
"→" sind die Mittelpunkte der Zellen der horizontalen Geschwindigkeitskomponenten (versetzt)
"↑" sind die Mittelpunkte der Zellen der vertikalen Geschwindigkeitskomponenten (versetzt)

Für jeden Rand gibt es pro Geschwindigkeitskomponente zwei Rand- bzw. Ghost-Zellen, 
wobei in der zur jeweiligen Komponente gehörigen Dimension nur eine echte Ghost-Zelle nötig ist,
weil die Zelle davor exakt auf dem Rand liegt. Ansonsten sind es zwei echte Ghost-Zellen.
=#

# Gibt zu dem Index einer Druckvariable den Index der rechts daneben liegenden horizontalen Geschwindigkeitskomponente zurück
get_right_vx_index(i, j) = (i + 2, j + 2)
# Gibt zu dem Index einer Druckvariable den Index der links daneben liegenden horizontalen Geschwindigkeitskomponente zurück
get_left_vx_index(i, j) = (i + 1, j + 2)

# Gibt zu dem Index einer Druckvariable den Index der oberhalb liegenden vertikalen Geschwindigkeitskomponente zurück
get_top_vy_index(i, j) = (i + 2, j + 2)
# Gibt zu dem Index einer Druckvariable den Index der unterhalb liegenden vertikalen Geschwindigkeitskomponente zurück
get_bottom_vy_index(i, j) = (i + 2, j + 1)


# Gibt zu dem Index einer horizontalen Geschwindigkeitskomponente den Index der rechts daneben liegenden Druckvariable zurück
get_right_pressure_index(i, j) = (i - 1, j - 2)
# Gibt zu dem Index einer horizontalen Geschwindigkeitskomponente den Index der links daneben liegenden Druckvariable zurück
get_left_pressure_index(i, j) = (i - 2, j - 2)

# Gibt zu dem Index einer vertikalen Geschwindigkeitskomponente den Index der oberhalb liegenden Druckvariable zurück
get_top_pressure_index(i, j) = (i - 2, j - 1)
# Gibt zu dem Index einer vertikalen Geschwindigkeitskomponente den Index der unterhalb liegenden Druckvariable zurück
get_bottom_pressure_index(i, j) = (i - 2, j - 2)


# Gibt zu dem Index einer horizontalen Geschwindigkeitskomponente den Index der oberhalb liegenden, nach links versetzten vertikalen Geschwindigkeitskomponenten zurück
get_top_left_vy_index(i, j) = (i, j)
# Gibt zu dem Index einer horizontalen Geschwindigkeitskomponente den Index der oberhalb liegenden, nach rechts versetzten vertikalen Geschwindigkeitskomponenten zurück
get_top_right_vy_index(i, j) = (i + 1, j)

# Gibt zu dem Index einer vertikalen Geschwindigkeitskomponente den Index der rechts daneben liegenden, nach unten versetzten horizontalen Geschwindigkeitskomponenten zurück
get_right_bottom_vx_index(i, j) = (i, j)
# Gibt zu dem Index einer vertikalen Geschwindigkeitskomponente den Index der rechts daneben liegenden, nach oben versetzten horizontalen Geschwindigkeitskomponenten zurück
get_right_top_vx_index(i, j) = (i, j + 1)


# Gibt die Position des Zellmittelpunkts einer Druckvariable zurück
get_pressure_position(i, j, dx::T, dy::T) where T = ((i - T(0.5)) * dx, (j - T(0.5)) * dy)
# Gibt die Position des Zellmittelpunkts einer horizontalen Geschwindigkeitskomponente zurück
get_vx_position(i, j, dx::T, dy::T) where T = ((i - 2) * dx, (j - T(2.5)) * dy)
# Gibt die Position des Zellmittelpunkts einer vertikalen Geschwindigkeitskomponente zurück
get_vy_position(i, j, dx::T, dy::T) where T = ((i - T(2.5)) * dx, (j - 2) * dy)

struct Parameters{T<:AbstractFloat}
    Lx::T # Seitenlänge in x-Richtung der rechteckigen Oberfläche [m]
    Ly::T # Seitenlänge in y-Richtung der rechteckigen Oberfläche [m]
    dynamic_viscosity::T # [Pa*s]
    density::T # [kg/m^3]
    kinematic_viscosity::T # [m^2/s]
    horizontal_velocity::T # Einströmungsgeschwindigkeit am linken Rand [m/s]

    dx::T # [m]
    dy::T # [m]

    nx_pressure::Int
    ny_pressure::Int
    lin_solve::CudssSolver{T,Int32}
    coefficient_mat::CuSparseMatrixCSR{T,Int32}
    pressure::CuMatrix{T}
    divergence::CuMatrix{T}

    nx_vx::Int
    ny_vx::Int
    fluxes_vx::CuArray{T,3}

    nx_vy::Int
    ny_vy::Int
    fluxes_vy::CuArray{T,3}

    # WENO-Parameter
    ep::T
    p::T

    function Parameters(; Lx::T, Ly::T, dynamic_viscosity::T, density::T, horizontal_velocity::T, nx, ny, ep::T, p::T) where T<:AbstractFloat
        dx = Lx / nx
        dy = Ly / ny

        coefficient_mat = get_coefficient_mat(nx, ny, dx, dy, density)
        pressure = CUDA.zeros(T, nx, ny)
        divergence = CUDA.zeros(T, nx, ny)

        # Definiere ein lineares Gleichungssystem der Form Ax = b
        A = coefficient_mat
        x = vec(pressure)
        b = vec(divergence)
        lin_solve = CudssSolver(A, "G", 'F')
        cudss_set(lin_solve, "ir_n_steps", 1)
        cudss("analysis", lin_solve, x, b)
        cudss("factorization", lin_solve, x, b)

        nx_vx = nx + 3
        ny_vx = ny + 4
        fluxes_vx = CUDA.zeros(T, nx_vx - 3, ny_vx - 3, 2)

        nx_vy = nx + 4
        ny_vy = ny + 3
        fluxes_vy = CUDA.zeros(T, nx_vy - 3, ny_vy - 3, 2)

        new{T}(
            Lx, Ly, dynamic_viscosity, density, dynamic_viscosity / density, horizontal_velocity,
            dx, dy,
            nx, ny, lin_solve, coefficient_mat, pressure, divergence,
            nx_vx, ny_vx, fluxes_vx,
            nx_vy, ny_vy, fluxes_vy,
            ep, p
        )
    end
end

function fluxes_vx_kernel!(fluxes_vx::CuDeviceArray{T,3}, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, kinematic_viscosity::T, dx::T, dy::T, nx_vx, ny_vx, ep::T, p::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # fast nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    # beginne jeweils bei 2, sodass auch der linke bzw. untere Fluss in die jeweils erste innere Zelle berechnet wird
    if 2 <= i <= nx_vx - 2 && 2 <= j <= ny_vx - 2
        # berechne die vertikale Geschwindigkeitskomponente am oberen Rand der aktuellen Zelle der horizontalen Geschwindigkeitskomponente
        i_, j_ = get_top_left_vy_index(i, j)
        vy_top_left = vy[i_, j_]

        i_, j_ = get_top_right_vy_index(i, j)
        vy_top_right = vy[i_, j_]

        vy_top = T(0.5) * (vy_top_left + vy_top_right)

        vx_l, vx_r = recover_x_vx(i, j, dx, dy, vx, ep, p)
        flux_x = local_lax_friedrichs_x_vx(vx_l, vx_r)

        vx_l, vx_r = recover_y_vx(i, j, dx, dy, vx, ep, p)
        flux_y = local_lax_friedrichs_y_vx(vx_l, vx_r, vy_top)

        fluxes_vx[i-1, j-1, 1] = flux_x - kinematic_viscosity * (vx[i+1, j] - vx[i, j]) / dx

        fluxes_vx[i-1, j-1, 2] = flux_y - kinematic_viscosity * (vx[i, j+1] - vx[i, j]) / dy
    end

    return nothing
end

function fvm_vx_kernel!(dvx::CuDeviceMatrix{T}, fluxes_vx::CuDeviceArray{T,3}, nx_vx, ny_vx, dx::T, dy::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    if 3 <= i <= nx_vx - 2 && 3 <= j <= ny_vx - 2
        flux_x_l = fluxes_vx[i-2, j-1, 1]
        flux_x_r = fluxes_vx[i-1, j-1, 1]

        flux_y_l = fluxes_vx[i-1, j-2, 2]
        flux_y_r = fluxes_vx[i-1, j-1, 2]

        dvx[i, j] = -(flux_x_r - flux_x_l) / dx - (flux_y_r - flux_y_l) / dy
    end

    return nothing
end

function fluxes_vy_kernel!(fluxes_vy::CuDeviceArray{T,3}, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, kinematic_viscosity::T, dx::T, dy::T, nx_vy, ny_vy, ep::T, p::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # fast nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    # beginne jeweils bei 2, sodass auch der linke bzw. untere Fluss in die jeweils erste innere Zelle berechnet wird
    if 2 <= i <= nx_vy - 2 && 2 <= j <= ny_vy - 2
        # berechne die horizontale Geschwindigkeitskomponente am rechten Rand der aktuellen Zelle der vertikalen Geschwindigkeitskomponente
        i_, j_ = get_right_bottom_vx_index(i, j)
        vx_right_bottom = vx[i_, j_]

        i_, j_ = get_right_top_vx_index(i, j)
        vx_right_top = vx[i_, j_]

        vx_right = T(0.5) * (vx_right_bottom + vx_right_top)

        vy_l, vy_r = recover_x_vy(i, j, dx, dy, vy, ep, p)
        flux_x = local_lax_friedrichs_x_vy(vy_l, vy_r, vx_right)

        vy_l, vy_r = recover_y_vy(i, j, dx, dy, vy, ep, p)
        flux_y = local_lax_friedrichs_y_vy(vy_l, vy_r)

        fluxes_vy[i-1, j-1, 1] = flux_x - kinematic_viscosity * (vy[i+1, j] - vy[i, j]) / dx

        fluxes_vy[i-1, j-1, 2] = flux_y - kinematic_viscosity * (vy[i, j+1] - vy[i, j]) / dy
    end

    return nothing
end

function fvm_vy_kernel!(dvy::CuDeviceMatrix{T}, fluxes_vy::CuDeviceArray{T,3}, nx_vy, ny_vy, dx::T, dy::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    if 3 <= i <= nx_vy - 2 && 3 <= j <= ny_vy - 2
        flux_x_l = fluxes_vy[i-2, j-1, 1]
        flux_x_r = fluxes_vy[i-1, j-1, 1]

        flux_y_l = fluxes_vy[i-1, j-2, 2]
        flux_y_r = fluxes_vy[i-1, j-1, 2]

        dvy[i, j] = -(flux_x_r - flux_x_l) / dx - (flux_y_r - flux_y_l) / dy
    end

    return nothing
end

function navier_stokes!(du::CuVector{T}, u::CuVector{T}, param, t::T) where T
    fill!(du, T(0.0))

    vx_length = param.nx_vx * param.ny_vx
    vy_length = param.nx_vy * param.ny_vy

    dvx = reshape(view(du, 1:vx_length), param.nx_vx, param.ny_vx)
    dvy = reshape(view(du, vx_length+1:vx_length+vy_length), param.nx_vy, param.ny_vy)

    vx = reshape(view(u, 1:vx_length), param.nx_vx, param.ny_vx)
    vy = reshape(view(u, vx_length+1:vx_length+vy_length), param.nx_vy, param.ny_vy)


    # Berechnung der numersichen Flüsse für die horizontalen Geschwindigkeitskomponenten
    fluxes_vx_kernel = @cuda launch = false fluxes_vx_kernel!(param.fluxes_vx, vx, vy, param.kinematic_viscosity, param.dx, param.dy, param.nx_vx, param.ny_vx, param.ep, param.p)
    config = CUDA.launch_configuration(fluxes_vx_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_vx, threads_per_dim)
    x_blocks = cld(param.nx_vx, x_threads)

    y_threads = min(param.ny_vx, threads_per_dim)
    y_blocks = cld(param.ny_vx, y_threads)

    fluxes_vx_kernel(param.fluxes_vx, vx, vy, param.kinematic_viscosity, param.dx, param.dy, param.nx_vx, param.ny_vx, param.ep, param.p, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))

    # Berechnung der rechten Seite für die horizontalen Geschwindigkeitskomponenten
    fvm_vx_kernel = @cuda launch = false fvm_vx_kernel!(dvx, param.fluxes_vx, param.nx_vx, param.ny_vx, param.dx, param.dy)
    config = CUDA.launch_configuration(fvm_vx_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_vx, threads_per_dim)
    x_blocks = cld(param.nx_vx, x_threads)

    y_threads = min(param.ny_vx, threads_per_dim)
    y_blocks = cld(param.ny_vx, y_threads)

    fvm_vx_kernel(dvx, param.fluxes_vx, param.nx_vx, param.ny_vx, param.dx, param.dy, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))


    # Berechnung der numersichen Flüsse für die vertikalen Geschwindigkeitskomponenten
    fluxes_vy_kernel = @cuda launch = false fluxes_vy_kernel!(param.fluxes_vy, vx, vy, param.kinematic_viscosity, param.dx, param.dy, param.nx_vy, param.ny_vy, param.ep, param.p)
    config = CUDA.launch_configuration(fluxes_vy_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_vy, threads_per_dim)
    x_blocks = cld(param.nx_vy, x_threads)

    y_threads = min(param.ny_vy, threads_per_dim)
    y_blocks = cld(param.ny_vy, y_threads)

    fluxes_vy_kernel(param.fluxes_vy, vx, vy, param.kinematic_viscosity, param.dx, param.dy, param.nx_vy, param.ny_vy, param.ep, param.p, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))

    # Berechnung der rechten Seite für die vertikalen Geschwindigkeitskomponenten
    fvm_vy_kernel = @cuda launch = false fvm_vy_kernel!(dvy, param.fluxes_vy, param.nx_vy, param.ny_vy, param.dx, param.dy)
    config = CUDA.launch_configuration(fvm_vy_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_vy, threads_per_dim)
    x_blocks = cld(param.nx_vy, x_threads)

    y_threads = min(param.ny_vy, threads_per_dim)
    y_blocks = cld(param.ny_vy, y_threads)

    fvm_vy_kernel(dvy, param.fluxes_vy, param.nx_vy, param.ny_vy, param.dx, param.dy, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

function get_u0(param::Parameters{T}) where T
    vx0s = zeros(T, param.nx_vx, param.ny_vx)
    vy0s = zeros(T, param.nx_vy, param.ny_vy)

    # nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    for j in 3:param.ny_vx-2, i in 3:param.nx_vx-2
        x, y = get_vx_position(i, j, param.dx, param.dy)
        vx0s[i, j] = vx0(x, y)
    end

    # nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    for j in 3:param.ny_vy-2, i in 3:param.nx_vy-2
        x, y = get_vy_position(i, j, param.dx, param.dy)
        vy0s[i, j] = vy0(x, y)
    end

    vx0s = CuMatrix(vx0s)
    vy0s = CuMatrix(vy0s)

    apply_boundary_conditions_vx!(vx0s, param)
    apply_boundary_conditions_vy!(vy0s, param)

    return vcat(vec(vx0s), vec(vy0s))
end

function interpolate_to_centers(vx::Matrix{T}, vy::Matrix{T}, param) where T
    vx_center = zeros(T, param.nx_pressure, param.ny_pressure)
    vy_center = zeros(T, param.nx_pressure, param.ny_pressure)

    for j in 1:param.ny_pressure, i in 1:param.nx_pressure
        i_, j_ = get_left_vx_index(i, j)
        vx_left = vx[i_, j_]
        i_, j_ = get_right_vx_index(i, j)
        vx_right = vx[i_, j_]

        vx_center[i, j] = T(0.5) * (vx_left + vx_right)

        i_, j_ = get_bottom_vy_index(i, j)
        vy_bottom = vy[i_, j_]
        i_, j_ = get_top_vy_index(i, j)
        vy_top = vy[i_, j_]

        vy_center[i, j] = T(0.5) * (vy_bottom + vy_top)
    end

    return vx_center, vy_center
end

function get_solution(saved_values, t, param)
    ts = saved_values.t
    solutions = saved_values.saveval

    # finde die Lösung, die dem übergebenen Zeitpunkt am nächsten liegt
    i = argmin(
        abs.(t .- ts)
    )

    vx = solutions[i][1]
    vy = solutions[i][2]
    pressure = solutions[i][3]

    vx_center, vy_center = interpolate_to_centers(vx, vy, param)

    return vx_center, vy_center, pressure
end

type = Float32

param = Parameters(
    Lx=type(5.0),
    Ly=type(1.0),

    # Luft
    dynamic_viscosity=type(1.5e-5),
    density=type(1.204), 
    
    horizontal_velocity=type(2.5), 
    
    nx=2500,
    ny=500, 
    
    ep=type(1.0e-6),
    p=type(0.6)
)

# Anfangsbedingungen
vx0(x::T, y::T) where T = T(0.0)
vy0(x::T, y::T) where T = T(0.0)

u0 = get_u0(param)

# Plotting-Parameter
max_plot_size = (1920, 1080) # maximale Auflösung der Plots
fps = 60 # Bildwiederholrate der Animationen
factor_plot_size = min(max_plot_size[1] / param.nx_pressure, max_plot_size[2] / param.ny_pressure)
plot_size = (factor_plot_size * param.nx_pressure, factor_plot_size * param.ny_pressure)

# Zeitintervall
tspan = (type(0.0), type(5.0))
n_frames = ceil(Int, tspan[2] - tspan[1]) * fps
saveat = tspan[1]:type(1 / fps):tspan[2]

# Lösen des semidiskreten ODE-Systems (Linienmethode)
solver = Tsit5((stage_limiter!)=apply_boundary_conditions_limiter!)
tol = type(1e-5)

prob = ODEProblem(navier_stokes!, u0, tspan, param)

condition(u, t, integrator) = true
update_velocities_cb = DiscreteCallback(condition, update_velocities_cb!; save_positions=(false, false))

saved_values = SavedValues(type, NTuple{3,Matrix{type}})
saving_cb = SavingCallback(save_func, saved_values; saveat=saveat, save_everystep=false, save_start=false, save_end=false)

cb = CallbackSet(update_velocities_cb, saving_cb)
sol = CUDA.@time solve(prob, solver; reltol=tol, abstol=tol, callback=cb, save_everystep=false, save_start=false, save_end=false)

display(sol.stats)

# Plotting
t = Observable(tspan[2])
vx = @lift get_solution(saved_values, $t, param)[1]
vy = @lift get_solution(saved_values, $t, param)[2]
pressure = @lift get_solution(saved_values, $t, param)[3]

# Plotte das Tempo (Betrag der Geschwindigkeit)
velocity_norm = @lift sqrt.($vx .^ 2 .+ $vy .^ 2)

fig = Figure(size=plot_size)
ax = GLMakie.Axis(fig[1, 1], title="Betrag der Geschwindigkeit")
heatmap!(velocity_norm)
save("plots/fluid_norm_2d.png", fig)

t[] = type(2.5)

GLMakie.record(fig, "plots/fluid_norm_2d.mp4", range(tspan[1], tspan[2]; length=n_frames); framerate=fps) do time
    t[] = time
end