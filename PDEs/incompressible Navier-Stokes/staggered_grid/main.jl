using CUDA
using CUDA.CUSPARSE
using CUDSS
using Images
using OrdinaryDiffEq
using DiffEqCallbacks
using Interpolations
using GLMakie

include("pressure.jl")
include("velocities.jl")
include("saving.jl")

#=  Staggered Grid inklusive Rand- bzw. Ghost-Zellen

+ - - - + - - - + - - - + - - - + - - - + - - - + - - - + - - - + - - - + - - - +
|       |       |       |       |       |       |       |       |       |       |
|   c   →   c   →   c   →   c   →   c   →   c   →   c   →   c   →   c   →   c   |
|       |       |       |       |       |       |       |       |       |       |
+ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +
|       |       |       |       |       |       |       |       |       |       |
|   c   →   c   →   c   →   c   →   c   →   c   →   c   →   c   →   c   →   c   |
|       |       |       |       |       |       |       |       |       |       |
+ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +
|       |       |       |       |       |       |       |       |       |       |
|   c   →   c   →   •   →   •   →   •   →   •   →   •   →   •   →   c   →   c   |
|       |       |       |       |       |       |       |       |       |       |
+ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +
|       |       |       |       |       |       |       |       |       |       |
|   c   →   c   →   •   →   •   →   •   →   •   →   •   →   •   →   c   →   c   |
|       |       |       |       |       |       |       |       |       |       |
+ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +
|       |       |       |       |       |       |       |       |       |       |
|   c   →   c   →   •   →   •   →   •   →   •   →   •   →   •   →   c   →   c   |
|       |       |       |       |       |       |       |       |       |       |
+ - ↑ - + - ↑ - 0 - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +
|       |       |       |       |       |       |       |       |       |       |
|   c   →   c   →   c   →   c   →   c   →   c   →   c   →   c   →   c   →   c   |
|       |       |       |       |       |       |       |       |       |       |
+ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - +
|       |       |       |       |       |       |       |       |       |       |
|   c   →   c   →   c   →   c   →   c   →   c   →   c   →   c   →   c   →   c   |
|       |       |       |       |       |       |       |       |       |       |
+ - - - + - - - + - - - + - - - + - - - + - - - + - - - + - - - + - - - + - - - +


nx = nx_pressure = 6
ny = ny_pressure = 3

nx_vx = nx + 3 = 9
ny_vx = ny + 4 = 7

nx_vy = nx + 4 = 10
ny_vy = ny + 3 = 6

nx_c = nx + 4
ny_c = ny + 4

"•" sind die Mittelpunkte der Druckzellen und der c-Zellen im eigentlichen Berechnungsgebiet (nicht versetzt)
"c" sind die Mittelpunkte der c-Ghost-Zellen außerhalb des eigentlichen Berechnungsgebiets (nicht versetzt)
"→" sind die Mittelpunkte der Zellen der horizontalen Geschwindigkeitskomponenten (versetzt)
"↑" sind die Mittelpunkte der Zellen der vertikalen Geschwindigkeitskomponenten (versetzt)

Für jeden Rand gibt es pro Geschwindigkeitskomponente zwei Rand- bzw. Ghost-Zellen, 
wobei in der zur jeweiligen Komponente gehörigen Dimension nur eine echte Ghost-Zelle nötig ist,
weil die Zelle davor exakt auf dem Rand liegt. Ansonsten sind es zwei echte Ghost-Zellen.

Für das c-Feld pro gibt es pro Rand zwei Lagen Ghost-Zellen.
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


# Gibt zu dem Index einer Druckvariable den Index des an der gleichen Stelle liegenden c-Werts zurück
get_c_index(i, j) = (i + 2, j + 2)
# Gibt zu dem Index eines c-Werts den Index der an der gleichen Stelle liegenden Druckvariable zurück
get_pressure_index(i, j) = (i - 2, j - 2)


# Gibt zu dem Index einer horizontalen Geschwindigkeitskomponente den Index des links daneben liegenden c-Werts zurück
get_left_c_index(i, j) = (i, j)
# Gibt zu dem Index einer horizontalen Geschwindigkeitskomponente den Index des rechts daneben liegenden c-Werts zurück
get_right_c_index(i, j) = (i + 1, j)
# Gibt zu dem Index einer horizontalen Geschwindigkeitskomponente den Index des oberhalb liegenden, nach links versetzten c-Werts zurück
get_top_left_c_index(i, j) = (i, j + 1)
# Gibt zu dem Index einer horizontalen Geschwindigkeitskomponente den Index des oberhalb liegenden, nach rechts versetzten c-Werts zurück
get_top_right_c_index(i, j) = (i + 1, j + 1)


# Gibt zu dem Index einer vertikalen Geschwindigkeitskomponente den Index des unterhalb liegenden c-Werts zurück
get_bottom_c_index(i, j) = (i, j)
# Gibt zu dem Index einer vertikalen Geschwindigkeitskomponente den Index des oberhalb liegenden c-Werts zurück#
get_top_c_index(i, j) = (i, j + 1)
# Gibt zu dem Index einer vertikalen Geschwindigkeitskomponente den Index des rechts daneben liegenden, nach unten versetzten c-Werts zurück
get_right_bottom_c_index(i, j) = (i + 1, j)
# Gibt zu dem Index einer vertikalen Geschwindigkeitskomponente den Index des rechts daneben liegenden, nach oben versetzten c-Werts zurück
get_right_top_c_index(i, j) = (i + 1, j + 1)


# Gibt die Position des Zellmittelpunkts einer Druckvariable zurück
get_pressure_position(i, j, dx::T, dy::T) where T = ((i - T(0.5)) * dx, (j - T(0.5)) * dy)
# Gibt die Position des Zellmittelpunkts eines c-Werts zurück
get_c_position(i, j, dx::T, dy::T) where T = ((i - T(2.5)) * dx, (j - T(2.5)) * dy)
# Gibt die Position des Zellmittelpunkts einer horizontalen Geschwindigkeitskomponente zurück
get_vx_position(i, j, dx::T, dy::T) where T = ((i - 2) * dx, (j - T(2.5)) * dy)
# Gibt die Position des Zellmittelpunkts einer vertikalen Geschwindigkeitskomponente zurück
get_vy_position(i, j, dx::T, dy::T) where T = ((i - T(2.5)) * dx, (j - 2) * dy)

struct Parameters{T<:AbstractFloat}
    Lx::T # Seitenlänge in x-Richtung der rechteckigen Oberfläche [m]
    Ly::T # Seitenlänge in y-Richtung der rechteckigen Oberfläche [m]

    # Fluid 1 (c = 1)
    dynamic_viscosity_1::T # [Pa*s]
    density_1::T # [kg/m^3]

    # Fluid 2 (c = 0)
    dynamic_viscosity_2::T # [Pa*s]
    density_2::T # [kg/m^3]

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
    fluxes_vx::CuArray{T,4}
    vx_corrected::CuMatrix{T}

    nx_vy::Int
    ny_vy::Int
    fluxes_vy::CuArray{T,4}
    vy_corrected::CuMatrix{T}

    nx_c::Int
    ny_c::Int
    fluxes_c::CuArray{T,3}
    c_advected::CuMatrix{T}

    # WENO-Parameter
    ep::T
    p::T

    omega::T # Stärke der Interface-Compression

    cfl::T # CFL-Zahl
    potential_time_steps::CuMatrix{T}

    function Parameters(; Lx::T, Ly::T, dynamic_viscosity_1::T, density_1::T, dynamic_viscosity_2::T, density_2::T, nx, ny, ep::T, p::T, omega::T, cfl::T) where T<:AbstractFloat
        dx = Lx / nx
        dy = Ly / ny

        coefficient_mat = get_coefficient_mat(nx, ny, dx, dy)
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
        fluxes_vx = CUDA.zeros(T, nx_vx - 3, ny_vx - 3, 2, 2)
        vx_corrected = CUDA.zeros(T, nx_vx, ny_vx)

        nx_vy = nx + 4
        ny_vy = ny + 3
        fluxes_vy = CUDA.zeros(T, nx_vy - 3, ny_vy - 3, 2, 2)
        vy_corrected = CUDA.zeros(T, nx_vy, ny_vy)

        nx_c = nx + 4
        ny_c = ny + 4
        fluxes_c = CUDA.zeros(T, nx_c - 3, ny_c - 3, 2)
        c_advected = CUDA.zeros(T, nx_c, ny_c)

        potential_time_steps = CUDA.zeros(T, nx, ny)

        new{T}(
            Lx, Ly, 
            dynamic_viscosity_1, density_1,
            dynamic_viscosity_2, density_2,
            dx, dy,
            nx, ny, lin_solve, coefficient_mat, pressure, divergence,
            nx_vx, ny_vx, fluxes_vx, vx_corrected,
            nx_vy, ny_vy, fluxes_vy, vy_corrected,
            nx_c, ny_c, fluxes_c, c_advected,
            ep, p,
            omega,
            cfl, potential_time_steps
        )
    end
end

function fluxes_vx_kernel!(fluxes_vx::CuDeviceArray{T,4}, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, c_advected::CuDeviceMatrix{T}, dynamic_viscosity_1::T, dynamic_viscosity_2::T, dx::T, dy::T, nx_vx, ny_vx, ep::T, p::T) where T
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

        fluxes_vx[i-1, j-1, 1, 1] = flux_x
        fluxes_vx[i-1, j-1, 2, 1] = flux_y

        # berechne die dynamische Viskosität am rechten und oberen Rand der aktuellen Zelle der horizontalen Geschwindigkeitskomponente
        i_, j_ = get_left_c_index(i, j)
        c_left = c_advected[i_, j_]
        i_, j_ = get_right_c_index(i, j)
        c_right = c_advected[i_, j_]

        i_, j_ = get_top_left_c_index(i, j)
        c_top_left = c_advected[i_, j_]
        i_, j_ = get_top_right_c_index(i, j)
        c_top_right = c_advected[i_, j_]
        
        c_top = T(0.25) * (c_left + c_right + c_top_left + c_top_right)

        dynamic_viscosity_right = c_right * dynamic_viscosity_1 + (T(1.0) - c_right) * dynamic_viscosity_2
        dynamic_viscosity_top = c_top * dynamic_viscosity_1 + (T(1.0) - c_top) * dynamic_viscosity_2

        fluxes_vx[i-1, j-1, 1, 2] = dynamic_viscosity_right * (vx[i+1, j] - vx[i, j]) / dx
        fluxes_vx[i-1, j-1, 2, 2] = dynamic_viscosity_top * (vx[i, j+1] - vx[i, j]) / dy
    end

    return nothing
end

function fvm_vx_kernel!(dvx::CuDeviceMatrix{T}, fluxes_vx::CuDeviceArray{T,4}, c_advected::CuDeviceMatrix{T}, density_1::T, density_2::T, dx::T, dy::T, nx_vx, ny_vx) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    if 3 <= i <= nx_vx - 2 && 3 <= j <= ny_vx - 2
        flux_x_l = fluxes_vx[i-2, j-1, 1, 1]
        flux_x_r = fluxes_vx[i-1, j-1, 1, 1]

        flux_y_l = fluxes_vx[i-1, j-2, 2, 1]
        flux_y_r = fluxes_vx[i-1, j-1, 2, 1]

        dvx[i, j] = -(flux_x_r - flux_x_l) / dx - (flux_y_r - flux_y_l) / dy

        viscous_flux_x_l = fluxes_vx[i-2, j-1, 1, 2]
        viscous_flux_x_r = fluxes_vx[i-1, j-1, 1, 2]

        viscous_flux_y_l = fluxes_vx[i-1, j-2, 2, 2]
        viscous_flux_y_r = fluxes_vx[i-1, j-1, 2, 2]

        i_, j_ = get_left_c_index(i, j)
        c_left = c_advected[i_, j_]
        i_, j_ = get_right_c_index(i, j)
        c_right = c_advected[i_, j_]
        
        c = T(0.5) * (c_left + c_right)
        density = c * density_1 + (T(1.0) - c) * density_2

        dvx[i, j] += ((viscous_flux_x_r - viscous_flux_x_l) / dx + (viscous_flux_y_r - viscous_flux_y_l) / dy) / density
    end

    return nothing
end

function fluxes_vy_kernel!(fluxes_vy::CuDeviceArray{T,4}, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, c_advected::CuDeviceMatrix{T}, dynamic_viscosity_1::T, dynamic_viscosity_2::T, dx::T, dy::T, nx_vy, ny_vy, ep::T, p::T) where T
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

        fluxes_vy[i-1, j-1, 1, 1] = flux_x
        fluxes_vy[i-1, j-1, 2, 1] = flux_y

        # berechne die dynamische Viskosität am rechten und oberen Rand der aktuellen Zelle der vertikalen Geschwindigkeitskomponente
        i_, j_ = get_bottom_c_index(i, j)
        c_bottom = c_advected[i_, j_]
        i_, j_ = get_top_c_index(i, j)
        c_top = c_advected[i_, j_]

        i_, j_ = get_right_bottom_c_index(i, j)
        c_right_bottom = c_advected[i_, j_]
        i_, j_ = get_right_top_c_index(i, j)
        c_right_top = c_advected[i_, j_]
        
        c_right = T(0.25) * (c_bottom + c_top + c_right_bottom + c_right_top)

        dynamic_viscosity_right = c_right * dynamic_viscosity_1 + (T(1.0) - c_right) * dynamic_viscosity_2
        dynamic_viscosity_top = c_top * dynamic_viscosity_1 + (T(1.0) - c_top) * dynamic_viscosity_2

        fluxes_vy[i-1, j-1, 1, 2] = dynamic_viscosity_right * (vy[i+1, j] - vy[i, j]) / dx
        fluxes_vy[i-1, j-1, 2, 2] = dynamic_viscosity_top * (vy[i, j+1] - vy[i, j]) / dy
    end

    return nothing
end

function fvm_vy_kernel!(dvy::CuDeviceMatrix{T}, fluxes_vy::CuDeviceArray{T,4}, c_advected::CuDeviceMatrix{T}, density_1::T, density_2::T, dx::T, dy::T, nx_vy, ny_vy) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    if 3 <= i <= nx_vy - 2 && 3 <= j <= ny_vy - 2
        flux_x_l = fluxes_vy[i-2, j-1, 1, 1]
        flux_x_r = fluxes_vy[i-1, j-1, 1, 1]

        flux_y_l = fluxes_vy[i-1, j-2, 2, 1]
        flux_y_r = fluxes_vy[i-1, j-1, 2, 1]

        dvy[i, j] = -(flux_x_r - flux_x_l) / dx - (flux_y_r - flux_y_l) / dy - T(9.81)

        viscous_flux_x_l = fluxes_vy[i-2, j-1, 1, 2]
        viscous_flux_x_r = fluxes_vy[i-1, j-1, 1, 2]

        viscous_flux_y_l = fluxes_vy[i-1, j-2, 2, 2]
        viscous_flux_y_r = fluxes_vy[i-1, j-1, 2, 2]

        i_, j_ = get_bottom_c_index(i, j)
        c_bottom = c_advected[i_, j_]
        i_, j_ = get_top_c_index(i, j)
        c_top = c_advected[i_, j_]
        
        c = T(0.5) * (c_bottom + c_top)
        density = c * density_1 + (T(1.0) - c) * density_2

        dvy[i, j] += ((viscous_flux_x_r - viscous_flux_x_l) / dx + (viscous_flux_y_r - viscous_flux_y_l) / dy) / density
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
    fluxes_vx_kernel = @cuda launch = false fluxes_vx_kernel!(param.fluxes_vx, vx, vy, param.c_advected, param.dynamic_viscosity_1, param.dynamic_viscosity_2, param.dx, param.dy, param.nx_vx, param.ny_vx, param.ep, param.p)
    config = CUDA.launch_configuration(fluxes_vx_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_vx, threads_per_dim)
    x_blocks = cld(param.nx_vx, x_threads)

    y_threads = min(param.ny_vx, threads_per_dim)
    y_blocks = cld(param.ny_vx, y_threads)

    fluxes_vx_kernel(param.fluxes_vx, vx, vy, param.c_advected, param.dynamic_viscosity_1, param.dynamic_viscosity_2, param.dx, param.dy, param.nx_vx, param.ny_vx, param.ep, param.p, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))

    # Berechnung der rechten Seite für die horizontalen Geschwindigkeitskomponenten
    fvm_vx_kernel = @cuda launch = false fvm_vx_kernel!(dvx, param.fluxes_vx, param.c_advected, param.density_1, param.density_2, param.dx, param.dy, param.nx_vx, param.ny_vx)
    config = CUDA.launch_configuration(fvm_vx_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_vx, threads_per_dim)
    x_blocks = cld(param.nx_vx, x_threads)

    y_threads = min(param.ny_vx, threads_per_dim)
    y_blocks = cld(param.ny_vx, y_threads)

    fvm_vx_kernel(dvx, param.fluxes_vx, param.c_advected, param.density_1, param.density_2, param.dx, param.dy, param.nx_vx, param.ny_vx, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))


    # Berechnung der numersichen Flüsse für die vertikalen Geschwindigkeitskomponenten
    fluxes_vy_kernel = @cuda launch = false fluxes_vy_kernel!(param.fluxes_vy, vx, vy, param.c_advected, param.dynamic_viscosity_1, param.dynamic_viscosity_2, param.dx, param.dy, param.nx_vy, param.ny_vy, param.ep, param.p)
    config = CUDA.launch_configuration(fluxes_vy_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_vy, threads_per_dim)
    x_blocks = cld(param.nx_vy, x_threads)

    y_threads = min(param.ny_vy, threads_per_dim)
    y_blocks = cld(param.ny_vy, y_threads)

    fluxes_vy_kernel(param.fluxes_vy, vx, vy, param.c_advected, param.dynamic_viscosity_1, param.dynamic_viscosity_2, param.dx, param.dy, param.nx_vy, param.ny_vy, param.ep, param.p, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))

    # Berechnung der rechten Seite für die vertikalen Geschwindigkeitskomponenten
    fvm_vy_kernel = @cuda launch = false fvm_vy_kernel!(dvy, param.fluxes_vy, param.c_advected, param.density_1, param.density_2, param.dx, param.dy, param.nx_vy, param.ny_vy)
    config = CUDA.launch_configuration(fvm_vy_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_vy, threads_per_dim)
    x_blocks = cld(param.nx_vy, x_threads)

    y_threads = min(param.ny_vy, threads_per_dim)
    y_blocks = cld(param.ny_vy, y_threads)

    fvm_vy_kernel(dvy, param.fluxes_vy, param.c_advected, param.density_1, param.density_2, param.dx, param.dy, param.nx_vy, param.ny_vy, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

#=
function fluxes_c_kernel!(fluxes_c::CuDeviceArray{T,3}, c::CuDeviceMatrix{T}, vx_corrected::CuDeviceMatrix, vy_corrected::CuDeviceMatrix{T}, nx_c, ny_c) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # fast nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    # beginne jeweils bei 2, sodass auch der linke bzw. untere Fluss in die jeweils erste innere Zelle berechnet wird
    if 2 <= i <= nx_c - 2 && 2 <= j <= ny_c - 2
        # berechne die horizontale Geschwindigkeitskomponente am rechten Rand der aktuellen Zelle
        i_, j_ = get_right_vx_index(i-2, j-2)
        vx_right = vx_corrected[i_, j_]

        # berechne die vertikale Geschwindigkeitskomponente am oberen Rand der aktuellen Zelle
        i_, j_ = get_top_vy_index(i-2, j-2)
        vy_top = vy_corrected[i_, j_]

        c_l, c_r = recover_x_c(i, j, c)
        flux_x = upwind_x_c(c_l, c_r, vx_right)

        c_l, c_r = recover_y_c(i, j, c)
        flux_y = upwind_y_c(c_l, c_r, vy_top)

        fluxes_c[i-1, j-1, 1] = flux_x

        fluxes_c[i-1, j-1, 2] = flux_y
    end

    return nothing
end
=#

function fluxes_c_kernel!(fluxes_c::CuDeviceArray{T,3}, c::CuDeviceMatrix{T}, vx_corrected::CuDeviceMatrix, vy_corrected::CuDeviceMatrix{T}, dx::T, dy::T, nx_c, ny_c, omega::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    eps_norm = eps(T) * T(100.0) # vermeide Division durch 0 beim Normieren
    c_cutoff = T(0.05) # Schwelle für den c-Wert zur Interface-Detection

    # fast nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    # beginne jeweils bei 2, sodass auch der linke bzw. untere Fluss in die jeweils erste innere Zelle berechnet wird
    if 2 <= i <= nx_c - 2 && 2 <= j <= ny_c - 2
        # Berechnung der Geschwindigkeitskomponenten am rechten und oberen Rand der aktuellen Zelle

        # berechne die horizontale Geschwindigkeitskomponente am rechten Rand der aktuellen Zelle
        i_, j_ = get_pressure_index(i, j)
        i_, j_ = get_right_vx_index(i_, j_)
        vx_right = vx_corrected[i_, j_]

        # berechne die vertikale Geschwindigkeitskomponente am oberen Rand der aktuellen Zelle
        i_, j_ = get_pressure_index(i, j)
        i_, j_ = get_top_vy_index(i_, j_)
        vy_top = vy_corrected[i_, j_]

        # berechne die vertikale Geschwindigkeitskomponente am rechten Rand der aktuellen Zelle
        i_, j_ = get_pressure_index(i, j)
        i_, j_ = get_top_vy_index(i_, j_)
        vy_top_right = vy_corrected[i_ + 1, j_]

        i_, j_ = get_pressure_index(i, j)
        i_, j_ = get_bottom_vy_index(i_, j_)
        vy_bottom = vy_corrected[i_, j_]

        i_, j_ = get_pressure_index(i, j)
        i_, j_ = get_bottom_vy_index(i_, j_)
        vy_bottom_right = vy_corrected[i_ + 1, j_]

        vy_right = T(0.25) * (vy_top + vy_top_right + vy_bottom + vy_bottom_right)

        # berechne die horitontale Geschwindigkeitskomponente am oberen Rand der aktuellen Zelle
        i_, j_ = get_pressure_index(i, j)
        i_, j_ = get_right_vx_index(i_, j_)
        vx_right_top = vx_corrected[i_, j_ + 1]

        i_, j_ = get_pressure_index(i, j)
        i_, j_ = get_left_vx_index(i_, j_)
        vx_left = vx_corrected[i_, j_]

        i_, j_ = get_pressure_index(i, j)
        i_, j_ = get_left_vx_index(i_, j_)
        vx_left_top = vx_corrected[i_, j_ + 1]

        vx_top = T(0.25) * (vx_right + vx_right_top + vx_left + vx_left_top)
        

        # Berechnung der numerischen Flüsse in x-Richtung
        c_l, c_r = recover_x_c(i, j, c)
        flux_x = upwind_x_c(c_l, c_r, vx_right)

        
        # Berechnung der Interface-Compression in x-Richtung

        # berechne die Gradienten links und rechts der Zellkante
        grad_x_here = (c[i+1, j] - c[i-1, j]) / (T(2.0) * dx)
        grad_y_here = (c[i, j+1] - c[i, j-1]) / (T(2.0) * dy)

        grad_x_neighbor = (c[i+2, j] - c[i, j]) / (T(2.0) * dx)
        grad_y_neighbor = (c[i+1, j+1] - c[i+1, j-1]) / (T(2.0) * dy)

        # normiere die Gradienten
        grad_norm_here = sqrt(grad_x_here^2 + grad_y_here^2) + eps_norm 
        grad_norm_neighbor = sqrt(grad_x_neighbor^2 + grad_y_neighbor^2) + eps_norm 

        # berechne die Einheitsnormalenvektoren des Interfaces links und rechts der Zellkante
        normal_x_here = grad_x_here / grad_norm_here
        normal_y_here = grad_y_here / grad_norm_here

        normal_x_neighbor = grad_x_neighbor / grad_norm_neighbor
        normal_y_neighbor = grad_y_neighbor / grad_norm_neighbor

        # berechne den Einheitsnormalenvektor des Interfaces auf der Zellkante
        normal_x_face = T(0.5) * (normal_x_here + normal_x_neighbor)
        normal_y_face = T(0.5) * (normal_y_here + normal_y_neighbor)
        # stelle sicher, dass auch der gemittelte Normalenvektor normiert bleibt
        normal_norm_face = sqrt(normal_x_face^2 + normal_y_face^2) + eps_norm
        normal_x_face /= normal_norm_face
        normal_y_face /= normal_norm_face

        # berechne den Betrag der Geschwindigkeit am rechten Rand der aktuellen Zelle
        velocity_norm = sqrt(vx_right^2 + vy_right^2)
        # berechne den horitontalen Anteil der Kompressionsgeschwindigkeit in Richtung der dichteren Phase (c = 1)
        compression_velocity_x = omega * velocity_norm * normal_x_face
        # berechne den c-Wert auf dem rechten Rand der aktuellen Zelle aus den rekonstruierten Werten auf der Zellkante
        c_face = T(0.5) * (c_l + c_r)
        # berechne den Kompressionsfluss in x-Richtung
        compression_flux_x = compression_velocity_x * (c_face * (T(1.0) - c_face))

        # überprüfe, ob wirklich ein Interface vorliegt
        is_interface_x = (c_cutoff < c[i, j] < T(1.0) - c_cutoff) || (c_cutoff < c[i+1, j] < T(1.0) - c_cutoff)
        compression_flux_x = is_interface_x ? compression_flux_x : T(0.0)


        # Berechnung der numerischen Flüsse in y-Richtung
        c_l, c_r = recover_y_c(i, j, c)
        flux_y = upwind_y_c(c_l, c_r, vy_top)


        # Berechnung der Interface-Compression in y-Richtung

        # berechne den Gradienten oberhalb der Zellkante, unterhalb wurde er bereits berechnet
        grad_x_neighbor = (c[i+1, j+1] - c[i-1, j+1]) / (T(2.0) * dx)
        grad_y_neighbor = (c[i, j+2] - c[i, j]) / (T(2.0) * dy)

        # normiere den Gradienten
        grad_norm_neighbor = sqrt(grad_x_neighbor^2 + grad_y_neighbor^2) + eps_norm 

        # berechne den Einheitsnormalenvektor des Interfaces oberhalb der Zellkante, unterhalb wurde er bereits berechnet
        normal_x_neighbor = grad_x_neighbor / grad_norm_neighbor
        normal_y_neighbor = grad_y_neighbor / grad_norm_neighbor

        # berechne den Einheitsnormalenvektor des Interfaces auf der Zellkante
        normal_x_face = T(0.5) * (normal_x_here + normal_x_neighbor)
        normal_y_face = T(0.5) * (normal_y_here + normal_y_neighbor)
        # stelle sicher, dass auch der gemittelte Normalenvektor normiert bleibt
        normal_norm_face = sqrt(normal_x_face^2 + normal_y_face^2) + eps_norm
        normal_x_face /= normal_norm_face
        normal_y_face /= normal_norm_face

        # berechne den Betrag der Geschwindigkeit am oberen Rand der aktuellen Zelle
        velocity_norm = sqrt(vx_top^2 + vy_top^2)
        # berechne den vertikalen Anteil der Kompressionsgeschwindigkeit in Richtung der dichteren Phase (c = 1)
        compression_velocity_y = omega * velocity_norm * normal_y_face
        # berechne den c-Wert auf dem oberen Rand der aktuellen Zelle aus den rekonstruierten Werten auf der Zellkante
        c_face = T(0.5) * (c_l + c_r)
        # berechne den Kompressionsfluss in y-Richtung
        compression_flux_y = compression_velocity_y * (c_face * (T(1.0) - c_face))

        # überprüfe, ob wirklich ein Interface vorliegt
        is_interface_y = (c_cutoff < c[i, j] < T(1.0) - c_cutoff) || (c_cutoff < c[i, j+1] < T(1.0) - c_cutoff)
        compression_flux_y = is_interface_y ? compression_flux_y : T(0.0)


        fluxes_c[i-1, j-1, 1] = flux_x + compression_flux_x
        fluxes_c[i-1, j-1, 2] = flux_y + compression_flux_y
    end

    return nothing
end

function fvm_c_kernel!(dc::CuDeviceMatrix{T}, fluxes_c::CuDeviceArray{T,3}, nx_c, ny_c, dx::T, dy::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    if 3 <= i <= nx_c - 2 && 3 <= j <= ny_c - 2
        flux_x_l = fluxes_c[i-2, j-1, 1]
        flux_x_r = fluxes_c[i-1, j-1, 1]

        flux_y_l = fluxes_c[i-1, j-2, 2]
        flux_y_r = fluxes_c[i-1, j-1, 2]

        dc[i, j] = -(flux_x_r - flux_x_l) / dx - (flux_y_r - flux_y_l) / dy
    end

    return nothing
end

function advection!(du::CuVector{T}, u::CuVector{T}, param, t::T) where T
    fill!(du, T(0.0))

    dc = reshape(du, param.nx_c, param.ny_c)
    c = reshape(u, param.nx_c, param.ny_c)

    # Berechnung der numersichen Flüsse
    fluxes_kernel = @cuda launch = false fluxes_c_kernel!(param.fluxes_c, c, param.vx_corrected, param.vy_corrected, param.dx, param.dy, param.nx_c, param.ny_c, param.omega)
    config = CUDA.launch_configuration(fluxes_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_c, threads_per_dim)
    x_blocks = cld(param.nx_c, x_threads)

    y_threads = min(param.ny_c, threads_per_dim)
    y_blocks = cld(param.ny_c, y_threads)

    fluxes_kernel(param.fluxes_c, c, param.vx_corrected, param.vy_corrected, param.dx, param.dy, param.nx_c, param.ny_c, param.omega, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))

    # Berechnung der rechten Seite
    fvm_kernel = @cuda launch = false fvm_c_kernel!(dc, param.fluxes_c, param.nx_c, param.ny_c, param.dx, param.dy)
    config = CUDA.launch_configuration(fvm_kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_c, threads_per_dim)
    x_blocks = cld(param.nx_c, x_threads)

    y_threads = min(param.ny_c, threads_per_dim)
    y_blocks = cld(param.ny_c, y_threads)

    fvm_kernel(dc, param.fluxes_c, param.nx_c, param.ny_c, param.dx, param.dy, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

function get_navier_stokes_u0(param::Parameters{T}) where T
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

function get_advection_u0(param::Parameters{T}) where T
    c0s = zeros(T, param.nx_c, param.ny_c)

    # nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    for j in 3:param.ny_c-2, i in 3:param.nx_c-2
        x, y = get_pressure_position(i, j, param.dx, param.dy)
        c0s[i, j] = c0(x, y)
    end

    c0s = CuMatrix(c0s)

    apply_boundary_conditions_c!(c0s, param)

    return vec(c0s)
end

function calculate_potential_time_steps_kernel!(potential_time_steps::CuDeviceMatrix{T}, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, dx::T, dy::T, nx_pressure, ny_pressure, cfl::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i <= nx_pressure && j <= ny_pressure
        i_, j_ = get_left_vx_index(i, j)
        vx_left = vx[i_, j_]
        i_, j_ = get_right_vx_index(i, j)
        vx_right = vx[i_, j_]

        i_, j_ = get_bottom_vy_index(i, j)
        vy_bottom = vy[i_, j_]
        i_, j_ = get_top_vy_index(i, j)
        vy_top = vy[i_, j_]

        max_vx = max(abs(vx_left), abs(vx_right))
        max_vy = max(abs(vy_bottom), abs(vy_top))

        eps = T(1e-12)
        potential_time_steps[i, j] = cfl / ((max_vx / dx) + (max_vy / dy) + eps)
    end

    return nothing
end

function calculate_potential_time_steps!(vx::CuMatrix{T}, vy::CuMatrix{T}, param) where T
    kernel = @cuda launch = false calculate_potential_time_steps_kernel!(param.potential_time_steps, vx, vy, param.dx, param.dy, param.nx_pressure, param.ny_pressure, param.cfl)
    config = CUDA.launch_configuration(kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_pressure, threads_per_dim)
    x_blocks = cld(param.nx_pressure, x_threads)

    y_threads = min(param.ny_pressure, threads_per_dim)
    y_blocks = cld(param.ny_pressure, y_threads)

    kernel(param.potential_time_steps, vx, vy, param.dx, param.dy, param.nx_pressure, param.ny_pressure, param.cfl, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

function calculate_time_step(navier_stokes_integrator)
    u = navier_stokes_integrator.u
    param = navier_stokes_integrator.p

    vx_length = param.nx_vx * param.ny_vx
    vy_length = param.nx_vy * param.ny_vy

    vx = reshape(view(u, 1:vx_length), param.nx_vx, param.ny_vx)
    vy = reshape(view(u, vx_length+1:vx_length+vy_length), param.nx_vy, param.ny_vy)

    calculate_potential_time_steps!(vx, vy, param)
    time_step = minimum(param.potential_time_steps)

    return time_step
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

function get_solution(ts, solutions, t, param)
    # finde die Lösung, die dem übergebenen Zeitpunkt am nächsten liegt
    i = argmin(
        abs.(t .- ts)
    )

    vx = solutions[i][1]
    vy = solutions[i][2]
    pressure = solutions[i][3]

    vx_center, vy_center = interpolate_to_centers(vx, vy, param)

    c = solutions[i][4]

    return vx_center, vy_center, pressure, c
end

type = Float32

param = Parameters(
    Lx=type(6.25),
    Ly=type(1.5),
    
    # Wasser (c = 1)
    dynamic_viscosity_1=type(1.0e-3),
    density_1=type(998.0),

    # Luft (c = 0)
    dynamic_viscosity_2=type(1.5e-5),
    density_2=type(1.204),
    
    nx=Int(1250),
    ny=Int(300),
    
    ep=type(1.0e-6), p=type(0.6),

    omega=type(0.1),

    cfl=type(0.3)
)

# Anfangsbedingungen
vx0(x::T, y::T) where T = T(0.0)
# vy0(x::T, y::T) where T = 3.0 <= x <= 3.5 && 1.2 <= y <= 1.4 ? T(-1.0) : T(0.0)
vy0(x::T, y::T) where T = T(0.0)

c0(x::T, y::T) where T = y <= 0.4 * sin((1 / 6.25) * 2 * pi * x) + 1.5 / 2 ? T(1.0) : T(0.0)
# c0(x::T, y::T) where T = y <= 0.75 ? T(1.0) : T(0.0)

#=
param = Parameters(
    Lx=type(1.0),
    Ly=type(3.0),
    
    # Wasser (c = 1)
    dynamic_viscosity_1=type(1e-3),
    density_1=type(998.0),

    # Öl (c = 0)
    dynamic_viscosity_2=type(5e-2),
    density_2=type(950.0),
    
    nx=Int(2.5*300),
    ny=Int(2.5*900),
    
    ep=type(1.0e-6), p=type(0.6),

    omega=type(0.3),

    cfl=type(0.3)
)

# Anfangsbedingungen
vx0(x::T, y::T) where T = T(0.0)
vy0(x::T, y::T) where T = T(0.0)

c0(x::T, y::T) where T = y <= 0.2*cos(2*pi*x) + 1.5 ? T(0.0) : T(1.0)
=#

navier_stokes_u0 = get_navier_stokes_u0(param)
advection_u0 = get_advection_u0(param)

# heatmap(Matrix(reshape(navier_stokes_u0[param.nx_vx*param.ny_vx+1:end], param.nx_vy, param.ny_vy)))
heatmap(Matrix(reshape(advection_u0, param.nx_c, param.ny_c)))

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
function run(; navier_stokes_u0::CuVector{T}, advection_u0::CuVector{T}, tspan, saveat, initial_time_step, param) where T
    navier_stokes_solver = Tsit5((stage_limiter!)=navier_stokes_limiter!)
    navier_stokes_prob = ODEProblem(navier_stokes!, navier_stokes_u0, tspan, param)
    navier_stokes_integrator = init(navier_stokes_prob, navier_stokes_solver; save_everystep=false, save_start=false, save_end=false, adaptive=false, dt=initial_time_step)

    advection_solver = Tsit5((stage_limiter!)=advection_limiter!)
    advection_prob = ODEProblem(advection!, advection_u0, tspan, param)
    advection_integrator = init(advection_prob, advection_solver; save_everystep=false, save_start=false, save_end=false, adaptive=false, dt=initial_time_step)

    next_save = iterate(saveat)
    t_start = tspan[1]
    t_end = tspan[2]
    ts = T[]

    solutions = Vector{NTuple{4,Matrix{T}}}(undef, 0)

    # speichere die Anfangswerte falls gefordert
    if next_save[1] <= t_start
        push!(ts, navier_stokes_integrator.t)
        push!(solutions, get_current_solution(navier_stokes_integrator, advection_integrator))
        next_save = iterate(saveat, next_save[2])
    end

    while navier_stokes_integrator.t < t_end
        time_step = calculate_time_step(navier_stokes_integrator)
        time_step = time_step <= T(5e-4) ? time_step : T(5e-4)
        set_proposed_dt!(navier_stokes_integrator, time_step)
        set_proposed_dt!(advection_integrator, time_step)

        step!(advection_integrator)
        advection_retcode = check_error(advection_integrator)
        c = reshape(advection_integrator.u, param.nx_c, param.ny_c)
        apply_boundary_conditions_c!(c, param)
        c .= clamp.(c, T(0.0), T(1.0))
        param.c_advected .= c

        step!(navier_stokes_integrator)
        navier_stokes_retcode = check_error(navier_stokes_integrator)

        update_velocities_cb!(navier_stokes_integrator)

        # speichere die aktuellen Werte falls gefordert
        if next_save !== nothing && next_save[1] <= navier_stokes_integrator.t
            push!(ts, navier_stokes_integrator.t)
            push!(solutions, get_current_solution(navier_stokes_integrator, advection_integrator))
            next_save = iterate(saveat, next_save[2])
        end

        if !SciMLBase.successful_retcode(navier_stokes_retcode)
            println("Fehler beim Navier-Stokes-Integrator:")
            println(navier_stokes_retcode)
            return navier_stokes_integrator, advection_integrator, ts, solutions
        end

        if !SciMLBase.successful_retcode(advection_retcode)
            println("Fehler beim Konvektions-Integrator:")
            println(advection_retcode)
            return navier_stokes_integrator, advection_integrator, ts, solutions
        end

        println(navier_stokes_integrator.t)
    end

    return navier_stokes_integrator, advection_integrator, ts, solutions
end

navier_stokes_integrator, advection_integrator, ts, solutions = CUDA.@time run(navier_stokes_u0=navier_stokes_u0, advection_u0=advection_u0, tspan=tspan, saveat=saveat, initial_time_step=type(1e-5), param=param)
display(navier_stokes_integrator.stats)

# Plotting
t = Observable(tspan[2])
vx = @lift get_solution(ts, solutions, $t, param)[1]
vy = @lift get_solution(ts, solutions, $t, param)[2]
velocity_norm = @lift sqrt.($vx .^ 2 .+ $vy .^ 2)
pressure = @lift get_solution(ts, solutions, $t, param)[3]
c = @lift get_solution(ts, solutions, $t, param)[4]

#=
# Plotte das Geschwindigkeitsfeld
fig = Figure(size=plot_size)
ax = GLMakie.Axis(fig[1, 1], title="Vektorfeld der Fließgeschwindigkeiten")

xs = range(param.dx/2, param.Lx-param.dx/2; length = param.nx_pressure)
ys = range(param.dy/2, param.Ly-param.dy/2; length = param.ny_pressure)

arrows2d!(xs, ys, vx, vy; 
        lengthscale = 0.02, normalize = true, 
        # arrowsize = 10, linewidth = 2.5,
        color = velocity_norm, colormap = :viridis)
save("plots/fluid_2d.png", fig)

t[] = type(2.5)

GLMakie.record(fig, "plots/fluid_2d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do time
    t[] = time
end
=#

# Plotte Stromlinien
vx_itp = @lift interpolate($vx, BSpline(Linear()))
vy_itp = @lift interpolate($vy, BSpline(Linear()))
vx_etp = @lift extrapolate($vx_itp, Flat())
vy_etp = @lift extrapolate($vy_itp, Flat())
vel_interp(x, y; field, dx, dy) = Point2(field[1](x / dx, y / dy), field[2](x / dx, y / dy))
sf = @lift (x, y) -> vel_interp(x, y; field=($vx_etp, $vy_etp), dx=param.dx, dy=param.dy)

fig = Figure(size=plot_size)
ax = GLMakie.Axis(fig[1, 1], title="Strömungslinien")
streamplot!(sf, 0 .. param.Lx, 0 .. param.Ly, colorscale=identity, colormap=:viridis)
save("plots/fluid_stream_2d.png", fig)

t[] = type(2.5)

GLMakie.record(fig, "plots/fluid_stream_2d.mp4", range(tspan[1], tspan[2]; length=n_frames); framerate=fps) do time
    t[] = time
end

# Plotte das Tempo (Betrag der Geschwindigkeit)
fig = Figure(size=plot_size)
ax = GLMakie.Axis(fig[1, 1], title="Betrag der Geschwindigkeit")
heatmap!(velocity_norm)
save("plots/fluid_norm_2d.png", fig)

t[] = type(2.5)

GLMakie.record(fig, "plots/fluid_norm_2d.mp4", range(tspan[1], tspan[2]; length=n_frames); framerate=fps) do time
    t[] = time
end

# Plotte den Druck
fig = Figure(size=plot_size)
ax = GLMakie.Axis(fig[1, 1], title="Druck")
heatmap!(pressure)
save("plots/fluid_pressure_2d.png", fig)

t[] = type(2.5)

GLMakie.record(fig, "plots/fluid_pressure_2d.mp4", range(tspan[1], tspan[2]; length=n_frames); framerate=fps) do time
    t[] = time
end

# Plotte die Phasenverteilung
fig = Figure(size=plot_size)
ax = GLMakie.Axis(fig[1, 1], title="Phasenverteilung")
heatmap!(c)
save("plots/phase_distribution_2d.png", fig)

t[] = type(2.5)

GLMakie.record(fig, "plots/phase_distribution_2d.mp4", range(tspan[1], tspan[2]; length=n_frames); framerate=fps) do time
    t[] = time
end