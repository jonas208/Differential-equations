using LinearAlgebra
using LinearSolve
using SparseArrays
using OrdinaryDiffEq
using Interpolations
using GLMakie

# Spezifikationen des Problems
L = 1.0 # Seitenlänge der quadratischen Oberfläche [m]
kinematic_viscosity = 1.0e-6
density = 1.0
horizontal_velocity = 0.75
# horizontal_velocity = 0.5

# Hindernis
struct Rectangle
    i0
    i1
    j0
    j1
    function Rectangle(center, x_len, y_len, dx)
        i0 = Int(round((center[1] - (x_len/2)) / dx))
        i1 = Int(round((center[1] + (x_len/2)) / dx))
        j0 = Int(round((center[2] - (y_len/2)) / dx))
        j1 = Int(round((center[2] + (y_len/2)) / dx))
        
        new(i0, i1, j0, j1)
    end
end

get_indices(rectangle) = rectangle.i0, rectangle.i1, rectangle.j0, rectangle.j1

function position_in_obstacle(obstacles, i, j)
    for obstacle in obstacles
        i0, i1, j0, j1 = get_indices(obstacle)
        if (i0 <= i <= i1 && j0 <= j <= j1)
            return true
        end
    end
    return false
end

plot_size = (1000, 1000) # Auflösung der Plots
fps = 25 # Bildwiederholrate der Animationen

# Zeitintervall
# tspan = (0.0, 10.0)
tspan = (0.0, 0.05)
n_frames = ceil(Int, tspan[2]-tspan[1])*fps

# Anfangsbedingungen
vx0(x, y) = 0.0
vy0(x, y) = 0.0

# Randbedingungen 
# Wände oben und unten (Dirichlet für Geschwindigkeit und Neumann für Druck)
# Zufluss links (Dirichlet für Geschwindigkeit und Neumann für Druck)
# Abfluss rechts (Neumann für Geschwindigkeit und Dirichlet für Druck)

# Ortsdiskretisierung
n = 160
dx = L/n
xs = range(dx/2, L-dx/2; length = n)
ys = range(dx/2, L-dx/2; length = n)

obstacles = [
    Rectangle((0.3, 0.3), 0.35, 0.15, dx),
    Rectangle((0.3, 0.7), 0.35, 0.15, dx)
]

#=
obstacles = [
    Rectangle((0.175, 0.3), 0.15, 0.4, dx),
    Rectangle((0.7, 0.85), 0.2, 0.1, dx),
]
=#

# Koeffizientenmatrix für die Poisson-Gleichung für den Druck
coefficient_mat = spzeros(n^2, n^2)
divergence_mat = zeros(n, n)
linear_indices = LinearIndices(divergence_mat)
for j in 1:n, i in 1:n
    row = linear_indices[i, j]
    # innerer Bereich
    if 3 <= i <= n-2 && 3 <= j <= n-2 && !position_in_obstacle(obstacles, i, j)
        col1 = linear_indices[i, j]
        col2 = linear_indices[i+1, j]
        col3 = linear_indices[i-1, j]
        col4 = linear_indices[i, j-1]
        col5 = linear_indices[i, j+1]
        coefficient_mat[row, col1] = -4.0
        coefficient_mat[row, col2] = 1.0
        coefficient_mat[row, col3] = 1.0
        coefficient_mat[row, col4] = 1.0
        coefficient_mat[row, col5] = 1.0
    end
    # Ränder
    if i <= 2 # Neumann
        col1 = linear_indices[i+1, j]
        col2 = linear_indices[i, j]
        coefficient_mat[row, col1] = -1.0
        coefficient_mat[row, col2] = 1.0
    end
    if i >= n-1
        # Dirichlet
        col = linear_indices[i, j]
        coefficient_mat[row, col] = 1.0
    end
    if j <= 2 # Neumann
        col1 = linear_indices[i, j+1]
        col2 = linear_indices[i, j]
        coefficient_mat[row, col1] = -1.0
        coefficient_mat[row, col2] = 1.0
    end
    if j >= n-1 # Neumann
        col1 = linear_indices[i, j-1]
        col2 = linear_indices[i, j]
        coefficient_mat[row, col1] = -1.0
        coefficient_mat[row, col2] = 1.0
    end

    for obstacle in obstacles
        i0, i1, j0, j1 = get_indices(obstacle)
    
        # i0, i1; j0, j1
        if i0 <= i <= i1 && j0 <= j <= j1
            # Kanten
            if i0+1 <= i <= i1-1 && j0+1 <= j <= j1-1 # Dirichlet
                col = linear_indices[i, j]
                coefficient_mat[row, col] = 1.0
            end
            if i == i0 && j != j0 && j != j1 # Neumann
                col1 = linear_indices[i-1, j]
                col2 = linear_indices[i, j]
                coefficient_mat[row, col1] = -1.0
                coefficient_mat[row, col2] = 1.0
            end
            if i == i1 && j != j0 && j != j1 # Neumann
                col1 = linear_indices[i+1, j]
                col2 = linear_indices[i, j]
                coefficient_mat[row, col1] = -1.0
                coefficient_mat[row, col2] = 1.0
            end
            if j == j0 && i != i0 && i != i1 # Neumann
                col1 = linear_indices[i, j-1]
                col2 = linear_indices[i, j]
                coefficient_mat[row, col1] = -1.0
                coefficient_mat[row, col2] = 1.0
            end
            if j == j1 && i != i0 && i != i1 # Neumann
                col1 = linear_indices[i, j+1]
                col2 = linear_indices[i, j]
                coefficient_mat[row, col1] = -1.0
                coefficient_mat[row, col2] = 1.0
            end

            # Ecken
            if i == i0 && j == j0 # Neumann
                col1 = linear_indices[i, j]
                col2 = linear_indices[i-1, j-1]
                coefficient_mat[row, col1] = -1.0
                coefficient_mat[row, col2] = 1.0
            end
            if i == i0 && j == j1 # Neumann
                col1 = linear_indices[i-1, j+1]
                col2 = linear_indices[i, j]
                coefficient_mat[row, col1] = -1.0
                coefficient_mat[row, col2] = 1.0
            end
            if i == i1 && j == j0 # Neumann
                col1 = linear_indices[i+1, j-1]
                col2 = linear_indices[i, j]
                coefficient_mat[row, col1] = -1.0
                coefficient_mat[row, col2] = 1.0
            end
            if i == i1 && j == j1 # Neumann
                col1 = linear_indices[i+1, j+1]
                col2 = linear_indices[i, j]
                coefficient_mat[row, col1] = -1.0
                coefficient_mat[row, col2] = 1.0
            end
        end

    end
end

# Definiere ein lineares Gleichungssystem der Form Ax = b
A = coefficient_mat
b = vec(divergence_mat)
lin_prob = LinearProblem(A, b)
lin_solver = KLUFactorization()
lin_solve = init(lin_prob, lin_solver)

# WENO-Parameter
use_weno = true
ep = 1.0e-6; p = 0.6;

# Parameter
flux_y_r = Matrix{Float64}(undef, n, 2)
param = (L, kinematic_viscosity, density, horizontal_velocity, obstacles, n, dx, coefficient_mat, divergence_mat, lin_solve, use_weno, ep, p, flux_y_r)

# DGL-System erster Ordnung (Linienmethode mit finiten Differenzen)
function navier_stokes_fdm!(du, u, param, t)
    L, kinematic_viscosity, density, horizontal_velocity, obstacles, n, dx, coefficient_mat, divergence_mat, lin_solve, use_weno, ep, p, flux_y_r = param

    vx = reshape(view(u, 1:n^2), n, n)
    vy = reshape(view(u, n^2+1:2*n^2), n, n)

    dvx = reshape(view(du, 1:n^2), n, n)
    dvy = reshape(view(du, n^2+1:2*n^2), n, n)

    for j in 3:n-2, i in 3:n-2
        if !position_in_obstacle(obstacles, i, j)
            spatial_deriv_vx_x = (vx[i+1, j] - vx[i-1, j]) / (2*dx)
            spatial_deriv_vx_y = (vx[i, j+1] - vx[i, j-1]) / (2*dx)
            laplacian_vx = (vx[i+1, j] + vx[i-1, j] + vx[i, j+1] + vx[i, j-1] - 4*vx[i, j]) / dx^2
            dvx[i, j] = -vx[i, j]*spatial_deriv_vx_x - vy[i, j]*spatial_deriv_vx_y + kinematic_viscosity*laplacian_vx

            spatial_deriv_vy_x = (vy[i+1, j] - vy[i-1, j]) / (2*dx)
            spatial_deriv_vy_y = (vy[i, j+1] - vy[i, j-1]) / (2*dx)
            laplacian_vy = (vy[i+1, j] + vy[i-1, j] + vy[i, j+1] + vy[i, j-1] - 4*vy[i, j]) / dx^2
            dvy[i, j] = -vx[i, j]*spatial_deriv_vy_x - vy[i, j]*spatial_deriv_vy_y + kinematic_viscosity*laplacian_vy
        end
    end

    println(t)
end

# 1D-CWENO-Interpolation in x-Richtung (P_{i}) mit zusätzlichem Term im Teilpolynom pC
# u_is = [u_{i-1,j}, u_{i,j}, u_{i+1,j}]
# u_js = [u_{i,j-1}, u_{i,j}, u_{i,j+1}]
function cweno_x(x, x_i, dx, u_is, u_js, ep, p)
    uL = u_is[2]-u_is[1]; uC = u_is[3]-2*u_is[2]+u_is[1]; uR = u_is[3]-u_is[2]; uCC = u_is[3]-u_is[1];
    ISL = uL^2; ISC = 13/3*uC^2 + 0.25*uCC^2; ISR = uR^2;
    aL = 0.25*(1/(ep+ISL))^p; aC = 0.5*(1/(ep+ISC))^p; aR = 0.25*(1/(ep+ISR))^p;
    suma = max(aL+aC+aR,eps(1.0)); 
    wL = aL/suma; wC = aC/suma; wR = aR/suma;
    pL = u_is[2] + uL/dx*(x-x_i);
    pC = u_is[2] - uC/12 - (u_js[3]-2*u_js[2]+u_js[1])/12 + uCC/(2*dx)*(x-x_i) + uC/dx^2*(x-x_i)^2;
    pR = u_is[2] + uR/dx*(x-x_i);
    return wL*pL + wC*pC + wR*pR
end

# 1D-CWENO-Interpolation in y-Richtung (P_{j}) mit zusätzlichem Term im Teilpolynom pC
# u_js = [u_{i,j-1}, u_{i,j}, u_{i,j+1}]
# u_is = [u_{i-1,j}, u_{i,j}, u_{i+1,j}]
function cweno_y(y, y_j, dy, u_js, u_is, ep, p)
    uL = u_js[2]-u_js[1]; uC = u_js[3]-2*u_js[2]+u_js[1]; uR = u_js[3]-u_js[2]; uCC = u_js[3]-u_js[1];
    ISL = uL^2; ISC = 13/3*uC^2 + 0.25*uCC^2; ISR = uR^2;
    aL = 0.25*(1/(ep+ISL))^p; aC = 0.5*(1/(ep+ISC))^p; aR = 0.25*(1/(ep+ISR))^p;
    suma = max(aL+aC+aR,eps(1.0)); 
    wL = aL/suma; wC = aC/suma; wR = aR/suma;
    pL = u_js[2] + uL/dy*(y-y_j);
    pC = u_js[2] - uC/12 - (u_is[3]-2*u_is[2]+u_is[1])/12 + uCC/(2*dy)*(y-y_j) + uC/dy^2*(y-y_j)^2;
    pR = u_js[2] + uR/dy*(y-y_j);
    return wL*pL + wC*pC + wR*pR
end

# berechne u^{\pm}_{i+1/2,j} wahlweise mit oder ohne WENO
function recover_x(i, j, dx, vx, vy, use_weno, ep, p)
    if use_weno
        vx_l = cweno_x(i*dx, (i-0.5)*dx, dx, view(vx, i-1:i+1, j), view(vx, i, j-1:j+1), ep, p) # P_i(x_{i+1/2})
        vx_r = cweno_x(i*dx, (i+0.5)*dx, dx, view(vx, i:i+2, j), view(vx, i+1, j-1:j+1), ep, p) # P_{i+1}(x_{i+1/2})
        vy_l = cweno_x(i*dx, (i-0.5)*dx, dx, view(vy, i-1:i+1, j), view(vy, i, j-1:j+1), ep, p) # P_i(x_{i+1/2})
        vy_r = cweno_x(i*dx, (i+0.5)*dx, dx, view(vy, i:i+2, j), view(vy, i+1, j-1:j+1), ep, p) # P_{i+1}(x_{i+1/2})
    else
        vx_l = vx[i, j]
        vy_l = vy[i, j]
        vx_r = vx[i+1, j]
        vy_r = vy[i+1, j]
    end
    return vx_l, vx_r, vy_l, vy_r
end

# berechne u^{\pm}_{i,j+1/2} wahlweise mit oder ohne WENO
function recover_y(i, j, dx, vx, vy, use_weno, ep, p)
    if use_weno
        vx_l = cweno_y(j*dx, (j-0.5)*dx, dx, view(vx, i, j-1:j+1), view(vx, i-1:i+1, j), ep, p) # P_j(y_{j+1/2})
        vx_r = cweno_y(j*dx, (j+0.5)*dx, dx, view(vx, i, j:j+2), view(vx, i-1:i+1, j+1), ep, p) # P_{j+1}(y_{j+1/2})
        vy_l = cweno_y(j*dx, (j-0.5)*dx, dx, view(vy, i, j-1:j+1), view(vy, i-1:i+1, j), ep, p) # P_j(y_{j+1/2})
        vy_r = cweno_y(j*dx, (j+0.5)*dx, dx, view(vy, i, j:j+2), view(vy, i-1:i+1, j+1), ep, p) # P_{j+1}(y_{j+1/2})
    else
        vx_l = vx[i, j]
        vy_l = vy[i, j]
        vx_r = vx[i, j+1]
        vy_r = vy[i, j+1]
    end
    return vx_l, vx_r, vy_l, vy_r
end

flux_x_1(vx, vy) = vx^2
flux_x_2(vx, vy) = vx*vy

flux_y_1(vx, vy) = vx*vy
flux_y_2(vx, vy) = vy^2

# berechne den numerischen Fluss in x-Richtung (H^x_{i+1/2,j})
function local_lax_friedrichs_x(vx_l, vx_r, vy_l, vy_r)
    # alpha = max(abs(vx_l), abs(vx_r))
    alpha = 2*max(abs(vx_l), abs(vx_r))
    flux_x_1_r = 0.5*(flux_x_1(vx_l, vy_l) + flux_x_1(vx_r, vy_r) - alpha*(vx_r - vx_l))
    flux_x_2_r = 0.5*(flux_x_2(vx_l, vy_l) + flux_x_2(vx_r, vy_r) - alpha*(vy_r - vy_l))

    return flux_x_1_r, flux_x_2_r
end

# berechne den numerischen Fluss in y-Richtung (H^y_{i,j+1/2})
function local_lax_friedrichs_y(vx_l, vx_r, vy_l, vy_r)
    # alpha = max(abs(vy_l), abs(vy_r))
    alpha = 2*max(abs(vy_l), abs(vy_r))
    flux_y_1_r = 0.5*(flux_y_1(vx_l, vy_l) + flux_y_1(vx_r, vy_r) - alpha*(vx_r - vx_l))
    flux_y_2_r = 0.5*(flux_y_2(vx_l, vy_l) + flux_y_2(vx_r, vy_r) - alpha*(vy_r - vy_l))

    return flux_y_1_r, flux_y_2_r
end

# DGL-System erster Ordnung (Linienmethode mit finiten Volumen)
function navier_stokes_fvm!(du, u, param, t)
    L, kinematic_viscosity, density, horizontal_velocity, obstacles, n, dx, coefficient_mat, divergence_mat, lin_solve, use_weno, ep, p, flux_y_r = param

    vx = reshape(view(u, 1:n^2), n, n)
    vy = reshape(view(u, n^2+1:2*n^2), n, n)

    dvx = reshape(view(du, 1:n^2), n, n)
    dvy = reshape(view(du, n^2+1:2*n^2), n, n)

    flux_x_1_r = 0.0
    flux_x_2_r = 0.0

    for j in 2:n-2, i in 2:n-2
        if i == 2
            vx_l, vx_r, vy_l, vy_r = recover_x(i, j, dx, vx, vy, use_weno, ep, p)
            flux_x_1_r, flux_x_2_r = local_lax_friedrichs_x(vx_l, vx_r, vy_l, vy_r)
        end

        if j == 2
            vx_l, vx_r, vy_l, vy_r = recover_y(i, j, dx, vx, vy, use_weno, ep, p)
            flux_y_r[i, 1], flux_y_r[i, 2] = local_lax_friedrichs_y(vx_l, vx_r, vy_l, vy_r)
        end

        if 3 <= i <= n-2 && 3 <= j <= n-2
            flux_x_1_l = flux_x_1_r
            flux_x_2_l = flux_x_2_r

            flux_y_1_l = flux_y_r[i, 1]
            flux_y_2_l = flux_y_r[i, 2]

            vx_l, vx_r, vy_l, vy_r = recover_x(i, j, dx, vx, vy, use_weno, ep, p)
            flux_x_1_r, flux_x_2_r = local_lax_friedrichs_x(vx_l, vx_r, vy_l, vy_r)

            vx_l, vx_r, vy_l, vy_r = recover_y(i, j, dx, vx, vy, use_weno, ep, p)
            flux_y_r[i, 1], flux_y_r[i, 2] = local_lax_friedrichs_y(vx_l, vx_r, vy_l, vy_r)

            if !position_in_obstacle(obstacles, i, j)
                laplacian_vx = (vx[i+1, j] + vx[i-1, j] + vx[i, j+1] + vx[i, j-1] - 4*vx[i, j]) / dx^2
                laplacian_vy = (vy[i+1, j] + vy[i-1, j] + vy[i, j+1] + vy[i, j-1] - 4*vy[i, j]) / dx^2

                dvx[i, j] = -(flux_x_1_r - flux_x_1_l)/dx - (flux_y_r[i, 1] - flux_y_1_l)/dx + kinematic_viscosity*laplacian_vx
                dvy[i, j] = -(flux_x_2_r - flux_x_2_l)/dx - (flux_y_r[i, 2] - flux_y_2_l)/dx + kinematic_viscosity*laplacian_vy
            end
        end
    end
end

function apply_boundary_conditions!(vx, vy, horizontal_velocity, obstacles)    
    # Dirichlet
    vx[1, :] .= horizontal_velocity
    vx[2, :] .= horizontal_velocity
    vx[:, 1] .= 0.0
    vx[:, 2] .= 0.0
    vx[:, end-1] .= 0.0
    vx[:, end] .= 0.0
    # Neumann
    vx[end-1, :] .= vx[end-2, :]
    vx[end, :] .= vx[end-1, :]

    # Dirichlet
    vy[1, :] .= 0.0
    vy[2, :] .= 0.0
    vy[:, 1] .= 0.0
    vy[:, 2] .= 0.0
    vy[:, end-1] .= 0.0
    vy[:, end] .= 0.0
    # Neumann
    vy[end-1, :] .= vy[end-2, :]
    vy[end, :] .= vy[end-1, :]

    # Dirichlet    
    for obstacle in obstacles
        i0, i1, j0, j1 = get_indices(obstacle)
        vx[i0:i1, j0:j1] .= 0.0
        vy[i0:i1, j0:j1] .= 0.0
    end
end

function calculate_pressure(param, time_step, vx, vy, time)
    L, kinematic_viscosity, density, horizontal_velocity, obstacles, n, dx, coefficient_mat, divergence_mat, lin_solve, use_weno, ep, p, flux_y_r = param

    divergence_mat .= 0.0

    for j in 3:n-2, i in 3:n-2
        divergence_mat[i, j] = (vx[i+1, j] - vx[i-1, j] + vy[i, j+1] - vy[i, j-1]) / (2*dx)
    end

    for obstacle in obstacles
        i0, i1, j0, j1 = get_indices(obstacle)
        divergence_mat[i0:i1, j0:j1] .= 0.0
    end
    
    divergence_mat[n-1, :] .= 0.0
    divergence_mat[n, :] .= 0.0

    divergence_mat *= (density/time_step) * dx^2

    #=
    # Definiere ein lineares Gleichungssystem der Form Ax = b
    A = coefficient_mat
    b = vec(divergence_mat)
    # Löse das lineare Gleichungssystem
    pressure = reshape(A \ b, n, n)
    =#
    lin_solve.b = vec(divergence_mat)
    pressure = reshape(solve!(lin_solve), n, n)

    return pressure
end

function pressure_correction!(integrator)
    u = integrator.u
    param = integrator.p 
    L, kinematic_viscosity, density, horizontal_velocity, obstacles, n, dx, coefficient_mat, divergence_mat, lin_solve, use_weno, ep, p, flux_y_r = param
    time_step = integrator.t - integrator.tprev

    vx = reshape(view(u, 1:n^2), n, n)
    vy = reshape(view(u, n^2+1:2*n^2), n, n)

    apply_boundary_conditions!(vx, vy, horizontal_velocity, obstacles)

    pressure = calculate_pressure(param, time_step, vx, vy, integrator.t)

    for j in 3:n-2, i in 3:n-2
        vx[i, j] -= (time_step / density) * ((pressure[i+1, j] - pressure[i-1, j]) / (2*dx))
        vy[i, j] -= (time_step / density) * ((pressure[i, j+1] - pressure[i, j-1]) / (2*dx))
    end

    apply_boundary_conditions!(vx, vy, horizontal_velocity, obstacles)

    println(integrator.t)
end

vx0s = [vx0(x, y) for x in xs, y in ys]
vy0s = [vy0(x, y) for x in xs, y in ys]

apply_boundary_conditions!(vx0s, vy0s, horizontal_velocity, obstacles)

u0 = vcat(vec(vx0s), vec(vy0s))

#=
using InteractiveUtils
using BenchmarkTools

u = u0
du = similar(u)
t = 0.0

@code_warntype navier_stokes_fvm!(du, u0, param, t)
@benchmark navier_stokes_fvm!($du, $u0, $param, $t)
=#

# f = ODEFunction(navier_stokes_fdm!)
f = ODEFunction(navier_stokes_fvm!)
prob = ODEProblem(f, u0, tspan, param)
condition(u, t, integrator) = true
cb = DiscreteCallback(condition, pressure_correction!)
# sol = @time solve(prob, Tsit5(); reltol = 1.0e-5, abstol = 1.0e-5, callback = cb, saveat = tspan[1]:1/fps:tspan[2])
sol = @time solve(prob, SSPRK432(); reltol = 1.0e-5, abstol = 1.0e-5, callback = cb, saveat = tspan[1]:1/fps:tspan[2])

display(sol.stats)

# Interface für Lösung

function get_velocity(t)
    v = sol(t)
    vx = reshape(view(v, 1:n^2), n, n)
    vy = reshape(view(v, n^2+1:2*n^2), n, n)
    return vx, vy
end

function get_pressure(vx, vy, t)
    time_step = sol.t[end] / div(length(sol.t), 2)
    return calculate_pressure(param, time_step, vx, vy, t)
end

# Plotting the velocity field

t = tspan[2]
us, vs = Observable.(get_velocity(t))
strength = @lift vec(sqrt.($us .^ 2 .+ $vs .^ 2))

pressure = @lift get_pressure($us, $vs, t)
itp = @lift interpolate($pressure, BSpline(Linear()))
etp = @lift extrapolate($itp, Flat())
pressure_interp = @lift [$etp(x/dx, y/dx) for x in xs, y in ys]

fig = Figure(size = plot_size)
ax = Axis(fig[1, 1], title = "Vektorfeld der Fließgeschwindigkeiten")
heatmap!(xs, ys, pressure_interp, colormap = :balance) # colormap = :balance
arrows!(xs, ys, us, vs; 
        lengthscale = 0.02, normalize = true, 
        arrowsize = 10, linewidth = 2.5,
        arrowcolor = strength, linecolor = strength,
        colormap = :viridis)
save("fluid_2d.png", fig)
display(fig)

us[], vs[] = get_velocity(2.5)

record(fig, "fluid_2d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do t
    us[], vs[] = get_velocity(t)
end

# Plotting the streamlines

us_itp = @lift interpolate($us, BSpline(Linear()))
vs_itp = @lift interpolate($vs, BSpline(Linear()))
us_etp = @lift extrapolate($us_itp, Flat())
vs_etp = @lift extrapolate($vs_itp, Flat())
vel_interp(x, y; field, dx) = Point2(field[1](x/dx, y/dx), field[2](x/dx, y/dx))
sf = @lift (x, y) -> vel_interp(x, y; field = ($us_etp, $vs_etp), dx = dx)

fig = Figure(size = plot_size)
ax = Axis(fig[1, 1], title = "Strömungslinien")
streamplot!(sf, 0..L, 0..L, colorscale = identity, colormap = :viridis)
save("fluid_stream_2d.png", fig)
display(fig)

us[], vs[] = get_velocity(tspan[2])

record(fig, "fluid_stream_2d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do t
    us[], vs[] = get_velocity(t)
end

# Plotting the norm of the velocity field

velocity_norm = @lift sqrt.($us .^ 2 .+ $vs .^ 2)

fig = Figure(size = plot_size)
ax = Axis(fig[1, 1], title = "Betrag der Geschwindigkeit")
heatmap!(velocity_norm)
save("fluid_norm_2d.png", fig)

us[], vs[] = get_velocity(2.5)

record(fig, "fluid_norm_2d.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do t
    us[], vs[] = get_velocity(t)
end