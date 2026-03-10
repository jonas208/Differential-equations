function get_coefficient_mat_cpu(divergence_mat::Matrix{T}, nx, ny, dx, dy, is_solid::BitMatrix) where T
    linear_indices = LinearIndices(divergence_mat)
    I = Int[]
    J = Int[]
    X = T[]

    for j in 1:ny, i in 1:nx
        row = linear_indices[i, j]

        # Ränder 
        if i <= 2 # Einströmungsbedingung (Neumann)
            push!(I, row, row)
            push!(J, linear_indices[i+1, j], row)
            push!(X, T(-1.0), T(1.0))
            continue
        elseif i >= nx - 1 # Ausströmungsbedingung (Dirichlet)
            push!(I, row)
            push!(J, row)
            push!(X, T(1.0))
            continue
        elseif j <= 2 # No-slip-Bedingung (Neumann)
            push!(I, row, row)
            push!(J, linear_indices[i, j+1], row)
            push!(X, T(-1.0), T(1.0))
            continue
        elseif j >= ny - 1 # No-slip-Bedingung (Neumann)
            push!(I, row, row)
            push!(J, linear_indices[i, j-1], row)
            push!(X, T(-1.0), T(1.0))
            continue
        end

        # Hindernisse
        if is_solid[i, j]
            # Ränder des Hindernisses
            if !is_solid[i-1, j] # No-slip-Bedingung (Neumann)
                push!(I, row, row)
                push!(J, linear_indices[i-1, j], row)
                push!(X, T(-1.0), T(1.0))
            elseif !is_solid[i+1, j] # No-slip-Bedingung (Neumann)
                push!(I, row, row)
                push!(J, linear_indices[i+1, j], row)
                push!(X, T(-1.0), T(1.0))
            elseif !is_solid[i, j-1] # No-slip-Bedingung (Neumann)
                push!(I, row, row)
                push!(J, linear_indices[i, j-1], row)
                push!(X, T(-1.0), T(1.0))
            elseif !is_solid[i, j+1] # No-slip-Bedingung (Neumann)
                push!(I, row, row)
                push!(J, linear_indices[i, j+1], row)
                push!(X, T(-1.0), T(1.0))
            else # innerer Bereich des Hindernisses
                push!(I, row)
                push!(J, row)
                push!(X, 1.0)
            end
            continue
        end

        # innerer Bereich ohne Hindernisse
        push!(I, row, row, row, row, row)
        push!(J, row, linear_indices[i+1, j], linear_indices[i-1, j], linear_indices[i, j-1], linear_indices[i, j+1])
        push!(X, T(-2.0 * (1 / dx^2 + 1 / dy^2)), T(1.0 / dx^2), T(1.0 / dx^2), T(1.0 / dy^2), T(1.0 / dy^2))
    end

    coefficient_mat = sparse(I, J, X, nx * ny, nx * ny)
    return coefficient_mat
end

function calculate_pressure(param, time_step::T, vx::AbstractMatrix{T}, vy::AbstractMatrix{T}) where T
    Lx, Ly, kinematic_viscosity, density, horizontal_velocity, simulate_smoke, is_solid, nx, ny, dx, dy, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param

    divergence_mat .= T(0.0)

    for j in 3:ny-2, i in 3:nx-2
        divergence_mat[i, j] = (vx[i+1, j] - vx[i-1, j]) / (2 * dx) + (vy[i, j+1] - vy[i, j-1]) / (2 * dy)
    end

    for j in 1:ny, i in 1:nx
        if is_solid[i, j]
            divergence_mat[i, j] = T(0.0)
        end
    end

    divergence_mat *= (density / time_step)

    lin_solve.b = vec(divergence_mat)
    pressure_vec .= solve!(lin_solve)
    pressure = reshape(pressure_vec, nx, ny)

    return pressure
end

function calculate_pressure_kernel!(divergence_mat::CuDeviceMatrix{T}, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, is_solid::CuDeviceMatrix{Bool}, density::T, nx, ny, dx::T, dy::T, time_step::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i <= nx && j <= ny
        divergence_mat[i, j] = T(0.0)
        if 3 <= i <= nx - 2 && 3 <= j <= ny - 2
            divergence_mat[i, j] = (vx[i+1, j] - vx[i-1, j]) / (2 * dx) + (vy[i, j+1] - vy[i, j-1]) / (2 * dy)
            divergence_mat[i, j] *= (density / time_step)
        end
        if is_solid[i, j]
            divergence_mat[i, j] = T(0.0)
        end
    end

    return nothing
end

function calculate_pressure(param, time_step::T, vx::CuMatrix{T}, vy::CuMatrix{T}) where T
    Lx, Ly, kinematic_viscosity, density, horizontal_velocity, simulate_smoke, is_solid, nx, ny, dx, dy, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param

    kernel = @cuda launch = false calculate_pressure_kernel!(divergence_mat, vx, vy, is_solid, density, nx, ny, dx, dy, time_step)
    config = CUDA.launch_configuration(kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(nx, threads_per_dim)
    x_blocks = cld(nx, x_threads)

    y_threads = min(ny, threads_per_dim)
    y_blocks = cld(ny, y_threads)

    kernel(divergence_mat, vx, vy, is_solid, density, nx, ny, dx, dy, time_step, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))

    x = pressure_vec
    b = vec(divergence_mat)
    cudss("solve", lin_solve, x, b)
    pressure = reshape(x, nx, ny)

    return pressure
end

function update_velocities!(vx::AbstractMatrix{T}, vy::AbstractMatrix{T}, pressure::AbstractMatrix{T}, density::T, nx, ny, dx::T, dy::T, time_step::T) where T
    for j in 3:ny-2, i in 3:nx-2
        vx[i, j] -= (time_step / density) * ((pressure[i+1, j] - pressure[i-1, j]) / (2 * dx))
        vy[i, j] -= (time_step / density) * ((pressure[i, j+1] - pressure[i, j-1]) / (2 * dy))
    end
end

function pressure_correction_cpu!(integrator)
    u = integrator.u
    param = integrator.p
    Lx, Ly, kinematic_viscosity, density, horizontal_velocity, simulate_smoke, is_solid, nx, ny, dx, dy, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param
    time_step = integrator.t - integrator.tprev

    vx = reshape(view(u, 1:nx*ny), nx, ny)
    vy = reshape(view(u, nx*ny+1:2*nx*ny), nx, ny)
    c = reshape(view(u, 2*nx*ny+1:3*nx*ny), nx, ny)

    apply_boundary_conditions!(vx, vy, c, horizontal_velocity, is_solid)
    pressure = calculate_pressure(param, time_step, vx, vy)
    update_velocities!(vx, vy, pressure, density, nx, ny, dx, dy, time_step)
    apply_boundary_conditions!(vx, vy, c, horizontal_velocity, is_solid)

    println(integrator.t)
end

function update_velocities_kernel!(vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, pressure::CuDeviceMatrix{T}, density::T, nx, ny, dx::T, dy::T, time_step::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if 3 <= i <= nx - 2 && 3 <= j <= ny - 2
        vx[i, j] -= (time_step / density) * ((pressure[i+1, j] - pressure[i-1, j]) / (2 * dx))
        vy[i, j] -= (time_step / density) * ((pressure[i, j+1] - pressure[i, j-1]) / (2 * dy))
    end

    return nothing
end

function update_velocities!(vx::CuMatrix{T}, vy::CuMatrix{T}, pressure::CuMatrix{T}, density::T, nx, ny, dx::T, dy::T, time_step::T) where T
    kernel = @cuda launch = false update_velocities_kernel!(vx, vy, pressure, density, nx, ny, dx, dy, time_step)
    config = CUDA.launch_configuration(kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(nx, threads_per_dim)
    x_blocks = cld(nx, x_threads)

    y_threads = min(ny, threads_per_dim)
    y_blocks = cld(ny, y_threads)

    kernel(vx, vy, pressure, density, nx, ny, dx, dy, time_step, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

function pressure_correction_gpu!(integrator)
    u = integrator.u
    param = integrator.p
    Lx, Ly, kinematic_viscosity, density, horizontal_velocity, simulate_smoke, is_solid, nx, ny, dx, dy, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param
    time_step = integrator.t - integrator.tprev

    vx = reshape(view(u, 1:nx*ny), nx, ny)
    vy = reshape(view(u, nx*ny+1:2*nx*ny), nx, ny)
    c = reshape(view(u, 2*nx*ny+1:3*nx*ny), nx, ny)

    apply_boundary_conditions!(vx, vy, c, horizontal_velocity, is_solid)
    pressure = calculate_pressure(param, time_step, vx, vy)
    update_velocities!(vx, vy, pressure, density, nx, ny, dx, dy, time_step)
    apply_boundary_conditions!(vx, vy, c, horizontal_velocity, is_solid)

    println(integrator.t)
end