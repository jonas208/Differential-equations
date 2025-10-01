function get_coefficient_mat_cpu(divergence_mat::Matrix{T}, nx, ny, dx, dy, obstacles) where T
    linear_indices = LinearIndices(divergence_mat)
    I = Int[]; J = Int[]; X = T[];
    for j in 1:ny, i in 1:nx
        row = linear_indices[i, j]
        # innerer Bereich
        if 3 <= i <= nx-2 && 3 <= j <= ny-2 && !position_in_obstacle(obstacles, i, j)
            col1 = linear_indices[i, j]
            col2 = linear_indices[i+1, j]
            col3 = linear_indices[i-1, j]
            col4 = linear_indices[i, j-1]
            col5 = linear_indices[i, j+1]

            push!(I, row)
            push!(I, row)
            push!(I, row)
            push!(I, row)
            push!(I, row)

            push!(J, col1)
            push!(J, col2)
            push!(J, col3)
            push!(J, col4)
            push!(J, col5)

            push!(X, T(-2.0*(1/dx^2 + 1/dy^2)))
            push!(X, T(1.0/dx^2))
            push!(X, T(1.0/dx^2))
            push!(X, T(1.0/dy^2))
            push!(X, T(1.0/dy^2))
        end
        # Ränder
        if i <= 2 # Neumann
            col1 = linear_indices[i+1, j]
            col2 = linear_indices[i, j]

            push!(I, row)
            push!(I, row)

            push!(J, col1)
            push!(J, col2)

            push!(X, T(-1.0))
            push!(X, T(1.0))
        end
        if i >= nx-1
            # Dirichlet
            col = linear_indices[i, j]

            push!(I, row)
            push!(J, col)
            push!(X, T(1.0))
        end
        if j <= 2 # Neumann
            col1 = linear_indices[i, j+1]
            col2 = linear_indices[i, j]
            
            push!(I, row)
            push!(I, row)

            push!(J, col1)
            push!(J, col2)

            push!(X, T(-1.0))
            push!(X, T(1.0))
        end
        if j >= ny-1 # Neumann
            col1 = linear_indices[i, j-1]
            col2 = linear_indices[i, j]
            
            push!(I, row)
            push!(I, row)

            push!(J, col1)
            push!(J, col2)

            push!(X, T(-1.0))
            push!(X, T(1.0))
        end

        for obstacle in obstacles
            i0, i1, j0, j1 = get_indices(obstacle)
        
            # i0, i1; j0, j1
            if i0 <= i <= i1 && j0 <= j <= j1
                # Kanten
                if i0+1 <= i <= i1-1 && j0+1 <= j <= j1-1 # Dirichlet
                    col = linear_indices[i, j]
                    
                    push!(I, row)
                    push!(J, col)
                    push!(X, T(1.0))
                end
                if i == i0 && j != j0 && j != j1 # Neumann
                    col1 = linear_indices[i-1, j]
                    col2 = linear_indices[i, j]
                    
                    push!(I, row)
                    push!(I, row)

                    push!(J, col1)
                    push!(J, col2)

                    push!(X, T(-1.0))
                    push!(X, T(1.0))
                end
                if i == i1 && j != j0 && j != j1 # Neumann
                    col1 = linear_indices[i+1, j]
                    col2 = linear_indices[i, j]
                    
                    push!(I, row)
                    push!(I, row)

                    push!(J, col1)
                    push!(J, col2)

                    push!(X, T(-1.0))
                    push!(X, T(1.0))
                end
                if j == j0 && i != i0 && i != i1 # Neumann
                    col1 = linear_indices[i, j-1]
                    col2 = linear_indices[i, j]
                    
                    push!(I, row)
                    push!(I, row)

                    push!(J, col1)
                    push!(J, col2)

                    push!(X, T(-1.0))
                    push!(X, T(1.0))
                end
                if j == j1 && i != i0 && i != i1 # Neumann
                    col1 = linear_indices[i, j+1]
                    col2 = linear_indices[i, j]
                    
                    push!(I, row)
                    push!(I, row)

                    push!(J, col1)
                    push!(J, col2)

                    push!(X, T(-1.0))
                    push!(X, T(1.0))
                end

                # Ecken
                if i == i0 && j == j0 # Neumann
                    col1 = linear_indices[i, j]
                    col2 = linear_indices[i-1, j-1]
                    
                    push!(I, row)
                    push!(I, row)

                    push!(J, col1)
                    push!(J, col2)

                    push!(X, T(-1.0))
                    push!(X, T(1.0))
                end
                if i == i0 && j == j1 # Neumann
                    col1 = linear_indices[i-1, j+1]
                    col2 = linear_indices[i, j]
                    
                    push!(I, row)
                    push!(I, row)

                    push!(J, col1)
                    push!(J, col2)

                    push!(X, T(-1.0))
                    push!(X, T(1.0))
                end
                if i == i1 && j == j0 # Neumann
                    col1 = linear_indices[i+1, j-1]
                    col2 = linear_indices[i, j]
                    
                    push!(I, row)
                    push!(I, row)

                    push!(J, col1)
                    push!(J, col2)

                    push!(X, T(-1.0))
                    push!(X, T(1.0))
                end
                if i == i1 && j == j1 # Neumann
                    col1 = linear_indices[i+1, j+1]
                    col2 = linear_indices[i, j]
                    
                    push!(I, row)
                    push!(I, row)

                    push!(J, col1)
                    push!(J, col2)

                    push!(X, T(-1.0))
                    push!(X, T(1.0))
                end
            end

        end
    end

    coefficient_mat = sparse(I, J, X, nx*ny, nx*ny, *)
    return coefficient_mat
end

function calculate_pressure(param, time_step::T, vx::AbstractMatrix{T}, vy::AbstractMatrix{T}) where T
    Lx, Ly, kinematic_viscosity, density, horizontal_velocity, obstacles, nx, ny, dx, dy, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param

    divergence_mat .= T(0.0)

    for j in 3:ny-2, i in 3:nx-2
        divergence_mat[i, j] = (vx[i+1, j] - vx[i-1, j])/(2*dx) + (vy[i, j+1] - vy[i, j-1])/(2*dy)
    end

    for obstacle in obstacles
        i0, i1, j0, j1 = get_indices(obstacle)
        divergence_mat[i0:i1, j0:j1] .= T(0.0)
    end
    
    divergence_mat[nx-1, :] .= T(0.0)
    divergence_mat[nx, :] .= T(0.0)

    divergence_mat *= (density/time_step)

    lin_solve.b = vec(divergence_mat)
    pressure_vec .= solve!(lin_solve)
    # pressure_vec .= convert(Vector{T}, coefficient_mat \ vec(divergence_mat))
    pressure = reshape(pressure_vec, nx, ny)

    return pressure
end

function calculate_pressure_kernel!(divergence_mat::CuDeviceMatrix{T}, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, obstacle_indices::CuDeviceVector, density::T, nx, ny, dx::T, dy::T, time_step::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i <= nx && j <= ny
        divergence_mat[i, j] = T(0.0)
        if 3 <= i <= nx-2 && 3 <= j <= ny-2
            divergence_mat[i, j] = (vx[i+1, j] - vx[i-1, j])/(2*dx) + (vy[i, j+1] - vy[i, j-1])/(2*dy)
            divergence_mat[i, j] *= (density/time_step)
        end
        for k in 1:4:length(obstacle_indices)
            i0 = obstacle_indices[k]
            i1 = obstacle_indices[k+1]
            j0 = obstacle_indices[k+2]
            j1 = obstacle_indices[k+3]
            if i0 <= i <= i1 && j0 <= j <= j1
                divergence_mat[i, j] = T(0.0)
            end
        end
    end

    return nothing
end

function calculate_pressure(param, time_step::T, vx::CuMatrix{T}, vy::CuMatrix{T}) where T
    Lx, Ly, kinematic_viscosity, density, horizontal_velocity, obstacles, nx, ny, dx, dy, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param

    obstacle_indices = CuArray(get_obstacle_indices(obstacles))

    kernel = @cuda launch=false calculate_pressure_kernel!(divergence_mat, vx, vy, obstacle_indices, density, nx, ny, dx, dy, time_step)
    config = CUDA.launch_configuration(kernel.fun)
    
    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(nx, threads_per_dim)
    x_blocks = cld(nx, x_threads)

    y_threads = min(ny, threads_per_dim)
    y_blocks = cld(ny, y_threads)

    kernel(divergence_mat, vx, vy, obstacle_indices, density, nx, ny, dx, dy, time_step, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))

    x = pressure_vec
    b = vec(divergence_mat)
    cudss("solve", lin_solve, x, b)
    pressure = reshape(x, nx, ny)

    return pressure
end

function update_velocities!(vx::AbstractMatrix{T}, vy::AbstractMatrix{T}, pressure::AbstractMatrix{T}, density::T, nx, ny, dx::T, dy::T, time_step::T) where T
    for j in 3:ny-2, i in 3:nx-2
        vx[i, j] -= (time_step / density) * ((pressure[i+1, j] - pressure[i-1, j]) / (2*dx))
        vy[i, j] -= (time_step / density) * ((pressure[i, j+1] - pressure[i, j-1]) / (2*dy))
    end
end

function pressure_correction_cpu!(integrator)
    u = integrator.u
    param = integrator.p 
    Lx, Ly, kinematic_viscosity, density, horizontal_velocity, obstacles, nx, ny, dx, dy, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param
    time_step = integrator.t - integrator.tprev

    vx = reshape(view(u, 1:nx*ny), nx, ny)
    vy = reshape(view(u, nx*ny+1:2*nx*ny), nx, ny)

    apply_boundary_conditions!(vx, vy, horizontal_velocity, obstacles)
    pressure = calculate_pressure(param, time_step, vx, vy)
    update_velocities!(vx, vy, pressure, density, nx, ny, dx, dy, time_step)
    apply_boundary_conditions!(vx, vy, horizontal_velocity, obstacles)

    println(integrator.t)
end

function update_velocities_kernel!(vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, pressure::CuDeviceMatrix{T}, density::T, nx, ny, dx::T, dy::T, time_step::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if 3 <= i <= nx-2 && 3 <= j <= ny-2
        vx[i, j] -= (time_step / density) * ((pressure[i+1, j] - pressure[i-1, j]) / (2*dx))
        vy[i, j] -= (time_step / density) * ((pressure[i, j+1] - pressure[i, j-1]) / (2*dy))
    end

    return nothing
end

function update_velocities!(vx::CuMatrix{T}, vy::CuMatrix{T}, pressure::CuMatrix{T}, density::T, nx, ny, dx::T, dy::T, time_step::T) where T
    kernel = @cuda launch=false update_velocities_kernel!(vx, vy, pressure, density, nx, ny, dx, dy, time_step)
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
    Lx, Ly, kinematic_viscosity, density, horizontal_velocity, obstacles, nx, ny, dx, dy, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param
    time_step = integrator.t - integrator.tprev

    vx = reshape(view(u, 1:nx*ny), nx, ny)
    vy = reshape(view(u, nx*ny+1:2*nx*ny), nx, ny)

    apply_boundary_conditions!(vx, vy, horizontal_velocity, obstacles)
    pressure = calculate_pressure(param, time_step, vx, vy)
    update_velocities!(vx, vy, pressure, density, nx, ny, dx, dy, time_step)
    apply_boundary_conditions!(vx, vy, horizontal_velocity, obstacles)

    println(integrator.t)
end