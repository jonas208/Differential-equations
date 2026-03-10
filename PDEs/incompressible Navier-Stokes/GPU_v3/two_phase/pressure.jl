function get_coefficient_mat_cpu(nx, ny, dx, dy, is_solid::BitMatrix, rho::AbstractMatrix{T}) where T
    linear_indices = LinearIndices((nx, ny))
    rowPtr = Int[]
    colVal = Int[]
    nzVal = T[]

    push!(rowPtr, 1)

    # Die Schleifenreihenfolge ist entscheident, denn die linearen Indizes (= rows) müssen in aufsteigend sortierter Reihenfolge durchlaufen werden
    for j in 1:ny, i in 1:nx
        row = linear_indices[i, j] # (j - 1) * nx + i

        # innerer Bereich der Hindernisse
        if is_solid[i, j]
            is_deep_solid = true
            if i > 1 && !is_solid[i-1, j]
                is_deep_solid = false
            end
            if i < nx && !is_solid[i+1, j]
                is_deep_solid = false
            end
            if j > 1 && !is_solid[i, j-1]
                is_deep_solid = false
            end
            if j < ny && !is_solid[i, j+1]
                is_deep_solid = false
            end

            if is_deep_solid
                push!(colVal, row)
                push!(nzVal, T(1.0))
                push!(rowPtr, rowPtr[end] + 1)
                continue
            end
        end

        # Ausströmungsbedingung (Dirichlet)
        if i >= nx
            push!(colVal, row)
            push!(nzVal, T(1.0))
            push!(rowPtr, rowPtr[end] + 1)
            continue
        end

        diag_val = T(0.0)
        row_length = 0

        # Stelle durch die richtige Reihenfolge sicher, dass die Spaltenindizes innerhalb jeder Zeile strikt aufsteigend sortiert sind

        # Fluss nach unten (j-1), kleinster linearer Spaltenindex (row - nx)
        # ((j-1) - 1) * nx + i = (j-1) * nx - nx + i = row - nx 
        if j > 1 && !is_solid[i, j-1]
            # Inverse des harmonischen Mittels
            inv_rho_face = T(0.5) * (T(1.0) / rho[i, j] + T(1.0) / rho[i, j-1])
            coeff = inv_rho_face * T(1.0) / (dy^2)

            push!(colVal, linear_indices[i, j-1])
            push!(nzVal, coeff)
            diag_val -= coeff

            row_length += 1
        end # fester Rand, d.h. kein Fluss wegen No-slip-Bedingung (Neumann)

        # Fluss nach links (i-1), zweitkleinster Spaltenindex (row - 1)
        # (j - 1) * nx + (i-1) = row - 1
        if i > 1 && !is_solid[i-1, j] # kein fester Rand
            inv_rho_face = T(0.5) * (T(1.0) / rho[i, j] + T(1.0) / rho[i-1, j])
            coeff = inv_rho_face * T(1.0) / (dx^2)

            push!(colVal, linear_indices[i-1, j])
            push!(nzVal, coeff)
            diag_val -= coeff

            row_length += 1
        end

        # Diagonale (eigene Zelle), mittlerer Spaltenindex (row)
        diag_index = length(colVal) + 1
        push!(colVal, row)
        push!(nzVal, diag_val)

        row_length += 1

        # Fluss nach rechts (i+1), zweitgrößter Spaltenindex (row + 1)
        # (j - 1) * nx + (i+1) = row + 1
        if i < nx && !is_solid[i+1, j]
            inv_rho_face = T(0.5) * (T(1.0) / rho[i, j] + T(1.0) / rho[i+1, j])
            coeff = inv_rho_face * T(1.0) / (dx^2)

            push!(colVal, linear_indices[i+1, j])
            push!(nzVal, coeff)
            nzVal[diag_index] -= coeff

            row_length += 1
        end

        # Fluss nach oben (j+1), größter Spaltenindex (row + nx)
        # ((j+1) - 1) * nx + i = ((j-1) + 1) * nx + i = (j-1) * nx + nx + i = row + nx
        if j < ny && !is_solid[i, j+1]
            inv_rho_face = T(0.5) * (T(1.0) / rho[i, j] + T(1.0) / rho[i, j+1])
            coeff = inv_rho_face * T(1.0) / (dy^2)

            push!(colVal, linear_indices[i, j+1])
            push!(nzVal, coeff)
            nzVal[diag_index] -= coeff

            row_length += 1
        end

        push!(rowPtr, rowPtr[end] + row_length)
    end

    return rowPtr, colVal, nzVal
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