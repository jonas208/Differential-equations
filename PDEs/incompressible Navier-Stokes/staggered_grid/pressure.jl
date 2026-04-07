linear_indices(i, j, nx_pressure) = (j - 1) * nx_pressure + i

function get_coefficient_mat(nx_pressure, ny_pressure, dx::T, dy::T) where T
    rowPtr = Int[]
    colVal = Int[]
    nzVal = T[]

    push!(rowPtr, 1)

    # Die Schleifenreihenfolge ist entscheident, denn die linearen Indizes (= rows) müssen in aufsteigend sortierter Reihenfolge durchlaufen werden
    for j in 1:ny_pressure, i in 1:nx_pressure
        row = linear_indices(i, j, nx_pressure) # (j - 1) * nx + i
        
        diag_val = T(0.0)
        row_length = 0

        # Stelle durch die richtige Reihenfolge sicher, dass die Spaltenindizes innerhalb jeder Zeile strikt aufsteigend sortiert sind

        # Fluss nach unten (j-1), kleinster linearer Spaltenindex (row - nx)
        # ((j-1) - 1) * nx + i = (j-1) * nx - nx + i = row - nx 
        if j > 1 # kein undurchlässiger Rand
            inv_density_face = T(1.0) # Platzhalter zum initialen Erzeugen der Matrix
            coeff = inv_density_face / dy^2

            push!(colVal, linear_indices(i, j - 1, nx_pressure))
            push!(nzVal, coeff)
            diag_val -= coeff

            row_length += 1
        end # undurchlässiger Rand, d.h. kein Fluss wegen No-slip-Bedingung (Neumann)

        # Fluss nach links (i-1), zweitkleinster Spaltenindex (row - 1)
        # (j - 1) * nx + (i-1) = row - 1
        if i > 1 # kein undurchlässiger Rand
            inv_density_face = T(1.0) # Platzhalter zum initialen Erzeugen der Matrix
            coeff = inv_density_face / dx^2

            push!(colVal, linear_indices(i - 1, j, nx_pressure))
            push!(nzVal, coeff)
            diag_val -= coeff

            row_length += 1
        end # undurchlässiger Rand, d.h. kein Fluss wegen No-Slip-Bedingung (Neumann)

        # Diagonale (eigene Zelle), mittlerer Spaltenindex (row)
        diag_index = length(colVal) + 1
        push!(colVal, row)
        push!(nzVal, diag_val)

        row_length += 1

        # Fluss nach rechts (i+1), zweitgrößter Spaltenindex (row + 1)
        # (j - 1) * nx + (i+1) = row + 1
        if i < nx_pressure # kein undurchlässiger Rand
            inv_density_face = T(1.0) # Platzhalter zum initialen Erzeugen der Matrix
            coeff = inv_density_face / dx^2

            push!(colVal, linear_indices(i + 1, j, nx_pressure))
            push!(nzVal, coeff)
            nzVal[diag_index] -= coeff

            row_length += 1
        end # undurchlässiger Rand, d.h. kein Fluss wegen No-slip-Bedingung (Neumann)

        # Fluss nach oben (j+1), größter Spaltenindex (row + nx)
        # ((j+1) - 1) * nx + i = ((j-1) + 1) * nx + i = (j-1) * nx + nx + i = row + nx
        inv_density_face = T(1.0) # Platzhalter zum initialen Erzeugen der Matrix
        coeff = inv_density_face / dy^2

        if j < ny_pressure # kein Ausströmungsbereich
            push!(colVal, linear_indices(i, j + 1, nx_pressure))
            push!(nzVal, coeff)

            row_length += 1
        end # Ausströmungsbereich, d.h. kein Eintrag wegen Ausströmungsbedingung (Dirichlet)

        # Fluss findet immer statt
        nzVal[diag_index] -= coeff

        push!(rowPtr, rowPtr[end] + row_length)
    end

    return CuSparseMatrixCSR(
        CuVector(rowPtr),
        CuVector(colVal),
        CuVector(nzVal),
        (nx_pressure * ny_pressure, nx_pressure * ny_pressure))
end

function update_coefficient_mat_kernel!(rowPtr::CuDeviceVector{Int32}, nzVal::CuDeviceVector{T}, c_advected::CuDeviceMatrix{T}, density_1::T, density_2::T, dx::T, dy::T, nx_pressure, ny_pressure) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i <= nx_pressure && j <= ny_pressure
        # Indizes linearisieren und Startindex in nzVal finden
        row = linear_indices(i, j, nx_pressure)
        index = Int(rowPtr[row])

        diag_val = T(0.0)

        # Stelle durch die richtige Reihenfolge sicher, dass die Spaltenindizes innerhalb jeder Zeile strikt aufsteigend sortiert sind

        # Fluss nach unten (j-1), kleinster linearer Spaltenindex (row - nx)
        # ((j-1) - 1) * nx + i = (j-1) * nx - nx + i = row - nx 
        if j > 1 # kein undurchlässiger Rand
            i_, j_ = get_c_index(i, j)
            c_here = c_advected[i_, j_]
            c_neighbor = c_advected[i_, j_ - 1]

            inv_density_here = T(1.0) / (c_here * density_1 + (T(1.0) - c_here) * density_2)
            inv_density_neighbor = T(1.0) / (c_neighbor * density_1 + (T(1.0) - c_neighbor) * density_2)
            inv_density_face = T(0.5) * (inv_density_here + inv_density_neighbor)
            coeff = inv_density_face / dy^2

            nzVal[index] = coeff
            index += 1
            diag_val -= coeff
        end # undurchlässiger Rand, d.h. kein Fluss wegen No-slip-Bedingung (Neumann)

        # Fluss nach links (i-1), zweitkleinster Spaltenindex (row - 1)
        # (j - 1) * nx + (i-1) = row - 1
        if i > 1 # kein undurchlässiger Rand
            i_, j_ = get_c_index(i, j)
            c_here = c_advected[i_, j_]
            c_neighbor = c_advected[i_ - 1, j_]

            inv_density_here = T(1.0) / (c_here * density_1 + (T(1.0) - c_here) * density_2)
            inv_density_neighbor = T(1.0) / (c_neighbor * density_1 + (T(1.0) - c_neighbor) * density_2)
            inv_density_face = T(0.5) * (inv_density_here + inv_density_neighbor)
            coeff = inv_density_face / dx^2

            nzVal[index] = coeff
            index += 1
            diag_val -= coeff
        end # undurchlässiger Rand, d.h. kein Fluss wegen No-slip-Bedingung (Neumann)

        # Diagonale (eigene Zelle), mittlerer Spaltenindex (row)
        diag_index = index
        nzVal[diag_index] = diag_val
        index += 1

        # Fluss nach rechts (i+1), zweitgrößter Spaltenindex (row + 1)
        # (j - 1) * nx + (i+1) = row + 1
        if i < nx_pressure # kein undurchlässiger Rand
            i_, j_ = get_c_index(i, j)
            c_here = c_advected[i_, j_]
            c_neighbor = c_advected[i_ + 1, j_]

            inv_density_here = T(1.0) / (c_here * density_1 + (T(1.0) - c_here) * density_2)
            inv_density_neighbor = T(1.0) / (c_neighbor * density_1 + (T(1.0) - c_neighbor) * density_2)
            inv_density_face = T(0.5) * (inv_density_here + inv_density_neighbor)
            coeff = inv_density_face / dx^2

            nzVal[index] = coeff
            index += 1
            nzVal[diag_index] -= coeff
        end # undurchlässiger Rand, d.h. kein Fluss wegen No-slip-Bedingung (Neumann)

        # Fluss nach oben (j+1), größter Spaltenindex (row + nx)
        # ((j+1) - 1) * nx + i = ((j-1) + 1) * nx + i = (j-1) * nx + nx + i = row + nx
        i_, j_ = get_c_index(i, j)
        c_here = c_advected[i_, j_]
        c_neighbor = c_advected[i_, j_ + 1]

        inv_density_here = T(1.0) / (c_here * density_1 + (T(1.0) - c_here) * density_2)
        inv_density_neighbor = T(1.0) / (c_neighbor * density_1 + (T(1.0) - c_neighbor) * density_2)
        inv_density_face = T(0.5) * (inv_density_here + inv_density_neighbor)
        coeff = inv_density_face / dy^2

        if j < ny_pressure # kein Ausströmungsbereich
            nzVal[index] = coeff
            index += 1
        end # Ausströmungsbereich, d.h. kein Eintrag wegen Ausströmungsbedingung (Dirichlet)

        # Fluss findet immer statt
        nzVal[diag_index] -= coeff

        @cuassert index == rowPtr[row+1]
    end

    return nothing
end

function update_coefficient_mat!(param)
    kernel = @cuda launch = false update_coefficient_mat_kernel!(param.coefficient_mat.rowPtr, param.coefficient_mat.nzVal, param.c_advected, param.density_1, param.density_2, param.dx, param.dy, param.nx_pressure, param.ny_pressure)
    config = CUDA.launch_configuration(kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_pressure, threads_per_dim)
    x_blocks = cld(param.nx_pressure, x_threads)

    y_threads = min(param.ny_pressure, threads_per_dim)
    y_blocks = cld(param.ny_pressure, y_threads)

    kernel(param.coefficient_mat.rowPtr, param.coefficient_mat.nzVal, param.c_advected, param.density_1, param.density_2, param.dx, param.dy, param.nx_pressure, param.ny_pressure, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

function calculate_divergence_kernel!(divergence::CuDeviceMatrix{T}, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, time_step::T, dx::T, dy::T, nx_pressure, ny_pressure) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i <= nx_pressure && j <= ny_pressure
        i_, j_ = get_right_vx_index(i, j)
        vx_right = vx[i_, j_]
        i_, j_ = get_left_vx_index(i, j)
        vx_left = vx[i_, j_]

        i_, j_ = get_top_vy_index(i, j)
        vy_top = vy[i_, j_]
        i_, j_ = get_bottom_vy_index(i, j)
        vy_bottom = vy[i_, j_]

        value = (vx_right - vx_left) / dx + (vy_top - vy_bottom) / dy
        value /= time_step
        divergence[i, j] = value
    end

    return nothing
end

function calculate_divergence!(vx::CuMatrix{T}, vy::CuMatrix{T}, time_step::T, param) where T
    kernel = @cuda launch = false calculate_divergence_kernel!(param.divergence, vx, vy, time_step, param.dx, param.dy, param.nx_pressure, param.ny_pressure)
    config = CUDA.launch_configuration(kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_pressure, threads_per_dim)
    x_blocks = cld(param.nx_pressure, x_threads)

    y_threads = min(param.ny_pressure, threads_per_dim)
    y_blocks = cld(param.ny_pressure, y_threads)

    kernel(param.divergence, vx, vy, time_step, param.dx, param.dy, param.nx_pressure, param.ny_pressure, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

function calculate_pressure!(vx::CuMatrix{T}, vy::CuMatrix{T}, time_step::T, param) where T
    calculate_divergence!(vx, vy, time_step, param)

    update_coefficient_mat!(param)

    x = vec(param.pressure)
    b = vec(param.divergence)
    cudss("solve", param.lin_solve, x, b)
    cudss("refactorization", param.lin_solve, x, b)
end

function update_vx_kernel!(vx::CuDeviceMatrix{T}, pressure::CuDeviceMatrix{T}, c_advected::CuDeviceMatrix{T}, time_step::T, density_1::T, density_2::T, dx::T, nx_vx, ny_vx) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    if 3 <= i <= nx_vx - 2 && 3 <= j <= ny_vx - 2
        i_, j_ = get_right_pressure_index(i, j)
        pressure_right = pressure[i_, j_]

        i_, j_ = get_left_pressure_index(i, j)
        pressure_left = pressure[i_, j_]

        i_, j_ = get_right_c_index(i, j)
        c_right = c_advected[i_, j_]

        i_, j_ = get_left_c_index(i, j)
        c_left = c_advected[i_, j_]

        inv_density_right = T(1.0) / (c_right * density_1 + (T(1.0) - c_right) * density_2)
        inv_density_left = T(1.0) / (c_left * density_1 + (T(1.0) - c_left) * density_2)
        inv_density_face = T(0.5) * (inv_density_right + inv_density_left)

        vx[i, j] -= time_step * inv_density_face * (pressure_right - pressure_left) / dx
    end

    return nothing
end

function update_vx!(vx::CuMatrix{T}, time_step::T, param) where T
    kernel = @cuda launch = false update_vx_kernel!(vx, param.pressure, param.c_advected, time_step, param.density_1, param.density_2, param.dx, param.nx_vx, param.ny_vx)
    config = CUDA.launch_configuration(kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_vx, threads_per_dim)
    x_blocks = cld(param.nx_vx, x_threads)

    y_threads = min(param.ny_vx, threads_per_dim)
    y_blocks = cld(param.ny_vx, y_threads)

    kernel(vx, param.pressure, param.c_advected, time_step, param.density_1, param.density_2, param.dx, param.nx_vx, param.ny_vx, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

function update_vy_kernel!(vy::CuDeviceMatrix{T}, pressure::CuDeviceMatrix{T}, c_advected::CuDeviceMatrix{T}, time_step::T, density_1::T, density_2::T, dy::T, nx_vy, ny_vy) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # nur innere Zellen, d.h. keine Rand- bzw. Ghost-Zellen
    if 3 <= i <= nx_vy - 2 && 3 <= j <= ny_vy - 2
        i_, j_ = get_top_pressure_index(i, j)
        pressure_top = pressure[i_, j_]

        i_, j_ = get_bottom_pressure_index(i, j)
        pressure_bottom = pressure[i_, j_]

        i_, j_ = get_top_c_index(i, j)
        c_top = c_advected[i_, j_]

        i_, j_ = get_bottom_c_index(i, j)
        c_bottom = c_advected[i_, j_]

        inv_density_top = T(1.0) / (c_top * density_1 + (T(1.0) - c_top) * density_2)
        inv_density_bottom = T(1.0) / (c_bottom * density_1 + (T(1.0) - c_bottom) * density_2)
        inv_density_face = T(0.5) * (inv_density_top + inv_density_bottom)

        vy[i, j] -= time_step * inv_density_face * (pressure_top - pressure_bottom) / dy
    end

    return nothing
end

function update_vy!(vy::CuMatrix{T}, time_step::T, param) where T
    kernel = @cuda launch = false update_vy_kernel!(vy, param.pressure, param.c_advected, time_step, param.density_1, param.density_2, param.dy, param.nx_vy, param.ny_vy)
    config = CUDA.launch_configuration(kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_vy, threads_per_dim)
    x_blocks = cld(param.nx_vy, x_threads)

    y_threads = min(param.ny_vy, threads_per_dim)
    y_blocks = cld(param.ny_vy, y_threads)

    kernel(vy, param.pressure, param.c_advected, time_step, param.density_1, param.density_2, param.dy, param.nx_vy, param.ny_vy, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

function update_velocities_cb!(navier_stokes_integrator)
    u = navier_stokes_integrator.u
    param = navier_stokes_integrator.p
    time_step = navier_stokes_integrator.t - navier_stokes_integrator.tprev

    vx_length = param.nx_vx * param.ny_vx
    vy_length = param.nx_vy * param.ny_vy

    vx = reshape(view(u, 1:vx_length), param.nx_vx, param.ny_vx)
    vy = reshape(view(u, vx_length+1:vx_length+vy_length), param.nx_vy, param.ny_vy)

    apply_boundary_conditions_vx!(vx, param)
    apply_boundary_conditions_vy!(vy, param)

    calculate_pressure!(vx, vy, time_step, param)

    update_vx!(vx, time_step, param)
    update_vy!(vy, time_step, param)

    apply_boundary_conditions_vx!(vx, param)
    apply_boundary_conditions_vy!(vy, param)

    param.vx_corrected .= vx
    param.vy_corrected .= vy
end