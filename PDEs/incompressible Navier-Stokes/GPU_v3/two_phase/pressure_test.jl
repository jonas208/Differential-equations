using SparseArrays
using CUDA
using CUDA.CUSPARSE
using CUDSS
using GLMakie

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

function get_coefficient_mat_cpu_2(divergence_mat::Matrix{T}, nx, ny, dx, dy, is_solid::BitMatrix, rho::Matrix{T}) where T
    linear_indices = LinearIndices(divergence_mat)
    I = Int[]
    J = Int[]
    X = T[]

    for j in 1:ny, i in 1:nx
        row = linear_indices[i, j]

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
                push!(I, row)
                push!(J, row)
                push!(X, T(1.0))
                continue
            end
        end

        # Ausströmungsbedingung (Dirichlet)
        if i >= nx
            push!(I, row)
            push!(J, row)
            push!(X, T(1.0))
            continue
        end

        diag_val = T(0.0)

        # Fluss nach links (i-1)
        if i > 1 && !is_solid[i-1, j] # kein fester Rand
            # Inverse des harmonischen Mittels
            inv_rho_face = T(0.5) * (T(1.0) / rho[i, j] + T(1.0) / rho[i-1, j])
            coeff = inv_rho_face * T(1.0) / (dx^2)

            push!(I, row)
            push!(J, linear_indices[i-1, j])
            push!(X, coeff)
            diag_val -= coeff
        end # fester Rand, d.h. kein Fluss wegen No-slip-Bedingung (Neumann)

        # Fluss nach rechts (i+1)
        if i < nx && !is_solid[i+1, j]
            inv_rho_face = T(0.5) * (T(1.0) / rho[i, j] + T(1.0) / rho[i+1, j])
            coeff = inv_rho_face * T(1.0) / (dx^2)

            push!(I, row)
            push!(J, linear_indices[i+1, j])
            push!(X, coeff)
            diag_val -= coeff
        end

        # Fluss nach unten (j-1)
        if j > 1 && !is_solid[i, j-1]
            inv_rho_face = T(0.5) * (T(1.0) / rho[i, j] + T(1.0) / rho[i, j-1])
            coeff = inv_rho_face * T(1.0) / (dy^2)

            push!(I, row)
            push!(J, linear_indices[i, j-1])
            push!(X, coeff)
            diag_val -= coeff
        end

        # Fluss nach oben (j+1)
        if j < ny && !is_solid[i, j+1]
            inv_rho_face = T(0.5) * (T(1.0) / rho[i, j] + T(1.0) / rho[i, j+1])
            coeff = inv_rho_face * T(1.0) / (dy^2)

            push!(I, row)
            push!(J, linear_indices[i, j+1])
            push!(X, coeff)
            diag_val -= coeff
        end

        push!(I, row)
        push!(J, row)
        push!(X, diag_val)
    end

    return sparse(I, J, X, nx * ny, nx * ny)
end

function get_coefficient_mat_cpu_3(nx, ny, dx, dy, is_solid::BitMatrix, rho::AbstractMatrix{T}) where T
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

function update_coefficient_mat_kernel!(rowPtr::CuDeviceVector{Int32}, colVal::CuDeviceVector{Int32}, nzVal::CuDeviceVector{T}, is_solid::CuDeviceMatrix{Bool}, c::CuDeviceMatrix{T}, rho_1::T, rho_2::T, nx, ny, dx::T, dy::T) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if 1 <= i <= nx && 1 <= j <= ny
        # Indizes linearisieren und Startindex in nzVal finden
        row = (j - 1) * nx + i
        index = Int(rowPtr[row])

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
                nzVal[index] = T(1.0)
                return nothing
            end
        end

        # Ausströmungsbedingung (Dirichlet)
        if i >= nx # Dass es nur eine Zelle sein sollen, ändere ich nachher wieder!
            nzVal[index] = T(1.0)
            return nothing
        end

        diag_val = T(0.0)

        # Stelle durch die richtige Reihenfolge sicher, dass die Spaltenindizes innerhalb jeder Zeile strikt aufsteigend sortiert sind

        # Fluss nach unten (j-1), kleinster linearer Spaltenindex (row - nx)
        # ((j-1) - 1) * nx + i = (j-1) * nx - nx + i = row - nx 
        if j > 1 && !is_solid[i, j-1]
            rho_here = c[i, j] * rho_1 + (T(1.0) - c[i, j]) * rho_2
            rho_neighbor = c[i, j-1] * rho_1 + (T(1.0) - c[i, j-1]) * rho_2

            # Inverse des harmonischen Mittels
            inv_rho_face = T(0.5) * (T(1.0) / rho_here + T(1.0) / rho_neighbor)
            coeff = inv_rho_face * T(1.0) / (dy^2)

            nzVal[index] = coeff
            index += 1
            diag_val -= coeff
        end # fester Rand, d.h. kein Fluss wegen No-slip-Bedingung (Neumann)

        # Fluss nach links (i-1), zweitkleinster Spaltenindex (row - 1)
        # (j - 1) * nx + (i-1) = row - 1
        if i > 1 && !is_solid[i-1, j]
            rho_here = c[i, j] * rho_1 + (T(1.0) - c[i, j]) * rho_2
            rho_neighbor = c[i-1, j] * rho_1 + (T(1.0) - c[i-1, j]) * rho_2

            inv_rho_face = T(0.5) * (T(1.0) / rho_here + T(1.0) / rho_neighbor)
            coeff = inv_rho_face * T(1.0) / (dx^2)

            nzVal[index] = coeff
            index += 1
            diag_val -= coeff
        end

        # Diagonale (eigene Zelle), mittlerer Spaltenindex (row)
        diag_index = index
        nzVal[diag_index] = diag_val
        index += 1

        # Fluss nach rechts (i+1), zweitgrößter Spaltenindex (row + 1)
        # (j - 1) * nx + (i+1) = row + 1
        if i < nx && !is_solid[i+1, j]
            rho_here = c[i, j] * rho_1 + (T(1.0) - c[i, j]) * rho_2
            rho_neighbor = c[i+1, j] * rho_1 + (T(1.0) - c[i+1, j]) * rho_2

            inv_rho_face = T(0.5) * (T(1.0) / rho_here + T(1.0) / rho_neighbor)
            coeff = inv_rho_face * T(1.0) / (dx^2)

            nzVal[index] = coeff
            index += 1
            nzVal[diag_index] -= coeff
        end

        # Fluss nach oben (j+1), größter Spaltenindex (row + nx)
        # ((j+1) - 1) * nx + i = ((j-1) + 1) * nx + i = (j-1) * nx + nx + i = row + nx
        if j < ny && !is_solid[i, j+1]
            rho_here = c[i, j] * rho_1 + (T(1.0) - c[i, j]) * rho_2
            rho_neighbor = c[i, j+1] * rho_1 + (T(1.0) - c[i, j+1]) * rho_2

            inv_rho_face = T(0.5) * (T(1.0) / rho_here + T(1.0) / rho_neighbor)
            coeff = inv_rho_face * T(1.0) / (dy^2)

            nzVal[index] = coeff
            index += 1
            nzVal[diag_index] -= coeff
        end

        # @cuassert index == rowPtr[row+1]
    end

    return nothing
end

function update_coefficient_mat!(coefficient_mat::CuSparseMatrixCSR{T,Int32}, is_solid::CuMatrix{Bool}, c::CuMatrix{T}, rho_1::T, rho_2::T, nx, ny, dx::T, dy::T) where T
    kernel = @cuda launch = false update_coefficient_mat_kernel!(coefficient_mat.rowPtr, coefficient_mat.colVal, coefficient_mat.nzVal, is_solid, c, rho_1, rho_2, nx, ny, dx, dy)
    config = CUDA.launch_configuration(kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(nx, threads_per_dim)
    x_blocks = cld(nx, x_threads)

    y_threads = min(ny, threads_per_dim)
    y_blocks = cld(ny, y_threads)

    kernel(coefficient_mat.rowPtr, coefficient_mat.colVal, coefficient_mat.nzVal, is_solid, c, rho_1, rho_2, nx, ny, dx, dy, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

type = Float32
Lx = type(12.5) # Seitenlänge in x-Richtung der rechteckigen Oberfläche [m]
Ly = type(3.0) # Seitenlänge in y-Richtung der rechteckigen Oberfläche [m]

nx = 2500
ny = 600
dx = Lx / nx
dy = Ly / ny

using Images
img = load("obstacles/A5_mask_medium_long.png")
img = imrotate(img, π / 2)
img = collect(img)
mask = BitMatrix(alpha.(img) .== 1)
# heatmap(mask)

is_solid_cpu = mask
is_solid_gpu = CuArray(is_solid_cpu)

# pressure_vec_gpu = CUDA.zeros(type, nx*ny)
pressure_vec_gpu = CUDA.rand(type, nx * ny)
# divergence_mat_gpu = CUDA.zeros(type, nx, ny)
divergence_mat_gpu = CUDA.rand(type, nx, ny)

divergence_mat_gpu[1:2, :] .= type(0.0)
divergence_mat_gpu[end-1:end, :] .= type(0.0)
divergence_mat_gpu[:, 1:2] .= type(0.0)
divergence_mat_gpu[:, end-1:end] .= type(0.0)
divergence_mat_gpu .*= .!is_solid_gpu

# coefficient_mat_cpu = get_coefficient_mat_cpu(zeros(type, nx, ny), nx, ny, dx, dy, is_solid_cpu)
coefficient_mat_cpu = get_coefficient_mat_cpu_2(zeros(type, nx, ny), nx, ny, dx, dy, is_solid_cpu, ones(type, nx, ny))

coefficient_mat_gpu = CuSparseMatrixCSR(coefficient_mat_cpu)

rowPtr, colVal, nzVal = get_coefficient_mat_cpu_3(nx, ny, dx, dy, is_solid_cpu, ones(type, nx, ny))
coefficient_mat_gpu_new = CuSparseMatrixCSR(CuVector(rowPtr), CuVector(colVal), CuVector(nzVal), (nx * ny, nx * ny))

@info isapprox(coefficient_mat_gpu, coefficient_mat_gpu_new)
coefficient_mat_gpu = coefficient_mat_gpu_new

# heatmap(Matrix(divergence_mat_gpu))

typeof(coefficient_mat_gpu.rowPtr);
length(coefficient_mat_gpu.rowPtr);
typeof(coefficient_mat_gpu.colVal);
length(coefficient_mat_gpu.colVal);
typeof(coefficient_mat_gpu.nzVal);
length(coefficient_mat_gpu.nzVal);
coefficient_mat_gpu.dims
coefficient_mat_gpu.nnz

# Definiere ein lineares Gleichungssystem der Form Ax = b
A_gpu = coefficient_mat_gpu
x_gpu = pressure_vec_gpu
b_gpu = vec(divergence_mat_gpu)
lin_solve_gpu = CudssSolver(A_gpu, "G", 'F')
cudss_set(lin_solve_gpu, "ir_n_steps", 1)
cudss("analysis", lin_solve_gpu, x_gpu, b_gpu)
cudss("factorization", lin_solve_gpu, x_gpu, b_gpu)
# cudss("refactorization", lin_solve_gpu, x_gpu, b_gpu)
cudss("solve", lin_solve_gpu, x_gpu, b_gpu)
pressure_gpu = reshape(x_gpu, nx, ny)

heatmap(Matrix(pressure_gpu))

c = CUDA.rand(type, nx, ny)
rho_1 = type(1.0)
# rho_2 = type(1000.0)
rho_2 = type(1.0)

coefficient_mat_gpu_new = copy(coefficient_mat_gpu)
update_coefficient_mat!(coefficient_mat_gpu_new, is_solid_gpu, c, rho_1, rho_2, nx, ny, dx, dy)
@info isapprox(coefficient_mat_gpu, coefficient_mat_gpu_new)

cudss("refactorization", lin_solve_gpu, x_gpu, b_gpu)
cudss("solve", lin_solve_gpu, x_gpu, b_gpu)
pressure_gpu = reshape(x_gpu, nx, ny)

heatmap(Matrix(pressure_gpu))

#=
# pressure_1 = copy(Matrix(pressure_gpu))
# pressure_2 = copy(Matrix(pressure_gpu))

heatmap(pressure_1)
heatmap(pressure_2)

heatmap(pressure_1 .- pressure_2)
heatmap(.!is_solid_cpu .* (pressure_1 .- pressure_2))

minimum(pressure_1 .- pressure_2)
maximum(pressure_1 .- pressure_2)
sum(abs.(pressure_1 .- pressure_2)) / (nx * ny)
=#

#=
c = CUDA.rand(type, nx, ny)
rho_1 = type(1.0)
# rho_2 = type(1000.0)
rho_2 = type(1.0)

Lx = type(1.0) # Seitenlänge in x-Richtung der rechteckigen Oberfläche [m]
Ly = type(1.0) # Seitenlänge in y-Richtung der rechteckigen Oberfläche [m]

nx = 3
ny = 3
dx = Lx / nx
dy = Ly / ny

I, J, X = get_coefficient_mat_cpu_2(zeros(type, nx, ny), nx, ny, dx, dy, is_solid_cpu, ones(type, nx, ny))

mat = CuSparseMatrixCOO(CuVector(I), CuVector(J), CuVector(X), (nx * ny, nx * ny))

coefficient_mat_gpu = CuSparseMatrixCSR(mat)

rowPtr, colVal, nzVal = get_coefficient_mat_cpu_3(zeros(type, nx, ny), nx, ny, dx, dy, is_solid_cpu, ones(type, nx, ny))
coefficient_mat_gpu_3 = CuSparseMatrixCSR(CuVector(rowPtr), CuVector(colVal), CuVector(nzVal), (nx * ny, nx * ny))

@info isapprox(coefficient_mat_gpu, coefficient_mat_gpu_3)

coefficient_mat_gpu_new = copy(coefficient_mat_gpu_3)
update_coefficient_mat!(coefficient_mat_gpu_new, is_solid_gpu, c, rho_1, rho_2, nx, ny, dx, dy)
coefficient_mat_gpu_new

isapprox(coefficient_mat_gpu, coefficient_mat_gpu_new)

sum(coefficient_mat_gpu)
sum(coefficient_mat_gpu_new)

for j in 1:ny, i in 1:nx
    @info LinearIndices((nx, ny))[i, j]
end
=#