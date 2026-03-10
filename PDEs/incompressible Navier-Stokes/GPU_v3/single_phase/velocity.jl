function apply_boundary_conditions!(vx::AbstractMatrix{T}, vy::AbstractMatrix{T}, c::AbstractMatrix{T}, horizontal_velocity::T, is_solid::BitMatrix) where T
    # Dirichlet am unteren Rand
    vx[:, 1] .= T(0.0)
    vx[:, 2] .= T(0.0)
    vy[:, 1] .= T(0.0)
    vy[:, 2] .= T(0.0)

    # Dirichlet am oberen Rand
    vx[:, end-1] .= T(0.0)
    vx[:, end] .= T(0.0)
    vy[:, end-1] .= T(0.0)
    vy[:, end] .= T(0.0)

    # Dirichlet am linken Rand

    vx[1, :] .= horizontal_velocity
    vx[2, :] .= horizontal_velocity

    vy[1, :] .= T(0.0)
    vy[2, :] .= T(0.0)
    
    # Neumann am rechten Rand
    vx[end-1, :] .= vx[end-2, :]
    vx[end, :] .= vx[end-1, :]
    vy[end-1, :] .= vy[end-2, :]
    vy[end, :] .= vy[end-1, :]

    # Dirichlet für die Hindernisse
    for j in 1:ny, i in 1:nx
        if is_solid[i, j]
            vx[i, j] = T(0.0)
            vy[i, j] = T(0.0)
        end
    end

    # Randbedingungen für den Rauch

    #=
    # Neumann am unteren Rand
    c[:, 2] .= c[:, 3]
    c[:, 1] .= c[:, 2]

    # Neumann am oberen Rand
    c[:, end-1] .= c[:, end-2]
    c[:, end] .= c[:, end-1]
    =#

    # Dirichlet am unteren Rand
    c[:, 2] .= T(0.0)
    c[:, 1] .= T(0.0)

    # Dirichlet am oberen Rand
    c[:, end-1] .= T(0.0)
    c[:, end] .= T(0.0)
    
    # Dirichlet am linken Rand
    c[1, :] .= T(0.0)
    c[2, :] .= T(0.0)

    #=
    # Neumann am rechten Rand
    c[end-1, :] .= c[end-2, :]
    c[end, :] .= c[end-1, :]
    =#

    # Dirichlet am rechten Rand
    c[end-1, :] .= T(0.0)
    c[end, :] .= T(0.0)
end

function apply_boundary_conditions_kernel!(vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, c::CuDeviceMatrix{T}, horizontal_velocity::T) where T
    k = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nx, ny = size(vx)

    if k <= nx
        # Dirichlet am unteren Rand
        vx[k, 1] = T(0.0)
        vx[k, 2] = T(0.0)
        vy[k, 1] = T(0.0)
        vy[k, 2] = T(0.0)

        # Dirichlet am oberen Rand
        vx[k, end-1] = T(0.0)
        vx[k, end] = T(0.0)
        vy[k, end-1] = T(0.0)
        vy[k, end] = T(0.0)

        # Randbedingungen für den Rauch

        #=
        # Neumann am unteren Rand
        c[k, 2] = c[k, 3]
        c[k, 1] = c[k, 2]

        # Neumann am oberen Rand
        c[k, end-1] = c[k, end-2]
        c[k, end] = c[k, end-1]
        =#

        # Dirichlet am unteren Rand
        c[k, 2] = T(0.0)
        c[k, 1] = T(0.0)

        # Dirichlet am oberen Rand
        c[k, end-1] = T(0.0)
        c[k, end] = T(0.0)
    end

    if k <= ny
        # Dirichlet am linken Rand

        vx[1, k] = horizontal_velocity
        vx[2, k] = horizontal_velocity

        vy[1, k] = T(0.0)
        vy[2, k] = T(0.0)
        
        # Neumann am rechten Rand
        vy[end-1, k] = vy[end-2, k]
        vy[end, k] = vy[end-1, k]
        vx[end-1, k] = vx[end-2, k]
        vx[end, k] = vx[end-1, k]

        # Randbedingungen für den Rauch

        # Dirichlet am linken Rand
        c[1, k] = T(0.0)
        c[2, k] = T(0.0)
        if 1 < k < 600 && k % 100 == 0
            c[1, k-5] = T(1.0)
            c[1, k-4] = T(1.0)
            c[1, k-3] = T(1.0)
            c[1, k-2] = T(1.0)
            c[1, k-1] = T(1.0)
            c[1, k] = T(1.0)
            c[1, k+1] = T(1.0)
            c[1, k+2] = T(1.0)
            c[1, k+3] = T(1.0)
            c[1, k+4] = T(1.0)
            c[1, k+5] = T(1.0)

            c[2, k-5] = T(1.0)
            c[2, k-4] = T(1.0)
            c[2, k-3] = T(1.0)
            c[2, k-2] = T(1.0)
            c[2, k-1] = T(1.0)
            c[2, k] = T(1.0)
            c[2, k+1] = T(1.0)
            c[2, k+2] = T(1.0)
            c[2, k+3] = T(1.0)
            c[2, k+4] = T(1.0)
            c[2, k+5] = T(1.0)
        end

        #=
        # Neumann am rechten Rand
        c[end-1, k] = c[end-2, k]
        c[end, k] = c[end-1, k]
        =#

        # Dirichlet am rechten Rand
        c[end-1, k] = T(0.0)
        c[end, k] = T(0.0)
    end

    return nothing
end

function apply_boundary_conditions!(vx::CuMatrix{T}, vy::CuMatrix{T}, c::CuMatrix{T}, horizontal_velocity::T, is_solid::CuMatrix{Bool}) where T
    nx, ny = size(vx)
    
    kernel = @cuda launch=false apply_boundary_conditions_kernel!(vx, vy, c, horizontal_velocity)
    config = CUDA.launch_configuration(kernel.fun)

    threads = config.threads

    n_max = max(nx, ny)
    x_threads = min(n_max, threads)
    x_blocks = cld(n_max, x_threads)

    kernel(vx, vy, c, horizontal_velocity, threads = x_threads, blocks = x_blocks)

    # Dirichlet für die Hindernisse  
    vx .*= .!is_solid
    vy .*= .!is_solid
end

# 1D-CWENO-Interpolation in x-Richtung (P_{i}) mit zusätzlichem Term im Teilpolynom pC
# u_is = [u_{i-1,j}, u_{i,j}, u_{i+1,j}]
# u_js = [u_{i,j-1}, u_{i,j}, u_{i,j+1}]
function cweno_x(x::T, x_i::T, dx::T, u_is::AbstractVector{T}, u_js::AbstractVector{T}, ep::T, p::T) where T <: AbstractFloat
    uL = u_is[2]-u_is[1]; uC = u_is[3]-2*u_is[2]+u_is[1]; uR = u_is[3]-u_is[2]; uCC = u_is[3]-u_is[1];
    ISL = uL^2; ISC = T(13/3)*uC^2 + T(0.25)*uCC^2; ISR = uR^2;
    aL = T(0.25)*(1/(ep+ISL))^p; aC = T(0.5)*(1/(ep+ISC))^p; aR = T(0.25)*(1/(ep+ISR))^p;
    suma = max(aL+aC+aR,eps(T(1.0))); 
    wL = aL/suma; wC = aC/suma; wR = aR/suma;
    pL = u_is[2] + uL/dx*(x-x_i);
    pC = u_is[2] - uC/12 - (u_js[3]-2*u_js[2]+u_js[1])/12 + uCC/(2*dx)*(x-x_i) + uC/dx^2*(x-x_i)^2;
    pR = u_is[2] + uR/dx*(x-x_i);
    return wL*pL + wC*pC + wR*pR
end

# 1D-CWENO-Interpolation in y-Richtung (P_{j}) mit zusätzlichem Term im Teilpolynom pC
# u_js = [u_{i,j-1}, u_{i,j}, u_{i,j+1}]
# u_is = [u_{i-1,j}, u_{i,j}, u_{i+1,j}]
function cweno_y(y::T, y_j::T, dy::T, u_js::AbstractVector{T}, u_is::AbstractVector{T}, ep::T, p::T) where T <: AbstractFloat
    uL = u_js[2]-u_js[1]; uC = u_js[3]-2*u_js[2]+u_js[1]; uR = u_js[3]-u_js[2]; uCC = u_js[3]-u_js[1];
    ISL = uL^2; ISC = T(13/3)*uC^2 + T(0.25)*uCC^2; ISR = uR^2;
    aL = T(0.25)*(1/(ep+ISL))^p; aC = T(0.5)*(1/(ep+ISC))^p; aR = T(0.25)*(1/(ep+ISR))^p;
    suma = max(aL+aC+aR,eps(T(1.0))); 
    wL = aL/suma; wC = aC/suma; wR = aR/suma;
    pL = u_js[2] + uL/dy*(y-y_j);
    pC = u_js[2] - uC/12 - (u_is[3]-2*u_is[2]+u_is[1])/12 + uCC/(2*dy)*(y-y_j) + uC/dy^2*(y-y_j)^2;
    pR = u_js[2] + uR/dy*(y-y_j);
    return wL*pL + wC*pC + wR*pR
end

# 1D-CWENO-Interpolation in x-Richtung (P_{i}) mit zusätzlichem Term im Teilpolynom pC
# u_is = [u_{i-1,j}, u_{i,j}, u_{i+1,j}]
# u_js = [u_{i,j-1}, u_{i,j}, u_{i,j+1}]
function cweno_x(x::T, x_i::T, dx::T, u_is1::T, u_is2::T, u_is3::T, u_js1::T, u_js2::T, u_js3::T, ep::T, p::T) where T <: AbstractFloat
    uL = u_is2-u_is1; uC = u_is3-2*u_is2+u_is1; uR = u_is3-u_is2; uCC = u_is3-u_is1;
    ISL = uL^2; ISC = T(13/3)*uC^2 + T(0.25)*uCC^2; ISR = uR^2;
    aL = T(0.25)*(1/(ep+ISL))^p; aC = T(0.5)*(1/(ep+ISC))^p; aR = T(0.25)*(1/(ep+ISR))^p;
    suma = max(aL+aC+aR,eps(T(1.0))); 
    wL = aL/suma; wC = aC/suma; wR = aR/suma;
    pL = u_is2 + uL/dx*(x-x_i);
    pC = u_is2 - uC/12 - (u_js3-2*u_js2+u_js1)/12 + uCC/(2*dx)*(x-x_i) + uC/dx^2*(x-x_i)^2;
    pR = u_is2 + uR/dx*(x-x_i);
    return wL*pL + wC*pC + wR*pR
end

# 1D-CWENO-Interpolation in y-Richtung (P_{j}) mit zusätzlichem Term im Teilpolynom pC
# u_js = [u_{i,j-1}, u_{i,j}, u_{i,j+1}]
# u_is = [u_{i-1,j}, u_{i,j}, u_{i+1,j}]
function cweno_y(y::T, y_j::T, dy::T, u_js1::T, u_js2::T, u_js3::T, u_is1::T, u_is2::T, u_is3::T, ep::T, p::T) where T <: AbstractFloat
    uL = u_js2-u_js1; uC = u_js3-2*u_js2+u_js1; uR = u_js3-u_js2; uCC = u_js3-u_js1;
    ISL = uL^2; ISC = T(13/3)*uC^2 + T(0.25)*uCC^2; ISR = uR^2;
    aL = T(0.25)*(1/(ep+ISL))^p; aC = T(0.5)*(1/(ep+ISC))^p; aR = T(0.25)*(1/(ep+ISR))^p;
    suma = max(aL+aC+aR,eps(T(1.0))); 
    wL = aL/suma; wC = aC/suma; wR = aR/suma;
    pL = u_js2 + uL/dy*(y-y_j);
    pC = u_js2 - uC/12 - (u_is3-2*u_is2+u_is1)/12 + uCC/(2*dy)*(y-y_j) + uC/dy^2*(y-y_j)^2;
    pR = u_js2 + uR/dy*(y-y_j);
    return wL*pL + wC*pC + wR*pR
end

# berechne u^{\pm}_{i+1/2,j} wahlweise mit oder ohne WENO
function recover_x(i, j, dx::T, vx::AbstractMatrix{T}, vy::AbstractMatrix{T}, c::AbstractMatrix{T}, simulate_smoke, use_weno, ep::T, p::T) where T
    if use_weno
        vx_l = cweno_x(i*dx, (i-T(0.5))*dx, dx, view(vx, i-1:i+1, j), view(vx, i, j-1:j+1), ep, p) # P_i(x_{i+1/2})
        vx_r = cweno_x(i*dx, (i+T(0.5))*dx, dx, view(vx, i:i+2, j), view(vx, i+1, j-1:j+1), ep, p) # P_{i+1}(x_{i+1/2})
        vy_l = cweno_x(i*dx, (i-T(0.5))*dx, dx, view(vy, i-1:i+1, j), view(vy, i, j-1:j+1), ep, p) # P_i(x_{i+1/2})
        vy_r = cweno_x(i*dx, (i+T(0.5))*dx, dx, view(vy, i:i+2, j), view(vy, i+1, j-1:j+1), ep, p) # P_{i+1}(x_{i+1/2})
        if simulate_smoke
            c_l = cweno_x(i*dx, (i-T(0.5))*dx, dx, view(c, i-1:i+1, j), view(c, i, j-1:j+1), ep, p) # P_i(x_{i+1/2})
            c_r = cweno_x(i*dx, (i+T(0.5))*dx, dx, view(c, i:i+2, j), view(c, i+1, j-1:j+1), ep, p) # P_{i+1}(x_{i+1/2})
        end
    else
        vx_l = vx[i, j]
        vy_l = vy[i, j]
        vx_r = vx[i+1, j]
        vy_r = vy[i+1, j]
        if simulate_smoke
            c_l = c[i, j]
            c_r = c[i+1, j]
        end
    end
    return vx_l, vx_r, vy_l, vy_r, c_l, c_r
end

# berechne u^{\pm}_{i,j+1/2} wahlweise mit oder ohne WENO
function recover_y(i, j, dy::T, vx::AbstractMatrix{T}, vy::AbstractMatrix{T}, c::AbstractMatrix{T}, simulate_smoke, use_weno, ep::T, p::T) where T
    if use_weno
        vx_l = cweno_y(j*dy, (j-T(0.5))*dy, dy, view(vx, i, j-1:j+1), view(vx, i-1:i+1, j), ep, p) # P_j(y_{j+1/2})
        vx_r = cweno_y(j*dy, (j+T(0.5))*dy, dy, view(vx, i, j:j+2), view(vx, i-1:i+1, j+1), ep, p) # P_{j+1}(y_{j+1/2})
        vy_l = cweno_y(j*dy, (j-T(0.5))*dy, dy, view(vy, i, j-1:j+1), view(vy, i-1:i+1, j), ep, p) # P_j(y_{j+1/2})
        vy_r = cweno_y(j*dy, (j+T(0.5))*dy, dy, view(vy, i, j:j+2), view(vy, i-1:i+1, j+1), ep, p) # P_{j+1}(y_{j+1/2})
        if simulate_smoke
            c_l = cweno_y(j*dy, (j-T(0.5))*dy, dy, view(c, i, j-1:j+1), view(c, i-1:i+1, j), ep, p) # P_j(y_{j+1/2})
            c_r = cweno_y(j*dy, (j+T(0.5))*dy, dy, view(c, i, j:j+2), view(c, i-1:i+1, j+1), ep, p) # P_{j+1}(y_{j+1/2})
        end
    else
        vx_l = vx[i, j]
        vy_l = vy[i, j]
        vx_r = vx[i, j+1]
        vy_r = vy[i, j+1]
        if simulate_smoke
           c_l = c[i, j]
           c_r = c[i, j+1]
        end
    end
    return vx_l, vx_r, vy_l, vy_r, c_l, c_r
end

# berechne u^{\pm}_{i+1/2,j} wahlweise mit oder ohne WENO
function recover_x(i, j, dx::T, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, c::CuDeviceMatrix{T}, simulate_smoke, use_weno, ep::T, p::T) where T
    if use_weno
        vx_l = cweno_x(i*dx, (i-T(0.5))*dx, dx, vx[i-1, j], vx[i, j], vx[i+1, j], vx[i, j-1], vx[i, j], vx[i, j+1], ep, p) # P_i(x_{i+1/2})
        vx_r = cweno_x(i*dx, (i+T(0.5))*dx, dx, vx[i, j], vx[i+1, j], vx[i+2, j], vx[i+1, j-1], vx[i+1, j], vx[i+1, j+1], ep, p) # P_{i+1}(x_{i+1/2})
        vy_l = cweno_x(i*dx, (i-T(0.5))*dx, dx, vy[i-1, j], vy[i, j], vy[i+1, j], vy[i, j-1], vy[i, j], vy[i, j+1], ep, p) # P_i(x_{i+1/2})
        vy_r = cweno_x(i*dx, (i+T(0.5))*dx, dx, vy[i, j], vy[i+1, j], vy[i+2, j], vy[i+1, j-1], vy[i+1, j], vy[i+1, j+1], ep, p) # P_{i+1}(x_{i+1/2})
        if simulate_smoke
            c_l = cweno_x(i*dx, (i-T(0.5))*dx, dx, c[i-1, j], c[i, j], c[i+1, j], c[i, j-1], c[i, j], c[i, j+1], ep, p) # P_i(x_{i+1/2})
            c_r = cweno_x(i*dx, (i+T(0.5))*dx, dx, c[i, j], c[i+1, j], c[i+2, j], c[i+1, j-1], c[i+1, j], c[i+1, j+1], ep, p) # P_{i+1}(x_{i+1/2})
        else
            c_l = T(0.0)
            c_r = T(0.0)
        end
    else
        vx_l = vx[i, j]
        vy_l = vy[i, j]
        vx_r = vx[i+1, j]
        vy_r = vy[i+1, j]
        if simulate_smoke
            c_l = c[i, j]
            c_r = c[i+1, j]
        else
            c_l = T(0.0)
            c_r = T(0.0)
        end
    end
    return vx_l, vx_r, vy_l, vy_r, c_l, c_r
end

# berechne u^{\pm}_{i,j+1/2} wahlweise mit oder ohne WENO
function recover_y(i, j, dy::T, vx::CuDeviceMatrix{T}, vy::CuDeviceMatrix{T}, c::CuDeviceMatrix{T}, simulate_smoke, use_weno, ep::T, p::T) where T
    if use_weno
        vx_l = cweno_y(j*dy, (j-T(0.5))*dy, dy, vx[i, j-1], vx[i, j], vx[i, j+1], vx[i-1, j], vx[i, j], vx[i+1, j], ep, p) # P_j(y_{j+1/2})
        vx_r = cweno_y(j*dy, (j+T(0.5))*dy, dy, vx[i, j], vx[i, j+1], vx[i, j+2], vx[i-1, j+1], vx[i, j+1], vx[i+1, j+1], ep, p) # P_{j+1}(y_{j+1/2})
        vy_l = cweno_y(j*dy, (j-T(0.5))*dy, dy, vy[i, j-1], vy[i, j], vy[i, j+1], vy[i-1, j], vy[i, j], vy[i+1, j], ep, p) # P_j(y_{j+1/2})
        vy_r = cweno_y(j*dy, (j+T(0.5))*dy, dy, vy[i, j], vy[i, j+1], vy[i, j+2], vy[i-1, j+1], vy[i, j+1], vy[i+1, j+1], ep, p) # P_{j+1}(y_{j+1/2})
        if simulate_smoke
            c_l = cweno_y(j*dy, (j-T(0.5))*dy, dy, c[i, j-1], c[i, j], c[i, j+1], c[i-1, j], c[i, j], c[i+1, j], ep, p) # P_j(y_{j+1/2})
            c_r = cweno_y(j*dy, (j+T(0.5))*dy, dy, c[i, j], c[i, j+1], c[i, j+2], c[i-1, j+1], c[i, j+1], c[i+1, j+1], ep, p) # P_{j+1}(y_{j+1/2})
        else
            c_l = T(0.0)
            c_r = T(0.0)
        end
    else
        vx_l = vx[i, j]
        vy_l = vy[i, j]
        vx_r = vx[i, j+1]
        vy_r = vy[i, j+1]
        if simulate_smoke
            c_l = c[i, j]
            c_r = c[i, j+1]
        else
            c_l = T(0.0)
            c_r = T(0.0)
        end
    end
    return vx_l, vx_r, vy_l, vy_r, c_l, c_r
end

flux_x_1(vx::T, vy::T) where T = vx^2
flux_x_2(vx::T, vy::T) where T = vx*vy

flux_y_1(vx::T, vy::T) where T = vx*vy
flux_y_2(vx::T, vy::T) where T = vy^2

flux_x_smoke(vx::T, c::T) where T = vx*c
flux_y_smoke(vy::T, c::T) where T = vy*c

# berechne den numerischen Fluss in x-Richtung (H^x_{i+1/2,j})
function local_lax_friedrichs_x(vx_l::T, vx_r::T, vy_l::T, vy_r::T, c_l::T, c_r::T) where T
    # alpha = max(abs(vx_l), abs(vx_r))
    alpha = 2*max(abs(vx_l), abs(vx_r))
    flux_x_1_r = T(0.5)*(flux_x_1(vx_l, vy_l) + flux_x_1(vx_r, vy_r) - alpha*(vx_r - vx_l))
    flux_x_2_r = T(0.5)*(flux_x_2(vx_l, vy_l) + flux_x_2(vx_r, vy_r) - alpha*(vy_r - vy_l))

    alpha_smoke = max(abs(vx_l), abs(vx_r))
    flux_x_smoke_r = T(0.5)*(flux_x_smoke(vx_l, c_l) + flux_x_smoke(vx_r, c_r) - alpha_smoke*(c_r - c_l))

    return flux_x_1_r, flux_x_2_r, flux_x_smoke_r
end

# berechne den numerischen Fluss in y-Richtung (H^y_{i,j+1/2})
function local_lax_friedrichs_y(vx_l::T, vx_r::T, vy_l::T, vy_r::T, c_l::T, c_r::T) where T
    # alpha = max(abs(vy_l), abs(vy_r))
    alpha = 2*max(abs(vy_l), abs(vy_r))
    flux_y_1_r = T(0.5)*(flux_y_1(vx_l, vy_l) + flux_y_1(vx_r, vy_r) - alpha*(vx_r - vx_l))
    flux_y_2_r = T(0.5)*(flux_y_2(vx_l, vy_l) + flux_y_2(vx_r, vy_r) - alpha*(vy_r - vy_l))

    alpha_smoke = max(abs(vy_l), abs(vy_r))
    flux_y_smoke_r = T(0.5)*(flux_y_smoke(vy_l, c_l) + flux_y_smoke(vy_r, c_r) - alpha_smoke*(c_r - c_l))

    return flux_y_1_r, flux_y_2_r, flux_y_smoke_r
end

# berechne den numerischen Fluss in x-Richtung (H^x_{i+1/2,j})
function hybrid_llf_upwind_x(vx_l::T, vx_r::T, vy_l::T, vy_r::T, c_l::T, c_r::T) where T
    # alpha = max(abs(vx_l), abs(vx_r))
    alpha = 2*max(abs(vx_l), abs(vx_r))
    flux_x_1_r = T(0.5)*(flux_x_1(vx_l, vy_l) + flux_x_1(vx_r, vy_r) - alpha*(vx_r - vx_l))
    flux_x_2_r = T(0.5)*(flux_x_2(vx_l, vy_l) + flux_x_2(vx_r, vy_r) - alpha*(vy_r - vy_l))

    vx_mean = T(0.5)*(vx_l+vx_r)
    flux_x_smoke_r = vx_mean >= T(0.0) ? flux_x_smoke(vx_mean, c_l) : flux_x_smoke(vx_mean, c_r)

    return flux_x_1_r, flux_x_2_r, flux_x_smoke_r
end

# berechne den numerischen Fluss in y-Richtung (H^y_{i,j+1/2})
function hybrid_llf_upwind_y(vx_l::T, vx_r::T, vy_l::T, vy_r::T, c_l::T, c_r::T) where T
    # alpha = max(abs(vy_l), abs(vy_r))
    alpha = 2*max(abs(vy_l), abs(vy_r))
    flux_y_1_r = T(0.5)*(flux_y_1(vx_l, vy_l) + flux_y_1(vx_r, vy_r) - alpha*(vx_r - vx_l))
    flux_y_2_r = T(0.5)*(flux_y_2(vx_l, vy_l) + flux_y_2(vx_r, vy_r) - alpha*(vy_r - vy_l))

    vy_mean = T(0.5)*(vy_l+vy_r)
    flux_y_smoke_r = vy_mean >= T(0.0) ? flux_y_smoke(vy_mean, c_l) : flux_y_smoke(vy_mean, c_r)

    return flux_y_1_r, flux_y_2_r, flux_y_smoke_r
end