function apply_boundary_conditions_vx_kernel!(vx::CuDeviceMatrix{T}, horizontal_velocity::T, nx_vx, ny_vx) where T
    k = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if k <= nx_vx
        # unterer Rand (No-Slip-Bedingung)
        vx[k, 2] = -vx[k, 3] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben
        vx[k, 1] = -vx[k, 4] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben

        # oberer Rand (No-Slip-Bedingung)
        vx[k, end-1] = T(2.0) * horizontal_velocity - vx[k, end-2] # hinter dem Rand, lineare Interpolation auf den Rand soll horizontal_velocity ergeben
        vx[k, end] = T(2.0) * horizontal_velocity - vx[k, end-3] # hinter dem Rand, lineare Interpolation auf den Rand soll horizontal_velocity ergeben
    end

    if k <= ny_vx
        # linker Rand (No-Slip-Bedingung)
        vx[2, k] = T(0.0) # auf dem Rand
        vx[1, k] = -vx[3, k] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben

        # rechter Rand (No-Slip-Bedingung)
        vx[end-1, k] = T(0.0) # auf dem Rand
        vx[end, k] = vx[end-2, k] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben
    end

    return nothing
end

function apply_boundary_conditions_vx!(vx::CuMatrix{T}, param) where T
    kernel = @cuda launch = false apply_boundary_conditions_vx_kernel!(vx, param.horizontal_velocity, param.nx_vx, param.ny_vx)
    config = CUDA.launch_configuration(kernel.fun)

    threads = config.threads

    n_max = max(param.nx_vx, param.ny_vx)
    x_threads = min(n_max, threads)
    x_blocks = cld(n_max, x_threads)

    kernel(vx, param.horizontal_velocity, param.nx_vx, param.ny_vx, threads=x_threads, blocks=x_blocks)
end

function apply_boundary_conditions_vy_kernel!(vy::CuDeviceMatrix{T}, nx_vy, ny_vy) where T
    k = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if k <= nx_vy
        # unterer Rand (No-Slip-Bedingung)
        vy[k, 2] = T(0.0) # auf dem Rand
        vy[k, 1] = -vy[k, 3] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben

        # oberer Rand (No-Slip-Bedingung)
        vy[k, end-1] = T(0.0) # auf dem Rand
        vy[k, end] = -vy[k, end-2] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben
    end

    if k <= ny_vy
        # linker Rand (No-Slip-Bedingung)
        vy[2, k] = -vy[3, k] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben
        vy[1, k] = -vy[4, k] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben

        # rechter Rand (No-Slip-Bedingung)
        vy[end-1, k] = -vy[end-2, k] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben
        vy[end, k] = -vy[end-3, k] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben
    end

    return nothing
end

function apply_boundary_conditions_vy!(vy::CuMatrix{T}, param) where T
    kernel = @cuda launch = false apply_boundary_conditions_vy_kernel!(vy, param.nx_vy, param.ny_vy)
    config = CUDA.launch_configuration(kernel.fun)

    threads = config.threads

    n_max = max(param.nx_vy, param.ny_vy)
    x_threads = min(n_max, threads)
    x_blocks = cld(n_max, x_threads)

    kernel(vy, param.nx_vy, param.ny_vy, threads=x_threads, blocks=x_blocks)
end

function apply_boundary_conditions_limiter!(u, integrator, param, t::T) where T
    vx_length = param.nx_vx * param.ny_vx
    vy_length = param.nx_vy * param.ny_vy

    vx = reshape(view(u, 1:vx_length), param.nx_vx, param.ny_vx)
    vy = reshape(view(u, vx_length+1:vx_length+vy_length), param.nx_vy, param.ny_vy)

    apply_boundary_conditions_vx!(vx, param)
    apply_boundary_conditions_vy!(vy, param)
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

function recover_x_vx(i, j, dx::T, dy::T, vx::CuDeviceMatrix{T}, ep::T, p::T) where T
    x_i_l = get_vx_position(i, j, dx, dy)[1] # x-Koordinate der Mitte des ersten Stencils
    x = x_i_l + T(0.5) * dx # x-Koordinate der Zellkante
    x_i_r = get_vx_position(i+1, j, dx, dy)[1] # x-Koordinate der Mitte des zweiten Stencils

    vx_l = cweno_x(x, x_i_l, dx, vx[i-1, j], vx[i, j], vx[i+1, j], vx[i, j-1], vx[i, j], vx[i, j+1], ep, p)
    vx_r = cweno_x(x, x_i_r, dx, vx[i, j], vx[i+1, j], vx[i+2, j], vx[i+1, j-1], vx[i+1, j], vx[i+1, j+1], ep, p)

    return vx_l, vx_r
end

function recover_y_vx(i, j, dx::T, dy::T, vx::CuDeviceMatrix{T}, ep::T, p::T) where T
    y_i_l = get_vx_position(i, j, dx, dy)[2] # y-Koordinate der Mitte des ersten Stencils
    y = y_i_l + T(0.5) * dy # y-Koordinate der Zellkante
    y_i_r = get_vx_position(i, j+1, dx, dy)[2] # y-Koordinate der Mitte des zweiten Stencils

    vx_l = cweno_y(y, y_i_l, dy, vx[i, j-1], vx[i, j], vx[i, j+1], vx[i-1, j], vx[i, j], vx[i+1, j], ep, p) # P_j(y_{j+1/2})
    vx_r = cweno_y(y, y_i_r, dy, vx[i, j], vx[i, j+1], vx[i, j+2], vx[i-1, j+1], vx[i, j+1], vx[i+1, j+1], ep, p) # P_{j+1}(y_{j+1/2})

    return vx_l, vx_r
end

function recover_x_vy(i, j, dx::T, dy::T, vy::CuDeviceMatrix{T}, ep::T, p::T) where T
    x_i_l = get_vy_position(i, j, dx, dy)[1] # x-Koordinate der Mitte des ersten Stencils
    x = x_i_l + T(0.5) * dx # x-Koordinate der Zellkante
    x_i_r = get_vy_position(i+1, j, dx, dy)[1] # x-Koordinate der Mitte des zweiten Stencils

    vy_l = cweno_x(x, x_i_l, dx, vy[i-1, j], vy[i, j], vy[i+1, j], vy[i, j-1], vy[i, j], vy[i, j+1], ep, p)
    vy_r = cweno_x(x, x_i_r, dx, vy[i, j], vy[i+1, j], vy[i+2, j], vy[i+1, j-1], vy[i+1, j], vy[i+1, j+1], ep, p)

    return vy_l, vy_r
end

function recover_y_vy(i, j, dx::T, dy::T, vy::CuDeviceMatrix{T}, ep::T, p::T) where T
    y_i_l = get_vy_position(i, j, dx, dy)[2] # y-Koordinate der Mitte des ersten Stencils
    y = y_i_l + T(0.5) * dy # y-Koordinate der Zellkante
    y_i_r = get_vy_position(i, j+1, dx, dy)[2] # y-Koordinate der Mitte des zweiten Stencils

    vy_l = cweno_y(y, y_i_l, dy, vy[i, j-1], vy[i, j], vy[i, j+1], vy[i-1, j], vy[i, j], vy[i+1, j], ep, p) # P_j(y_{j+1/2})
    vy_r = cweno_y(y, y_i_r, dy, vy[i, j], vy[i, j+1], vy[i, j+2], vy[i-1, j+1], vy[i, j+1], vy[i+1, j+1], ep, p) # P_{j+1}(y_{j+1/2})

    return vy_l, vy_r
end

flux_x_vx(vx::T) where T = vx^2
flux_y_vx(vx::T, vy::T) where T = vx*vy

flux_x_vy(vx::T, vy::T) where T = vx*vy
flux_y_vy(vy::T) where T = vy^2

# berechne den numerischen Fluss in x-Richtung (H^x_{i+1/2,j})
function local_lax_friedrichs_x_vx(vx_l::T, vx_r::T) where T
    alpha = max(abs(vx_l), abs(vx_r))
    flux_x = T(0.5) * (flux_x_vx(vx_l) + flux_x_vx(vx_r) - alpha * (vx_r - vx_l))

    return flux_x
end

# berechne den numerischen Fluss in y-Richtung (H^y_{i,j+1/2})
function local_lax_friedrichs_y_vx(vx_l::T, vx_r::T, vy_top::T) where T
    alpha = abs(vy_top)
    flux_y = T(0.5) * (flux_y_vx(vx_l, vy_top) + flux_y_vx(vx_r, vy_top) - alpha * (vx_r - vx_l))

    return flux_y
end

# berechne den numerischen Fluss in x-Richtung (H^x_{i+1/2,j})
function local_lax_friedrichs_x_vy(vy_l::T, vy_r::T, vx_right::T) where T
    alpha = abs(vx_right)
    flux_x = T(0.5) * (flux_x_vy(vx_right, vy_l) + flux_x_vy(vx_right, vy_r) - alpha * (vy_r - vy_l))

    return flux_x
end

# berechne den numerischen Fluss in y-Richtung (H^y_{i,j+1/2})
function local_lax_friedrichs_y_vy(vy_l::T, vy_r::T) where T
    alpha = max(abs(vy_l), abs(vy_r))
    flux_y = T(0.5) * (flux_y_vy(vy_l) + flux_y_vy(vy_r) - alpha * (vy_r - vy_l))

    return flux_y
end