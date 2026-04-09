function apply_boundary_conditions_vx_kernel!(vx::CuDeviceMatrix{T}, horizontal_velocity::T, nx_vx, ny_vx) where T
    k = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if k <= nx_vx
        # unterer Rand (No-Slip-Bedingung)
        vx[k, 2] = -vx[k, 3] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben
        vx[k, 1] = -vx[k, 4] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben

        # oberer Rand (No-Slip-Bedingung)
        vx[k, end-1] = -vx[k, end-2] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben
        vx[k, end] = -vx[k, end-3] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben
    end

    if k <= ny_vx
        # linker Rand (Einströmungsbedingung)
        vx[2, k] = horizontal_velocity # auf dem Rand
        vx[1, k] = T(2.0) * horizontal_velocity - vx[3, k] # hinter dem Rand, lineare Interpolation auf den Rand soll horizontal_velocity ergeben
        
        # rechter Rand (Ausströmungsbedingung mit Backflow-Limiter)
        outflow_val = max(T(0.0), vx[end-2, k]) # nur Ausströmung erlauben
        
        vx[end-1, k] = outflow_val # auf dem Rand
        vx[end, k] = outflow_val # hinter dem Rand
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
        # linker Rand (Einströmungsbedingung)
        vy[2, k] = -vy[3, k] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben
        vy[1, k] = -vy[4, k] # hinter dem Rand, lineare Interpolation auf den Rand soll 0 ergeben

        # rechter Rand (Ausströmungsbedingung)
        vy[end-1, k] = vy[end-2, k] # hinter dem Rand, Ableitung auf dem Rand soll 0 ergeben
        vy[end, k] = vy[end-2, k] # hinter dem Rand, Ableitung auf dem Rand soll 0 ergeben
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

function apply_obstacle_conditions_vx_kernel!(vx::CuDeviceMatrix{T}, is_solid::CuDeviceMatrix{Bool}, nx_vx, ny_vx) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # nur innerer Bereich, d.h. kein Rand- bzw. Ghost-Zellen
    if 3 <= i <= nx_vx - 2 && 3 <= j <= ny_vx - 2
        i_, j_ = get_left_pressure_index(i, j)
        is_solid_left = is_solid[i_, j_]
        i_, j_ = get_right_pressure_index(i, j)
        is_solid_right = is_solid[i_, j_]

        # genau auf dem vertikalen Rand oder innerhalb eines Hindernisses
        if is_solid_left || is_solid_right
            vx[i, j] = T(0.0)
        end
    end

    return nothing
end

function apply_obstacle_conditions_vx!(vx::CuMatrix{T}, param) where T
    kernel = @cuda launch = false apply_obstacle_conditions_vx_kernel!(vx, param.is_solid, param.nx_vx, param.ny_vx)
    config = CUDA.launch_configuration(kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_vx, threads_per_dim)
    x_blocks = cld(param.nx_vx, x_threads)

    y_threads = min(param.ny_vx, threads_per_dim)
    y_blocks = cld(param.ny_vx, y_threads)

    kernel(vx, param.is_solid, param.nx_vx, param.ny_vx, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

function apply_obstacle_conditions_vy_kernel!(vy::CuDeviceMatrix{T}, is_solid::CuDeviceMatrix{Bool}, nx_vy, ny_vy) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # nur innerer Bereich, d.h. kein Rand- bzw. Ghost-Zellen
    if 3 <= i <= nx_vy - 2 && 3 <= j <= ny_vy - 2
        i_, j_ = get_bottom_pressure_index(i, j)
        is_solid_bottom = is_solid[i_, j_]
        i_, j_ = get_top_pressure_index(i, j)
        is_solid_top = is_solid[i_, j_]

        # genau auf dem horizontalen Rand oder innerhalb eines Hindernisses
        if is_solid_bottom || is_solid_top
            vy[i, j] = T(0.0)
        end
    end

    return nothing
end

function apply_obstacle_conditions_vy!(vy::CuMatrix{T}, param) where T
    kernel = @cuda launch = false apply_obstacle_conditions_vy_kernel!(vy, param.is_solid, param.nx_vy, param.ny_vy)
    config = CUDA.launch_configuration(kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_vy, threads_per_dim)
    x_blocks = cld(param.nx_vy, x_threads)

    y_threads = min(param.ny_vy, threads_per_dim)
    y_blocks = cld(param.ny_vy, y_threads)

    kernel(vy, param.is_solid, param.nx_vy, param.ny_vy, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

function navier_stokes_limiter!(u, navier_stokes_integrator, param, t::T) where T
    vx_length = param.nx_vx * param.ny_vx
    vy_length = param.nx_vy * param.ny_vy

    vx = reshape(view(u, 1:vx_length), param.nx_vx, param.ny_vx)
    vy = reshape(view(u, vx_length+1:vx_length+vy_length), param.nx_vy, param.ny_vy)

    apply_boundary_conditions_vx!(vx, param)
    apply_boundary_conditions_vy!(vy, param)

    apply_obstacle_conditions_vx!(vx, param)
    apply_obstacle_conditions_vy!(vy, param)
end

function apply_boundary_conditions_c_kernel!(c::CuDeviceMatrix{T}, nx_c, ny_c) where T
    k = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if k <= nx_c
        # unterer Rand (Neumann-Randbedingung)
        c[k, 2] = c[k, 3] # hinter dem Rand, Ableitung auf dem Rand soll 0 ergeben
        c[k, 1] = c[k, 4] # hinter dem Rand, Ableitung auf dem Rand soll 0 ergeben

        # oberer Rand (Neumann-Randbedingung)
        c[k, end-1] = c[k, end-2] # hinter dem Rand, Ableitung auf dem Rand soll 0 ergeben
        c[k, end] = c[k, end-3] # hinter dem Rand, Ableitung auf dem Rand soll 0 ergeben
    end

    if k <= ny_c
        # linker Rand (Dirichlet-Randbedingung)
        # inlet_val = 95 <= k <= 505 && (k % 100 <= 5 || k % 100 >= 95) ? T(1.0) : T(0.0)
        inlet_val = 106 <= k <= 1157 && (k % 50 <= 5 || k % 50 >= 45) ? T(1.0) : T(0.0)
        c[2, k] = T(2.0) * inlet_val - c[3, k] # hinter dem Rand, Interpolation auf den Rand soll geforderten Wert ergeben
        c[1, k] = T(2.0) * inlet_val - c[4, k] # hinter dem Rand, Interpolation auf den Rand soll geforderten Wert ergeben

        # rechter Rand (Neumann-Randbedingung)
        c[end-1, k] = c[end-2, k] # hinter dem Rand, Ableitung auf dem Rand soll 0 ergeben
        c[end, k] = c[end-3, k] # hinter dem Rand, Ableitung auf dem Rand soll 0 ergeben
    end

    return nothing
end

function apply_boundary_conditions_c!(c::CuMatrix{T}, param) where T
    kernel = @cuda launch = false apply_boundary_conditions_c_kernel!(c, param.nx_c, param.ny_c)
    config = CUDA.launch_configuration(kernel.fun)

    threads = config.threads

    n_max = max(param.nx_c, param.ny_c)
    x_threads = min(n_max, threads)
    x_blocks = cld(n_max, x_threads)

    kernel(c, param.nx_c, param.ny_c, threads=x_threads, blocks=x_blocks)
end

function apply_obstacle_conditions_c_kernel!(c::CuDeviceMatrix{T}, is_solid::CuDeviceMatrix{Bool}, nx_c, ny_c) where T
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # nur innerer Bereich, d.h. kein Rand- bzw. Ghost-Zellen
    if 3 <= i <= nx_c - 2 && 3 <= j <= ny_c - 2
        # eigentlich wäre für undurchlässige Hindernisse auch eine Neumann-Randbedingung sinnvoll,
        # damit es einfacher ist, wird der Rauch in Hindernissen aber einfach auf 0 gesetzt
        if is_solid[i-2, j-2]
            c[i, j] = T(0.0)
        end
    end

    return nothing
end

function apply_obstacle_conditions_c!(c::CuMatrix{T}, param) where T
    kernel = @cuda launch = false apply_obstacle_conditions_c_kernel!(c, param.is_solid, param.nx_c, param.ny_c)
    config = CUDA.launch_configuration(kernel.fun)

    threads_per_dim = Int(floor(sqrt(config.threads)))

    x_threads = min(param.nx_c, threads_per_dim)
    x_blocks = cld(param.nx_c, x_threads)

    y_threads = min(param.ny_c, threads_per_dim)
    y_blocks = cld(param.ny_c, y_threads)

    kernel(c, param.is_solid, param.nx_c, param.ny_c, threads=(x_threads, y_threads), blocks=(x_blocks, y_blocks))
end

function advection_limiter!(u, advection_integrator, param, t::T) where T
    c = reshape(u, param.nx_c, param.ny_c)

    apply_boundary_conditions_c!(c, param)
    apply_obstacle_conditions_c!(c, param)
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

function superbee(r::T) where T
    return max(T(0.0), min(T(2.0) * r, T(1.0)), min(r, T(2.0)))
end

# 1D-MUSCL-Rekonstruktion in x-Richtung (TVD-Schema)
function recover_x_c(i, j, c::CuDeviceMatrix{T}) where T
    den1 = c[i+1, j] - c[i, j]
    r1 = den1 == T(0.0) ? T(0.0) : (c[i, j] - c[i-1, j]) / den1
    c_l = c[i, j] + T(0.5) * superbee(r1) * den1

    den2 = c[i+2, j] - c[i+1, j]
    r2 = den2 == T(0.0) ? T(0.0) : (c[i+1, j] - c[i, j]) / den2
    c_r = c[i+1, j] - T(0.5) * superbee(r2) * den2

    return c_l, c_r
end

# 1D-MUSCL-Rekonstruktion in y-Richtung (TVD-Schema)
function recover_y_c(i, j, c::CuDeviceMatrix{T}) where T
    den1 = c[i, j+1] - c[i, j]
    r1 = den1 == T(0.0) ? T(0.0) : (c[i, j] - c[i, j-1]) / den1
    c_l = c[i, j] + T(0.5) * superbee(r1) * den1

    den2 = c[i, j+2] - c[i, j+1]
    r2 = den2 == T(0.0) ? T(0.0) : (c[i, j+1] - c[i, j]) / den2
    c_r = c[i, j+1] - T(0.5) * superbee(r2) * den2

    return c_l, c_r
end

flux_x_vx(vx::T) where T = vx^2
flux_y_vx(vx::T, vy::T) where T = vx*vy

flux_x_vy(vx::T, vy::T) where T = vx*vy
flux_y_vy(vy::T) where T = vy^2

flux_x_c(vx::T, c::T) where T = vx*c
flux_y_c(vy::T, c::T) where T = vy*c

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

# berechne den numerischen Fluss in x-Richtung (H^x_{i+1/2,j})
function upwind_x_c(c_l::T, c_r::T, vx_right::T) where T
    return vx_right >= T(0.0) ? flux_x_c(vx_right, c_l) : flux_x_c(vx_right, c_r)
end

# berechne den numerischen Fluss in y-Richtung (H^y_{i,j+1/2})
function upwind_y_c(c_l::T, c_r::T, vy_top::T) where T
    return vy_top >= T(0.0) ? flux_x_c(vy_top, c_l) : flux_x_c(vy_top, c_r)
end