function save_func(u, t, integrator)
    param = integrator.p
    L, kinematic_viscosity, density, horizontal_velocity, obstacles, n, dx, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param
    
    u = copy(u)
    pressure_vec = copy(pressure_vec)

    vx = reshape(view(u, 1:n^2), n, n)
    vy = reshape(view(u, n^2+1:2*n^2), n, n)

    pressure = reshape(pressure_vec, n, n)

    return vx, vy, pressure
end