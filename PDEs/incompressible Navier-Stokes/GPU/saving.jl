function save_func(u, t, integrator)
    param = integrator.p
    Lx, Ly, kinematic_viscosity, density, horizontal_velocity, obstacles, nx, ny, dx, dy, coefficient_mat, pressure_vec, divergence_mat, lin_solve, use_weno, ep, p, fluxes_x, fluxes_y = param
    
    u = copy(u)
    pressure_vec = copy(pressure_vec)

    vx = reshape(view(u, 1:nx*ny), nx, ny)
    vy = reshape(view(u, nx*ny+1:2*nx*ny), nx, ny)

    pressure = reshape(pressure_vec, nx, ny)

    return vx, vy, pressure
end