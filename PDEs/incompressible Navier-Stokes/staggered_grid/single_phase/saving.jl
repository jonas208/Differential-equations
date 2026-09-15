function get_current_solution(navier_stokes_integrator, advection_integrator)
    param = navier_stokes_integrator.p

    u = navier_stokes_integrator.u
    
    vx_length = param.nx_vx * param.ny_vx
    vy_length = param.nx_vy * param.ny_vy

    vx = reshape(view(u, 1:vx_length), param.nx_vx, param.ny_vx)
    vy = reshape(view(u, vx_length+1:vx_length+vy_length), param.nx_vy, param.ny_vy)

    u = advection_integrator.u
    c = reshape(u, param.nx_c, param.ny_c)

    # kopiere die Matrizen in den RAM der CPU
    return Matrix(vx), Matrix(vy), Matrix(param.pressure), Matrix(c)
end