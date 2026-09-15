function save_func(u, t, integrator)
    param = integrator.p

    vx_length = param.nx_vx * param.ny_vx
    vy_length = param.nx_vy * param.ny_vy

    vx = reshape(view(u, 1:vx_length), param.nx_vx, param.ny_vx)
    vy = reshape(view(u, vx_length+1:vx_length+vy_length), param.nx_vy, param.ny_vy)

    # kopiere die Matrizen in den RAM der CPU
    return Matrix(vx), Matrix(vy), Matrix(param.pressure)
end