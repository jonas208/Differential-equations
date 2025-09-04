using OrdinaryDiffEq
using GLMakie

# Spezifikationen des Problems
L = 1000.0 # Länge des Flusses [m]
g = 9.81 # Erdbeschleunigung [m/s^2]

plot_size = (1000, 1000) # Auflösung der Plots
fps = 25 # Bildwiederholrate der Animationen

# Anfangsbedingung
u0(x) = x < 500.0 ? [10.0, 0.0] : [1.0, 0.0] # Deichbruchproblem

# Randbedingungen
bc(h_l, hv_l, h_r, hv_r) = h_l, 0.0, 1.0, hv_r # Randbedingungen Deichbruch

function u0!(u, n, dx)
    for i in 1:n
        u[i, :] .= u0((i-0.5)*dx)
    end
end

# Zeitintervall
tspan = (0.0, 30.0)
n_frames = ceil(Int, tspan[2]-tspan[1])*fps

# Ortsdiskretisierung
n = 500
dx = L/n
xs = range(dx/2, L-dx/2; length = n)

# WENO-Parameter
use_weno = true
ep = 1.0e-6; p = 0.6;

uL = zeros(n+1, 2); uR = zeros(n+1, 2)
param = (L, g, bc, n, dx, use_weno, ep, p, uL, uR)

get_vars(u) = u[1], u[2]/u[1]
get_abs_lam(h, v, g) = abs(v) + sqrt(g*h)
get_flux(h, v, g) = [h*v, v^2*h + 0.5*g*h^2]

function recover!(yL, yR, y)
    yL[:, 1] .= [0.5*(3*y[1, 1]-y[2, 1]); y[:, 1]]; yR[:, 1] .= [y[:, 1]; 0.5*(3*y[end, 1]-y[end-1, 1])]
    yL[:, 2] .= [0.5*(3*y[1, 2]-y[2, 2]); y[:, 2]]; yR[:, 2] .= [y[:, 2]; 0.5*(3*y[end, 2]-y[end-1, 2])]
end

function recover_weno!(yL, yR, y, ep, p)
    #-- Zellenmittelwerte auf Zellgrenzen interpolieren
    #-- L = upwind, R = downwind, WENO 3. Ordnung
    n = size(y)[1];
    for j in 1:2
        yL[1, j] = 11/6*y[1, j]-7/6*y[2, j]+y[3, j]/3; yR[1, j] = yL[1, j]; #-- Randwerte
        yL[2, j] = y[1, j]/3+5/6*y[2, j]-y[3, j]/6; 
        yL[n+1, j] = 11/6*y[n, j]-7/6*y[n-1, j]+y[n-2, j]/3; yR[n+1, j] = yL[n+1, j]; 
        yR[n, j] = y[n, j]/3+5/6*y[n-1, j]-y[n-2, j]/6; 
        for i in 2:n-1
            yR[i, j], yL[i+1, j] = weno3(y[i-1:i+1, j], ep, p);
        end
    end
end
      
function weno3(y, ep, p) #-- y = [y1, y2, y3]
    uL = y[2]-y[1]; uC = y[3]-2*y[2]+y[1]; uR = y[3]-y[2]; uCC = y[3]-y[1];
    ISL = uL^2; ISC = 13/3*uC^2 + 0.25*uCC^2; ISR = uR^2;
    aL = 0.25*(1/(ep+ISL))^p; aC = 0.5*(1/(ep+ISC))^p; aR = 0.25*(1/(ep+ISR))^p;
    suma = max(aL+aC+aR,eps(1.0)); 
    wL = aL/suma; wC = aC/suma; wR = aR/suma;
    y12 = (0.5*wL+5/12*wC)*y[1] + (0.5*wL+2/3*wC+1.5*wR)*y[2] + (-wC/12-0.5*wR)*y[3];
    y23 = (-0.5*wL-wC/12)*y[1] + (1.5*wL+2/3*wC+0.5*wR)*y[2] + (5/12*wC+0.5*wR)*y[3];
    return y12, y23
end

function local_lax_friedrichs(u_l, u_r, g)
    h_l, v_l = get_vars(u_l)
    h_r, v_r = get_vars(u_r)

    lam = max(get_abs_lam(h_l, v_l, g), get_abs_lam(h_r, v_r, g))
    flux_r = 0.5*(get_flux(h_l, v_l, g) .+ get_flux(h_r, v_r, g) .- lam*(u_r .- u_l))

    return flux_r
end

# DGL-System erster Ordnung (Linienmethode mit finiten Volumen)
function shallow_water_equations_fvm!(du, u, param, t)
    L, g, bc, n, dx, use_weno, ep, p, uL, uR = param

    if use_weno
        recover_weno!(uL, uR, u, ep, p)
    else
        recover!(uL, uR, u)
    end
    h_l, hv_l, h_r, hv_r = uL[1, 1], uL[1, 2], uR[n+1, 1], uR[n+1, 2]
    h_l_bc, hv_l_bc, h_r_bc, hv_r_bc = bc(h_l, hv_l, h_r, hv_r)
    flux_r = local_lax_friedrichs([h_l_bc, hv_l_bc], u[1, :], g)
    
    for i in 1:n
        flux_l = flux_r
        
        if i == n
            flux_r = local_lax_friedrichs(u[n, :], [h_r_bc, hv_r_bc], g)
        else
            flux_r = local_lax_friedrichs(uL[i+1, :], uR[i+1, :], g)
        end

        @. du[i, :] = -(flux_r - flux_l) / dx
    end
end

u0s = zeros(n, 2)
u0!(u0s, n, dx)

solver = SSPRK432()
tol = 1.0e-5

f = ODEFunction(shallow_water_equations_fvm!)
prob = ODEProblem(f, u0s, tspan, param)
sol = @time solve(prob, solver; reltol = tol, abstol = tol)

display(sol.stats)

# Interface für Lösung

get_hs(sol, t) = sol(t)[:, 1]
get_vs(sol, t) = sol(t)[:, 2] ./ sol(t)[:, 1]

# Plotting

time = Observable(tspan[2])
hs = @lift get_hs(sol, $time)
vs = @lift get_vs(sol, $time)

fig = Figure(size = plot_size)

ax1 = Axis(fig[1, 1], title = "Wasserspiegelhöhe (Flachwassergleichungen)"; limits = (0.0, L, 0.0, 15.0))
lin = lines!(xs, hs)

ax2 = Axis(fig[2, 1], title = "Volumenfluss (Flachwassergleichungen)"; limits = (0.0, L, 0.0, 40.0))
lin = lines!(xs, @lift($hs .* $vs))

record(fig, "Gewässer.mp4", range(tspan[1], tspan[2]; length = n_frames); framerate = fps) do t
    time[] = t
end