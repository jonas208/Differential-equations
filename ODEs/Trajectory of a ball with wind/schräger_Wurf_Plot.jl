using OrdinaryDiffEq
using Plots
using LaTeXStrings

# Werte für einen Handball
c_w = 0.45
rho = 1.2041
A = 0.58^2 / (4*π)
m = 0.425

function drag_force(ddx, dx, x, p, t)
    k, v_w = p
    ddx .= @. -k * (dx - v_w) * abs(dx - v_w)
end

# m, m, m/s, rad, [m/s], (s, s)
function plot_different_winds(h_0, x_0, v_0, alpha, winds, tspan)
    # Anfangswerte
    v_x0 = cos(alpha)*v_0 # m/s
    v_y0 = sin(alpha)*v_0 # m/s

    # für Plotting
    t_min = tspan[1]
    t_max = tspan[2]
    dt = (t_max - t_min)/10_000
    t_interval = t_min:dt:t_max

    plt = plot()
    for v_w in winds
        k = (2m)^-1*c_w*rho*A
        prob = SecondOrderODEProblem(drag_force, [v_x0], [x_0], tspan, [k, v_w])
        sol = solve(prob, Tsit5(), reltol = 1e-14, abstol = 1e-14)
        
        xs = x.(Ref(sol), t_interval)
        ys = h.(Ref(h_0), Ref(v_y0), t_interval)
        label = L"v_W = %$(round(v_w*3.6, digits=2)) \, \mathrm{km} \, \mathrm{h^{-1}}"
        plot!(plt, xs, ys, linewidth = 2, title = L"\mathbf{Schräger \ Wurf \ mit \ Wind}", xaxis = L"x \ \mathrm{in} \ \mathrm{m}", 
            yaxis = L"y \ \mathrm{in} \ \mathrm{m}", label = label, xlims = [0, 25], ylims = [0, 8.5])
    end
    hline!(plt, [h_0], linewidth = 2, color = :grey, label = L"\textrm{Abwurfhöhe}")
    return plt
end

# m, m, m/s, [rad], m/s, (s, s)
function plot_different_alphas(h_0, x_0, v_0, alphas, v_w, tspan)
    # für Plotting
    t_min = tspan[1]
    t_max = tspan[2]
    dt = (t_max - t_min)/10_000
    t_interval = t_min:dt:t_max

    plt = plot()
    for alpha in alphas
        # Anfangswerte
        v_x0 = cos(alpha)*v_0 # m/s
        v_y0 = sin(alpha)*v_0 # m/s

        k = (2m)^-1*c_w*rho*A
        prob = SecondOrderODEProblem(drag_force, [v_x0], [x_0], tspan, [k, v_w])
        sol = solve(prob, Tsit5(), reltol = 1e-14, abstol = 1e-14)
        
        xs = x.(Ref(sol), t_interval)
        ys = h.(Ref(h_0), Ref(v_y0), t_interval)
        label = L"\alpha = %$(round(rad2deg(alpha), digits=2))°"
        plot!(plt, xs, ys, linewidth = 2, title = L"\mathbf{Schräger \ Wurf \ mit \ Wind}", xaxis = L"x \ \mathrm{in} \ \mathrm{m}", 
            yaxis = L"y \ \mathrm{in} \ \mathrm{m}", label = label, xlims = [0, 25], ylims = [0, 8.5])
    end
    hline!(plt, [h_0], linewidth = 2, color = :grey, label = L"\textrm{Abwurfhöhe}")
    return plt
end

# Interace für Lösung
x(sol, t) = sol(t)[2]
dx(sol, t) = sol(t)[1]

# analytische Lösung für h(t)
h(h_0, v_y0, t) = -0.5*9.81*t^2 + v_y0*t + h_0

# Zeitintervall
tspan = (0.0, 5.0)

# Pfad zum speichern der Abbildungen
figures_path = "C:/Users/joerg/Documents/Schule/Schule Jahrgang Q2/Physik/Hausarbeit/figures/"

winds = [-15., -5., 0., 5., 15.] # km/h
plt_1 = plot_different_winds(1.85, 0.0, 50/3.6, deg2rad(45), [v_w/3.6 for v_w in winds], tspan)
plt_1
savefig(plt_1, figures_path * "Wurf_1.svg")

alphas = [35., 40., 45., 50., 55.] # Grad
plt_2 = plot_different_alphas(1.85, 0.0, 50/3.6, [deg2rad(alpha) for alpha in alphas], -30/3.6, tspan)
plt_2
savefig(plt_2, figures_path * "Wurf_2.svg")