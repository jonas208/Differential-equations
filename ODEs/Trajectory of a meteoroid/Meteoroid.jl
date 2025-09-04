using OrdinaryDiffEq
using Plots
using LaTeXStrings

# relevante Konstanten
m_earth = 5.9722e24	# kg
r_earth = 6_371_000.785 # m
g = 9.81 # m/s^2
mm_atmosphäre = 0.02896 # kg/mol
gas_constant = 8.31446261815324 # J/(mol*K)
gravitational_constant = 6.67430e-11 # m^3/(kg*s^2)
rho_0 = 1.2041 # kg/m^3

# nützliche Funktionen für Kugeln
r2V(r) = (4/3)*pi*r^3
V2r(V) = cbrt(V*(3/(4*pi)))
r2A(r) = pi*r^2

# Werte für einen kugelförmigen Meteoroiden
c_w = 0.45
V = 1e-6 # m^3
m = 7.874e-3 # kg

# Temperatur der Luft (wird als isotherm angenommen)
T = 273.15 # K (entspricht 0 °C)

function combined_acceleration(dh, h, t, p)
    m_earth, r_earth, g, mm_atmosphäre, gas_constant, gravitational_constant, rho_0, k = p
    return -gravitational_constant*(m_earth/(r_earth+h)^2) + k*rho_0*exp(-((mm_atmosphäre*g)/(gas_constant*T))*h)*dh^2
end

function combined_acceleration(ddh, dh, h, p, t)
    # m_earth, r_earth, g, mm_atmosphäre, gas_constant, gravitational_constant, rho_0, k = p
    # ddh .= @. -gravitational_constant*(m_earth/(r_earth+h)^2) + k*rho_0*exp(-((mm_atmosphäre*g)/(gas_constant*T))*h)*dh^2
    ddh .= combined_acceleration.(dh, h, t, Ref(p))
end

function solve_ivp(h_0, v_0, tspan, scale_factor) # Dichte bleibt unverändert
    V = 1e-6*scale_factor # m^3
    A = r2A(V2r(V)) # m^2
    m = 7.874e-3*scale_factor # kg
    k = (2m)^-1*c_w*A 
    p = [m_earth, r_earth, g, mm_atmosphäre, gas_constant, gravitational_constant, rho_0, k]
    prob = SecondOrderODEProblem(combined_acceleration, [v_0], [h_0], tspan, p)
    solve(prob, Tsit5(), reltol = 1e-15, abstol = 1e-15), p
end

function get_plots(sol, p, label; a_plot = plot(), v_plot = plot())
    a_t = a.(Ref(sol), Ref(p), t_interval)./1000
    v_t = abs.(v.(Ref(sol), t_interval))./1000
    h_t = [h(sol, t) for t in t_interval]./1000

    #= Einheiten als Bruch
    label = L"v_0 = %$v_0 \, \frac{\mathrm{m}}{\mathrm{s}}"
    a_plot = plot!(a_plot, h_t, a_t, linewidth = linewidth, title = L"\mathbf{Meteoroid}", xaxis = L"\mathrm{Höhe \ in} \ \mathrm{km}", 
        yaxis = L"\mathrm{Bremsbeschleunigung \ in} \ \frac{\mathrm{km}}{\mathrm{s^2}}", label = label)
    v_plot = plot!(v_plot, h_t, v_t, linewidth = linewidth, title = L"\mathrm{Meteoroid}", xaxis = L"\mathrm{Höhe \ in} \ \mathrm{km}", 
        yaxis = L"\mathrm{Fallgeschwindigkeit \ in} \ \frac{\mathrm{km}}{\mathrm{s}}", label = label)
    =#

    a_plot = plot!(a_plot, h_t, a_t, linewidth = linewidth, title = L"\mathbf{Meteoroid}", xaxis = L"\mathrm{Höhe \ in} \ \mathrm{km}", 
        yaxis = L"\mathrm{Bremsbeschleunigung \ in} \ \mathrm{km} \, \mathrm{s^{-2}}", label = label)
    v_plot = plot!(v_plot, h_t, v_t, linewidth = linewidth, title = L"\mathbf{Meteoroid}", xaxis = L"\mathrm{Höhe \ in} \ \mathrm{km}", 
        yaxis = L"\mathrm{Fallgeschwindigkeit \ in} \ \mathrm{km} \, \mathrm{s^{-1}}", label = label)
    return a_plot, v_plot
end

# Zeitintervall
tspan = (0.0, 30.0)

# für Plotting
t_min = tspan[1]
t_max = tspan[2]
dt = (t_max - t_min)/10_000
t_interval = t_min:dt:t_max

linewidth = 2

# Interace für Lösung
a(sol, p, t) = combined_acceleration(v(sol, t), h(sol, t), t, p)
v(sol, t) = sol(t)[1]
h(sol, t) = sol(t)[2]

# Pfad zum speichern der Abbildungen
figures_path = "C:/Users/joerg/Documents/Schule/Schule Jahrgang Q2/Physik/Hausarbeit/figures/"

# Plot der Auswirkung der Anfangsgeschwindgkeit

# h_0 in m, v_0 in m/s
sol_1, p_1 = solve_ivp(130e3, -15e3, tspan, 1)
sol_2, p_2 = solve_ivp(130e3, -25e3, tspan, 1)
sol_3, p_3 = solve_ivp(130e3, -35e3, tspan, 1)

label = L"v_0 = %$(round(sol_1[1, 1]/1000, digits=2)) \, \mathrm{km} \, \mathrm{s^{-1}}"; a_plot, v_plot = get_plots(sol_1, p_1, label)
label = L"v_0 = %$(round(sol_2[1, 1]/1000, digits=2)) \, \mathrm{km} \, \mathrm{s^{-1}}"; a_plot, v_plot = get_plots(sol_2, p_2, label, a_plot = a_plot, v_plot = v_plot)
label = L"v_0 = %$(round(sol_3[1, 1]/1000, digits=2)) \, \mathrm{km} \, \mathrm{s^{-1}}"; a_plot, v_plot = get_plots(sol_3, p_3, label, a_plot = a_plot, v_plot = v_plot)

a_plot
v_plot

savefig(a_plot, figures_path * "Meteoroid_1.svg")
savefig(v_plot, figures_path * "Meteoroid_2.svg")

# Plot der Auswirkung der Größe

# h_0 in m, v_0 in m/s
sol_4, p_4 = solve_ivp(130e3, -25e3, tspan, 10^0)
sol_5, p_5 = solve_ivp(130e3, -25e3, tspan, 10^1)
sol_6, p_6 = solve_ivp(130e3, -25e3, tspan, 10^2)

label = L"1V \ \mathrm{und} \ 1m"; a_plot, v_plot = get_plots(sol_4, p_4, label)
label = L"10V \ \mathrm{und} \ 10m"; a_plot, v_plot = get_plots(sol_5, p_5, label, a_plot = a_plot, v_plot = v_plot)
label = L"100V \ \mathrm{und} \ 100m"; a_plot, v_plot = get_plots(sol_6, p_6, label, a_plot = a_plot, v_plot = v_plot)

a_plot
v_plot

savefig(a_plot, figures_path * "Meteoroid_3.svg")
savefig(v_plot, figures_path * "Meteoroid_4.svg")

#=
v_t[end]*1000*3.6 # Endgeschwindigkeit
h_t[end]*1000 # Endhöhe

maximum(a_t)*1000/9.81 # Maximale erdfache Beschleunigung
=#