using OrdinaryDiffEq
using Plots
using LaTeXStrings

# Werte für einen LC-Reihenschwingkreis mit Widerstand in Reihe (RLC-Schaltung)
R = 10. # Ω
L = 100.e-3 # H 
C = 100.e-6 # F

# Resonanzfrequenz
f = 1/(2*pi*sqrt(L*C)) # Hz

# Eingangsspannung und ihre Ableitung
U_in(t) = 20*sin(2*pi*f*t) + 20*sin(2*pi*5*f*t)
dU_in(t) = 20*cos(2*pi*f*t)*2*pi*f + 20*cos(2*pi*5*f*t)*2*pi*5*f

# Anfangswerte
I_0 = 0. # A
dI_0 = 0. # A/s

# Zeitintervall
tspan = (0., 5*1/f) # s

# gedämpfter Oszillator
function ddI(ddI, dI, I, p, t)
    R, L, C = p
    ddI .= @. -(R/L)*dI - (1/(L*C))*I - (1/L)*dU_in(t)
end

prob = SecondOrderODEProblem(ddI, [dI_0], [I_0], tspan, [R, L, C])
sol = solve(prob, Tsit5(), reltol = 1e-7, abstol = 1e-7)

# Interface für Lösung
I(t) = sol(t)[2]
dI(t) = sol(t)[1]

# Ausgangsspannungen für Bandpass und Bandsperre
U_bp(t) = I(t)*R # band-pass filter
U_br(t) = -I(t)*R - U_in(t) # band-reject filter
# Summe aller Teilspannungen in der Masche
U_sum(t) = U_bp(t) + U_br(t) + U_in(t)

t_min = tspan[1]
t_max = tspan[2]
dt = (t_max - t_min)/10_000
t_interval = t_min:dt:t_max
plot(
    plot(t_interval, U_bp, label="U_bp(t)", xlabel="t in s", ylabel="U in V"),
    plot(t_interval, U_br, label="U_br(t)", xlabel="t in s", ylabel="U in V", color=2),
    plot(t_interval, U_in, label="U_in(t)", xlabel="t in s", ylabel="U in V", color=3),
    layout=(3, 1),
    dpi=400
)
savefig("LC_Reihenkreis_Filter.png")
# plot(t_interval, U_sum, label="U_sum(t)", yrange=(-1, 1), xlabel="t in s", ylabel="U in V")

plot(t_interval, U_bp, label="U_bp(t) = U_R(t)", xlabel="t in s", ylabel="U in V")
plot!(t_interval, U_in, label="U_in(t)", xlabel="t in s", ylabel="U in V")
plot!(t_interval, t -> -U_bp(t) - U_in(t), label="U_br(t) = -U_R(t) - U_in(t)", xlabel="t in s", ylabel="U in V")

# Plotting für Export

# Pfad zum speichern der Abbildungen
figures_path = "C:/Users/joerg/Documents/Schule/Schule Jahrgang Q2/Physik/Hausarbeit/figures/"
plot(
    # plot_title = "LC-Reihenschwingkreis als Filter",
    plot_title = L"\mathbf{LC-Reihenschwingkreis \ als \ Filter}",
    plot(t_interval, U_bp, linewidth = 2, label=L"U_{Bp}(t) = U_R(t)", xlabel=L"t \ \mathrm{in \ s}", ylabel=L"U \ \mathrm{in \ V}"),
    plot(t_interval, U_br, linewidth = 2, label=L"U_{Bs}(t) = U_{LC}(t)", xlabel=L"t \ \mathrm{in \ s}", ylabel=L"U \ \mathrm{in \ V}", color=2),
    plot(t_interval, U_in, linewidth = 2, label=L"U_{in}(t)", xlabel=L"t \ \mathrm{in \ s}", ylabel=L"U \ \mathrm{in \ V}", color=3),
    layout=(3, 1)
)
savefig(figures_path * "LC_Reihenschwingkreis_Filter.svg")