using OrdinaryDiffEq
using Plots
using LaTeXStrings

# Werte für einen LC-Parallelschwingkreis mit Widerstand in Reihe (RLC-Schaltung)
R = 100. # Ω
L = 100.e-3 # H 
C = 100.e-6 # F

L/(10*C)

# Resonanzfrequenz
f = 1/(2*pi*sqrt(L*C)) # Hz

# Eingangsspannung und ihre Ableitung
U_in(t) = 20*sin(2*pi*f*t) + 20*sin(2*pi*5*f*t)
dU_in(t) = 20*cos(2*pi*f*t)*2*pi*f + 20*cos(2*pi*5*f*t)*2*pi*5*f

# Anfangswerte
U_0 = 0. # V
dU_0 = 0. # V/s

# Zeitintervall
tspan = (0., 5*1/f) # s

# gedämpfter Oszillator
function ddU(ddU, dU, U, p, t)
    R, L, C = p
    ddU .= @. -(1/(R*C))*dU - (1/(L*C))*U - (1/(R*C))*dU_in(t)
end

prob = SecondOrderODEProblem(ddU, [dU_0], [U_0], tspan, [R, L, C])
sol = solve(prob, Tsit5(), reltol = 1e-7, abstol = 1e-7)

# Interface für Lösung
U(t) = sol(t)[2]
dU(t) = sol(t)[1]

# Ausgangsspannungen für Bandpass und Bandsperre
U_bp(t) = U(t) # band-pass filter
U_br(t) = -U(t) - U_in(t) # band-reject filter
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
savefig("LC_Parallelkreis_Filter.png")
# plot(t_interval, U_sum, label="U_sum(t)", yrange=(-1, 1), xlabel="t in s", ylabel="U in V")

plot(t_interval, U, label="U_bp(t) = U(t)", xlabel="t in s", ylabel="U in V")
plot!(t_interval, U_in, label="U_in(t)", xlabel="t in s", ylabel="U in V")
plot!(t_interval, t -> -U(t) - U_in(t), label="U_br(t) = -U(t) - U_in(t)", xlabel="t in s", ylabel="U in V")

# Plotting für Export

# Pfad zum speichern der Abbildungen
figures_path = "C:/Users/joerg/Documents/Schule/Schule Jahrgang Q2/Physik/Hausarbeit/figures/"
plot(
    # plot_title = "LC-Parallelschwingkreis als Filter",
    plot_title = L"\mathbf{LC-Parallelschwingkreis \ als \ Filter}",
    plot(t_interval, U_bp, linewidth = 2, label=L"U_{Bp}(t) = U(t)", xlabel=L"t \ \mathrm{in \ s}", ylabel=L"U \ \mathrm{in \ V}"),
    plot(t_interval, U_br, linewidth = 2, label=L"U_{Bs}(t) = U_R(t)", xlabel=L"t \ \mathrm{in \ s}", ylabel=L"U \ \mathrm{in \ V}", color=2),
    plot(t_interval, U_in, linewidth = 2, label=L"U_{in}(t)", xlabel=L"t \ \mathrm{in \ s}", ylabel=L"U \ \mathrm{in \ V}", color=3),
    layout=(3, 1)
)
savefig(figures_path * "LC_Parallelschwingkreis_Filter.svg")