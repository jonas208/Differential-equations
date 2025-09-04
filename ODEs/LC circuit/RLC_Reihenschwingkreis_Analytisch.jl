using OrdinaryDiffEq
using Plots
using LaTeXStrings

L = 100.e-3 # H 
C = 100.e-6 # F
U_0 = 20. # V

# Resonanzfrequenz
ω_0 = 1/sqrt(L*C)
f_0 = ω_0/(2*pi) # Hz

# Schwingfall
# R = 10. # Ω
# aperiodischer Grenzfall
# R = 2*L*ω_0
# Kriechfall
R = 100.

# Anfangswerte
Q_0 = U_0 * C # C
I_0 = 0. # A

# gedämpfter Oszillator
function ddQ(ddQ, I, Q, p, t)
    R, L, C = p
    ddQ .= @. -(R/L)*I - (1/(L*C))*Q
end

# Zeitintervall
tspan = (0., 5*1/f_0) # s

prob = SecondOrderODEProblem(ddQ, [I_0], [Q_0], tspan, [R, L, C])
sol = solve(prob, Tsit5(), reltol = 1e-7, abstol = 1e-7)

# Interface für Lösung
Q(t) = sol(t)[2]
I(t) = sol(t)[1]

# Spannung am Kondensator
U_C(t) = Q(t)/C

t_min = tspan[1]
t_max = tspan[2]
dt = (t_max - t_min)/10_000
t_interval = t_min:dt:t_max
plot(
    plot(t_interval, I, label="I(t)", xlabel="t in s", ylabel="I in A"),
    plot(t_interval, U_C, label="U_C(t)", xlabel="t in s", ylabel="U in V", color=2),
    layout=(2, 1),
    dpi=400
)

# Analytische Lösung

δ = R/(2*L)
if δ^2 < ω_0^2 # Schwingfall
    println("Schwingfall")
    ω = sqrt(ω_0^2 - δ^2)
    Φ = atan(-1/ω*(I_0/Q_0 + δ))
    Q_max = Q_0/cos(Φ)
    Q_exact(t) = Q_max*exp(-δ*t)*cos(ω*t + Φ)
elseif δ^2 == ω_0^2 # aperiodischer Grenzfall
    println("aperiodischer Grenzfall")
    Q_exact(t) = exp(-δ*t)*(Q_0 + (I_0 + δ*Q_0)*t)
else # Kriechfall
    println("Kriechfall")
    Q_exact(t) = exp(-δ*t)*(Q_0*cosh(sqrt(δ^2 - ω_0^2)*t) + (I_0 + δ*Q_0)/(sqrt(δ^2 - ω_0^2))*sinh(sqrt(δ^2 - ω_0^2)*t))
end

@info isapprox(Q.(t_interval), Q_exact.(t_interval), rtol = 1e-7, atol = 1e-7)

plot(t_interval, Q, label="Q(t)", xlabel="t in s", ylabel="Q in C")
plot!(t_interval, Q_exact, label="Q_exact(t)", xlabel="t in s", ylabel="Q in C")

# Plotting für Export

# Pfad zum speichern der Abbildungen
figures_path = "C:/Users/joerg/Documents/Schule/Schule Jahrgang Q2/Physik/Hausarbeit/figures/"

Is = I.(t_interval)
U_Cs = U_C.(t_interval)
limit_factor = 1.2

plot(t_interval, I, linewidth = 2, label=L"I(t)", legend=:topleft, title = L"\mathbf{RLC-Reihenschwingkreis}",
        xaxis=L"t \ \mathrm{in \ s}", yaxis=L"I \ \mathrm{in \ A}", ylims = [-limit_factor*maximum(abs.(Is)), limit_factor*maximum(abs.(Is))])
plot!(twinx(), t_interval, U_C, linewidth = 2, label=L"U_C(t)", yaxis=L"U \ \mathrm{in \ V}", 
        ylims = [-limit_factor*maximum(abs.(U_Cs)), limit_factor*maximum(abs.(U_Cs))], color = 2)
savefig(figures_path * "RLC_Reihenschwingkreis_3.svg")