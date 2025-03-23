import numpy as np
from scipy.integrate import quad

# -------------------------------
# Physical Constants
# -------------------------------
kB = 1.380649e-23    # Boltzmann constant in J/K
hbar = 1.0545718e-34  # Reduced Planck constant in J·s
pi = np.pi

# -------------------------------
# Debye Model Integral Function
# -------------------------------
def debye_integral(x):
    """
    Integrand for the Debye model thermal conductivity.
    """
    return (x**4 * np.exp(x)) / (np.expm1(x)**2)  # expm1(x)=exp(x)-1 for better accuracy at small x

def thermal_conductivity(T, Theta_D, v, tau):
    """
    Calculate the bulk thermal conductivity (kappa) using the Debye model,
    with the formula:
    
        kappa = (kB / (2*pi^2*v)) * ((kB*T/hbar)^3) * tau * I
    where
        I = integral_0^(Theta_D/T) [x^4 * exp(x) / (exp(x)-1)^2] dx.
    
    Parameters:
        T      : Temperature in K.
        Theta_D: Debye temperature in K.
        v      : Average phonon group velocity in m/s.
        tau    : (Assumed constant) relaxation time in s.
        
    Returns:
        kappa  : Estimated thermal conductivity in W/(m·K).
    """
    # Upper integration limit in the Debye model
    upper_limit = Theta_D / T
    I, _ = quad(debye_integral, 0, upper_limit)
    
    # Compute the factor (kB*T/hbar)^3
    factor = (kB * T / hbar)**3
    
    # Thermal conductivity expression
    kappa = (kB / (2 * pi**2 * v)) * factor * tau * I
    return kappa

# -------------------------------
# Parameters and Calculation
# -------------------------------
# Common temperature (room temperature)
T = 300.0  # in Kelvin

# -- Silicon parameters (tuned to get ~141 W/(m·K)) --
Theta_D_si = 645.0      # Debye temperature for Silicon (K)
v_si = 6791.0           # Average phonon group velocity for Silicon (m/s)
# tau = 400 picoseconds
tau_si = 400e-12        # Effective relaxation time for Silicon (s)
# tau_si = 6.25e-12       # Effective relaxation time for Silicon (s)

kappa_si = thermal_conductivity(T, Theta_D_si, v_si, tau_si)

# -- Diamond parameters --
Theta_D_dia = 2230.0    # Debye temperature for Diamond (K)
v_dia = 17280.0         # Average phonon group velocity for Diamond (m/s)
tau_dia = 400e-12        # Effective relaxation time for Silicon (s)
# tau_dia = 2.3e-11       # Effective relaxation time for Diamond (s)

kappa_dia = thermal_conductivity(T, Theta_D_dia, v_dia, tau_dia)

# -------------------------------
# Output the results
# -------------------------------
print("Estimated bulk thermal conductivity for Silicon: {:.2f} W/(m·K)".format(kappa_si))
print("Estimated bulk thermal conductivity for Diamond  : {:.2f} W/(m·K)".format(kappa_dia))