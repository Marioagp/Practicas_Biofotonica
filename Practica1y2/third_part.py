import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import first_part as fp

#LONGITUD DE ONDA 670 NM
mu_a_670 = [0.0032, 0.0080, 0.0029, 0.0071] # coeficiente de absorcion paciente 9 y 11 (background y target) respectivamente
mu_s_prime_670 = [0.94, 1.48, 1.2, 1.76 ] #  coeficiente de scattering reducido paciente 9 y 11 (background y target) respectivamente

# variables
N = 1000  # numeros de elementos en los arrays
r = np.linspace(0, 0.5, N)  # distancia radial


# figura longitud de onda 670 nm
fig1, ax11 = plt.subplots(nrows=1, ncols=1)
nom = ["sano Paci.9","tumoral Paci.9",  "sano Paci.11","tumoral Paci.11"]

# variacion de mu_a entre 0.01 y 4 cm^-1
for i in range(len(mu_a_670)):
    Phi = fp.diffusion_approx(r, mu_s_prime_670[i], mu_a_670[i])
    ax11.semilogy(r, Phi, label=nom[i])

ax11.set_xlabel('r [cm]')
ax11.set_ylabel('Razón de fluencia' + r' $\Phi$ [W/cm^2]')
ax11.set_title('Tasa de fluencia para para tejido mamario sano y tumoral a 670nm')
ax11.legend()
ax11.grid()


#LONGITUD DE ONDA 785 NM
mu_a_670 = [0.0028, 0.0055, 0.0024, 0.0042] # coeficiente de absorcion paciente 9 y 11 (background y target) respectivamente
mu_s_prime_670 = [0.83, 1.15, 1.1, 1.6 ] #  coeficiente de scattering reducido paciente 9 y 11 (background y target) respectivamente


# figura longitud de onda 785 nm
fig2, ax11 = plt.subplots(nrows=1, ncols=1)

# variacion de mu_a entre 0.01 y 4 cm^-1
for i in range(len(mu_a_670)):
    Phi = fp.diffusion_approx(r, mu_s_prime_670[i], mu_a_670[i])
    ax11.semilogy(r, Phi, label=nom[i])

ax11.set_xlabel('r [cm]')
ax11.set_ylabel('Razón de fluencia' + r' $\Phi$ [W/cm^2]')
ax11.set_title('Tasa de fluencia para para tejido mamario sano y tumoral a 785nm')
ax11.legend()
ax11.grid()