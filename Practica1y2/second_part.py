import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


import first_part as fp

mu_s = 0.05  # coeficiente de scattering
g = 0.85  # coeficiente de anisotropia
mu_s_prime_f = mu_s * (1 - g)  # coeficiente de scattering reducido

# variables
N = 1000  # numeros de elementos en los arrays
r = np.linspace(0.5, 1, N)  # distancia radial
mu_a = np.linspace(0, 4, 10)  # coeficiente de absorcion

# figura
fig1, ax11 = plt.subplots(nrows=1, ncols=1)

# variacion de mu_a entre 0.01 y 4 cm^-1
for i in range(len(mu_a)):
    Phi = fp.diffusion_approx(r, mu_s_prime_f, mu_a[i])
    ax11.semilogy(r, Phi, label=r'$\mu_a$=' + f'{np.round(mu_a[i], 2)}')

ax11.set_xlabel('r [cm]')
ax11.set_ylabel('Razón de fluencia' + r' $\Phi$ [W/cm^2]')
ax11.set_title('Tasa de fluencia para diferentes radios y diferentes ' + r'$\mu_a$')
ax11.legend()
ax11.grid()

# variacion de mu_s_prima entre 0.01 y 4 cm^-1
mu_s_prime = np.linspace(20, 40, 10)  # coeficiente de scattering
mu_a_f = 2.0

fig2, ax11 = plt.subplots(nrows=1, ncols=1)

for i in range(len(mu_s_prime)):
    Phi = fp.diffusion_approx(r, mu_s_prime[i], mu_a_f)
    ax11.semilogy(r, Phi, label=r'$\mu\prime_s$=' + f'{np.round(mu_s_prime[i], 2)}')

ax11.set_xlabel('r [cm]')
ax11.set_ylabel('Razón de fluencia' + r' $\Phi$ [W/cm^2]')
ax11.set_title('Tasa de fluencia para diferentes radios y diferentes ' + r'$\mu\prime_s$')
ax11.legend()
ax11.grid()

# variacion de mu_s_prima y mu_a 3D plot
mu_a, mu_s_prime = np.meshgrid(mu_a, mu_s_prime)
r = 0.5# distancia radial

Phi = fp.diffusion_approx(r, mu_s_prime, mu_a)

fig3, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(mu_a, mu_s_prime, Phi, cmap=cm.coolwarm,
                       linewidth=0)

ax.zaxis.set_major_formatter('{x:.02f}')
ax.set_xlabel(r'$\mu_a$')
ax.set_ylabel(r'$\mu\prime_s$')
ax.set_zlabel('Razón de fluencia' + r' $\Phi$ [W/cm^2]')


