import numpy as np
import matplotlib.pyplot as plt
import first_part as fg
import MonteCarloP2 as mcp

mu_a = np.linspace(0.1, 4, 10)  # Coeficiente de absorción (cm-1)
mu_s_prime = 10.0  # Coeficiente de scattering reducido (cm-1)
r = np.linspace(0, 0.5, 101)  # Puntos en los que evaluar la función de green y el Monte Carlo (cm)

Er = []

for i in range(len(mu_a)):
    MC = mcp.monte_carlo_diff(r, mu_s_prime, mu_a[i])
    FG = fg.diffusion_approx(r, mu_s_prime, mu_a[i])
    Er.append(100 * np.mean(np.abs(mu_a[i] * FG[1:] - MC[1:]) / np.abs(MC[1:])))

fig2 = plt.figure(
    'Diferencia de la Aproximación de Green y el MC(Error Relativo %)')  # Comparacion montecarlo-función de green (punto 1 de la práctica)
ax112 = fig2.add_subplot(1, 1, 1)
ax112.plot(mu_a, Er)
ax112.plot(mu_a, Er, 'bo')
ax112.grid(True)
ax112.set_xlabel(r'$\mu_a$ en [cm^-1]')
ax112.set_ylabel('Error Relativo [%]')
ax112.set_title('Diferencia de la Aproximación de Green y el MC(Error Relativo %)')
plt.tight_layout()
plt.axhline(y=5, color='r', linestyle='--')

for i in range(len(mu_a)):
    ax112.text(mu_a[i], Er[i]+2, str(np.round(Er[i])), ha="center")

plt.ylim(0, 26)


