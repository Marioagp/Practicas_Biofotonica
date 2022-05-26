import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mu_a = 4.0 #coeficiente de absorcion
    mu_s_prime = 20.0 # coeficiente de scattering reducido
    # Number of photons and bins
    N = 1000
    r = np.linspace(0, 0.5, N) #distancia radial

def diffusion_approx(r,mu_s_prime,mu_a):
    D = 1.0 / (3 * (mu_a + mu_s_prime)) #coeficiente de difusion
    mu_eff = np.sqrt(mu_a / D)    #coeficiente de atenuacion (o transporte) efectivo
    Phi = (1 / (4 * np.pi * D * r)) * np.exp(-mu_eff * r)  # Función de Green 3D
    return Phi

if __name__ == '__main__':
    fig1 = plt.figure(1)
    ax11 = fig1.add_subplot(1,1,1)
    ax11.semilogy(r,diffusion_approx(r,mu_s_prime,mu_a), label=r'$\mu_a$=4.0'+'\n'+r'$\mu\prime_s$=20.0')
    ax11.set_xlabel('r [cm]')
    ax11.set_ylabel('Razón de fluencia'+r' $\Phi$ [W/cm^2]' )
    ax11.set_title('Tasa de fluencia (Modelado de Green 3D) para diferentes radios')
    ax11.legend()
    ax11.grid()




