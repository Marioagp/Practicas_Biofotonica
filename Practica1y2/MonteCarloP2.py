import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  ## progressbar
import first_part as fg

if __name__ == '__main__':
    mu_a = 4.0  # Coeficiente de absorción (cm-1) # Error relativo del 18.65%
    mu_s_prime = 20.0  # Coeficiente de scattering reducido (cm-1)
    r = np.linspace(0, 0.5, 101)  # Puntos en los que evaluar la función de green y el Monte Carlo (cm)


def monte_carlo_diff(r, mu_s_prime, mu_a):
    microns_per_shell = (r[1] - r[0]) * 1e4  # Tamaño por delta r (µm)
    # microns_per_shell = 50
    photons = 1000  # Fotones a lanzar
    N_bins = r.shape[0]  # Puntos a evaluar
    # N_bins= 101

    ''' Variables derivadas de las anteriores'''
    heat = np.zeros(N_bins)  # Tasa de absorción de fotones en el medio / deposición específica (en W/m^3)
    albedo = mu_s_prime / (mu_s_prime + mu_a)  # Albedo: relación entre coeficiente de scattering y de transporte
    # cambio a  cm
    shells_per_mfp = 1e4 / microns_per_shell / (
                mu_s_prime + mu_a)  # Número de bins por camino medio libre de interacciones

    '''Main loop'''
    for i in tqdm(range(0, photons), desc='MC 1D'):
        r_ph_list = []  # Lista en la que guardar las posiciones

        # Lanzamiento de un único paquete
        r_ph = np.array([0.0, 0.0, 0.0])  # Posición inicial
        nu = np.array([0.0, 0.0, 1.0])  # Orientación hacia z positivas
        power = 1.0  # Potencia inicial del paquete: Cuando llega a cero, la simulación termina
        # y el fotón ha muerto

        # Vamos a ver como se distribuye esta potencia por el espacio a medida que el paquete
        # se va propagando
        alive = True  # Estudiamos el movimiento del fotón mientras no se extinga:
        # Inicialmente vivo
        while alive:
            '''Cálculo del movimiento relativo al camino libre de transporte'''
            step = -np.log(np.random.rand(1))  # ver el wang: Biomedical optics chapter 3
            r_ph += step * nu  # Añadimos a la posición original el salto en la orientación nu

            r_ph_list.append(r_ph.copy())  # Añadimos el paso nuevo a la lista
            r_ph_array = np.asarray(r_ph_list)
            # Estimamos a qué distancia nos encontramos del origen
            shell = np.linalg.norm(r_ph) * shells_per_mfp

            # Como hemos puesto 101 valores de R, tendremos 101 bins que van de 0 a 100
            # asi que vamos a dejar el último bin para acumular residuos de la simulación,
            # de forma que si nos salimos al calcular el shell lo metemos en el bin 100
            # y usamos los bins de 0 a 99 para calcular.
            if shell > N_bins - 1:
                shell = N_bins - 1

            '''Cambio de orientación del paquete de fotones'''
            nu = np.random.randn(3)  # Genera 3 números aleatorios gaussianos de mu=0, sigma=1
            nu = nu / np.linalg.norm(nu)  # División por la norma

            '''Absorción parcial del paquete de fotones'''
            # Para saber qué parte de nuestro paquete de fotones se absorbe calculamos la
            # tasa de absorción
            heat[int(shell)] += (1.0 - albedo) * power  # (1-albedo) es la fracción absorbida por el
            # el medio, el resto de la potencia sigue
            # viajando
            power *= albedo

            '''Ruleta rusa para eliminar fotones débiles'''
            # El método permite, de media, que 9 de los 10 fotones te los cargues
            # Sin embargo, el que sobreviva continúa con 10 veces más peso para conservar la
            # energía total
            if (power < 0.001):
                if (np.random.rand(1) > 0.1):
                    alive = False  # fin del bucle while
                power /= 0.1

    out_W = []  # Salida del MC

    for i in range(0, N_bins):
        # Posición, en centímetros
        # out_rW.append(i*microns_per_shell*1E-4)
        # Constante:
        # 4* pi * (um/shell)^3 (1/um^3) * 1E-12 (1/cm^3)*r^2     r^2=(i+DELTAr+1/2DELTAr)^2
        # Calculamos el volumen en el centro entre n y n+1 de un shell esférico de grosor deltar
        # V=4pi*r^2*deltar
        shell_volume = 4 * np.pi * (microns_per_shell ** 3) * (i ** 2 + i + 1 / 4.) * 1E-12
        # Deposición de potencia específica [W/cm^3]
        out_W.append(heat[i] / (shell_volume * photons))  # El paso dividido entre el número de fotones a simular

    return out_W


# ----------------------------------------------- Punto 1 --------------------------------------------------
if __name__ == '__main__':
    MC = monte_carlo_diff(r, mu_s_prime, mu_a)
    FG = fg.diffusion_approx(r, mu_s_prime, mu_a)

    fig1 = plt.figure(
        'Comparacion de la Green y MC')  # Comparacion montecarlo-función de green (punto 1 de la práctica)
    ax11 = fig1.add_subplot(1, 1, 1)
    ax11.semilogy(r[:-1], MC[:-1], '.k', alpha=0.5, label='Simulación de Monte Carlo')
    ax11.plot(r[:-1], mu_a * FG[:-1], label='Aproximación de difusión')
    ax11.grid(True)
    ax11.set_xlabel('Distancia [cm]')
    ax11.set_ylabel(r'Deposición de potencia específica [W/cm$^3$]')
    ax11.set_title('Comparación de Monte Carlo con la aproximación de difusión')
    plt.tight_layout()
    ax11.legend()
