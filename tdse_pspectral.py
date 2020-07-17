import numpy as np
import matplotlib.pyplot as plt

# real space/momentum space grid
n = 256 * 32
l = 800 
x = np.linspace(0, l, n)
dx = float(l) / n
kx = 2.0 * np.pi * np.fft.fftfreq(n, dx)

# initial wavefunction
psi = 2 * np.array(
    1 * np.exp(-((x - l / 3 / 2) ** 2) / 30 ** 2 + 0.75j * x), dtype=np.complex
)
particle_mass = 1

# potential barrier
V = np.zeros(n, dtype=np.complex)
V[np.floor(0.6 / 2 * n).astype(int) : np.ceil(0.605 / 2 * n).astype(int)] = 2.3

psi_hat = np.fft.fft(psi)
Vhat = np.fft.fft(V)

# timestepping parameters
num_steps = 50000
delta_t = 0.005

for c in range(num_steps):

    fac = np.exp((-1j * kx ** 2 * delta_t) / particle_mass)
    psi_hat = fac * (psi_hat + delta_t * np.fft.fft(-1j * V * psi))
    psi = np.fft.ifft(psi_hat)

    if c % 250 == 0:
        print(c)
        plt.plot(x, psi.real)
        plt.plot(x, psi.imag)
        plt.plot(x, V.real)
        plt.ylim([-3.5, 3.5])
        plt.xlim([0, l / 2])
        plt.savefig("figs/" + str(c).zfill(len(str(num_steps))) + ".png")
        plt.clf()
