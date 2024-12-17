# Compute the Fourier transforms for delta(t - 5) and delta(t - 10)
import numpy as np
import matplotlib.pyplot as plt

F_omega_5_real = np.exp(-1j * 5 * omega).real
F_omega_5_imag = np.exp(-1j * 5 * omega).imag
F_omega_10_real = np.exp(-1j * 10 * omega).real
F_omega_10_imag = np.exp(-1j * 10 * omega).imag

# Plot the real and imaginary parts of both Fourier transforms
plt.figure(figsize=(10, 6))

# Real and Imaginary parts of delta(t - 5)
plt.subplot(2, 1, 1)
plt.plot(omega, F_omega_5_real, label="Real part of δ(t - 5)", color="blue")
plt.plot(omega, F_omega_5_imag, label="Imaginary part of δ(t - 5)", color="orange")
plt.title("Fourier Transform of δ(t - 5)")
plt.xlabel("ω (Frequency)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Real and Imaginary parts of delta(t - 10)
plt.subplot(2, 1, 2)
plt.plot(omega, F_omega_10_real, label="Real part of δ(t - 10)", color="green")
plt.plot(omega, F_omega_10_imag, label="Imaginary part of δ(t - 10)", color="red")
plt.title("Fourier Transform of δ(t - 10)")
plt.xlabel("ω (Frequency)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

