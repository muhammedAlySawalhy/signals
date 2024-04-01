import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def exponential_function_sin(x):
    return np.exp(np.sin(x))

def exponential_function_cos(x):
    return math.exp(math.cos(x))

def trapezoidal_rule(a, b, n, input_function):
    h = (b - a) / n
    result = (input_function(a) + input_function(b)) / 2.0
    for i in range(1, n):
        result += input_function(a + i * h)
    result *= h
    return result

def lib_res(a, b, input_function):
    library_result, error = integrate.quad(input_function, a, b)
    return library_result

def fourier_coefficient(a, b, n, k, f):
    h = (b - a) / n
    coeff = 0
    for i in range(n):
        x_i = a + i * h
        x_next = a + (i + 1) * h
        coeff += f(x_i) * math.sin((k * math.pi * x_i) / (b - a)) * h
        coeff += f(x_next) * math.sin((k * math.pi * x_next) / (b - a)) * h
    return (2 / (b - a)) * coeff

def fourier_series(a, b, n, N, f):
    coefficients = []
    for k in range(N):
        coefficients.append(fourier_coefficient(a, b, n, k, f))
    return coefficients

def fourier_series_coefficients(a, b, N, f):
    num_samples = 10000  # Increase the number of samples
    x = np.linspace(a, b, num_samples)
    y = f(x)
    fourier_coeffs = np.fft.fft(y) / num_samples
    return fourier_coeffs[:N].real

if __name__ == '__main__':
    a, b, n = map(int, input().split())

    trapozodial = trapezoidal_rule(a, b, n, exponential_function_sin)
    lib_result = lib_res(a, b, exponential_function_sin)

    print(trapozodial, "calculated by trapezoidal rule")
    print(lib_result, "calculated by library")


    # Number of Fourier coefficients to calculate
    N = 10

    # Calculate Fourier series coefficients without library
    coefficients = fourier_series(a, b, n, N, exponential_function_sin)

    # Calculate Fourier series coefficients using library
    coefficients_library = fourier_series_coefficients(a, b, N, exponential_function_sin)

    # Plot the coefficients
    plt.plot(range(N), coefficients, label='Without Library')
    plt.plot(range(N), coefficients_library, label='With Library')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Coefficient Value')
    plt.title('Comparison of Fourier Series Coefficients')
    plt.legend()
    plt.show()

    # Calculate absolute differences between corresponding coefficients
    differences = np.abs(np.array(coefficients) - np.array(coefficients_library))

    # Print the differences
    print("Absolute differences between coefficients:", differences)

    # Check if the maximum difference is within a tolerance level
    tolerance = 1e-6
    if np.max(differences) < tolerance:
        print("The coefficients are approximately equal within the tolerance level.")
    else:
        print("The coefficients are not equal within the tolerance level.")
