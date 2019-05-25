import numpy as np
import matplotlib.pyplot as plt
import mathlib.random as rnd
import mathlib.misc as misc
import time

# Constants. 
grid_size = 1024 # 1024x1024
min_distance = 1 # 0.5 Mpc

# Create the random number generator
random = rnd.Random(1234678)

# The orders in the power spectrum.
n = [-1,-2,-3]

def main():

    # Generate the random uniform numbers that 
    # are later transformed to normal distributed variablaes.
    # The numbers are generated once to reduce computational time.
    random_numbers = random.gen_uniforms(grid_size*grid_size)

    for power in n:

        # Generate the field.
        
        field = misc.generate_hermitian_fs_2D(grid_size, min_distance, 
                                              gen_complex,random_numbers,power)
        # Plot it
        # The field is real, but it is still treated as a complex
        # value this, we have to take the r eal part.
        plt.imshow(np.fft.ifft2(field).real)
        plt.colorbar()
        plt.show()

def gen_complex(k, n, rand1, rand2):
    """
        Generate a complex number using the power
        spectrum.
    In:
        param:k -- The magnitude of the wavenumber.
        param:n -- The order of the power law.

    """
    sigma = 0

    if n == -2:
        sigma = 1/k
    else:
        sigma = np.sqrt(k**n)
    
    a,b = random.gen_normal_uniform(0,sigma,rand1,rand2)
    return complex(a,b)



if __name__ == "__main__":
    main()

