import mathlib.integrate as integrate
import mathlib.misc as misc
import mathlib.random as rnd
import matplotlib.pyplot as plt
import numpy as np

random = rnd.Random(123456)

def main():


    assigment4_a()
    assigment4_c()
    


def assigment4_a():

    # The value of the scale to integrate to
    a_max = 1/51

    # The values of the density parameters
    omega_m = 0.3
    omega_lambda = 0.7


    # The result
    print('D(a = 1/51) = ', calculate_scale(1e-7,a_max,omega_m,omega_lambda)) # integrate.romberg(func, 1e-7, a_max, 10)*pre_fac)

def assigment4_b():
    pass

def assigment4_c():

    # Constants and random number generator
    random = rnd.Random(12345678)
    grid_size = 64
    min_distance = 1
    power = -2

    # Create the hermitan matrix and the ifft. 
    # The ifft is used to plot the backgound
    matrix = misc.generate_hermitian_fs_2D(grid_size, min_distance, gen_complex, random.gen_uniforms(grid_size**2), power)
    matrix_ifft = np.fft.ifft2(matrix).real

    #P
    # raise ""

    #
    # Generate the displacemnet vectors
    #

    # Generate a mesh grid for the x and y  component

    wavenumbers = misc.gen_wavenumbers(grid_size, min_distance)

    kx, ky = np.meshgrid(wavenumbers,wavenumbers)
    matrix_x = matrix *kx *1J
    matrix_y = matrix *ky *1J

    # Give the matrices the correct symmetry
    matrix_x = misc.make_hessian2D(matrix_x)
    matrix_y = misc.make_hessian2D(matrix_y)

    # Generate the componentes of the displacemnet vectors
    s_x = np.fft.ifft2(matrix_x).real.flatten()
    s_y = np.fft.ifft2(matrix_y).real.flatten()
    #
    # Generate the  grid with particles
    # 

   # # positons and momentum
    pos_x, pos_y = np.meshgrid(range(0,grid_size),range(0,grid_size))
    pos_x = pos_x.flatten()
    pos_y = pos_y.flatten()
    pos_x_clone = pos_x.copy()
    pos_y_clone = pos_y.copy()

    p_x, p_y = np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size))

    # 
    # Begin simulation
    #
    # Movie is 30 frames per second and should take 3 sec, thus 90 frames.
    
    # The scale values
    a_min = 0.0025
    scale_factors = np.linspace(a_min,1,90)


    # We need to plot the position of the first 10 particles
    # thsus save those
    pos_top_10 = np.zeros((90,10))

    # Create the plot
    for idx,a in enumerate(scale_factors):
        
        # calculate D(a)
        d = calculate_scale(1e-7,a) #((1/a)-1)

        # update position
        pos_x = pos_x_clone + s_x*d
        pos_y = pos_y_clone + s_y*d

      #  pos_top_10[idx] = np.sqrt(pos_x[0:10]**2 + pos_y[0:10]**2)

        pos_x = pos_x % (grid_size - 1) # mod 64 always gives a positive value!!
        pos_y = pos_y % (grid_size - 1)
#        print(np.max(pos_x))

   
        plt.scatter(pos_x, pos_y,s=1,c='black')
        plt.imshow(matrix_ifft,alpha=0.7)
        plt.savefig('./Plots/4c/4c={0}.png'.format(idx))
        plt.close()

   # for i in range(0,10):
   #    plt.plot(scale_factors,pos_top_10[0:,i])#np.sqrt(pos_x**2+pos_y**2))
   # plt.ylabel('Position (magnitude)')
  #  plt.xlabel('Scale factor: a')
  #  plt.show()


   


def gen_complex(k, power, rand1, rand2):
    """
        Generate a complex number using the power
        spectrum.
    In:
        param:k -- The magnitude of the wavenumber.
        param:n -- The order of the power law.
    """
    

    sigma = np.sqrt(k**power)
    a,b = random.gen_normal_uniform(0,1,rand1,rand2)
    return complex(a,-b)*(1/k**2) *0.5*sigma


def calculate_scale(a_min, a_max, omega_m = 0.3, omega_lambda = 0.7):
    

    if a_min == a_max:
        return 0

    # The prefactor of the integral
    pre_factor = 0.5*(5*omega_m)*(omega_m*a_max**(-3) + omega_lambda)**(0.5)

    # The function to integrate.
    func = lambda a: a**(-3) /(omega_m * (1/(a**3))+omega_lambda)**(3/2) 

    return pre_factor*integrate.romberg(func,a_min,a_max,10)


if __name__ == "__main__":
    main()