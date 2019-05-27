import mathlib.integrate as integrate
import mathlib.misc as misc
import mathlib.random as rnd
import mathlib.helpers4 as helpers4
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack

random = rnd.Random(783341)

def main():
    assigment4_a()
    assigment4_c()
    assigment4_d()

def gen_complex(k, power, rand1, rand2):
    """
        Generate a complex number using the power
        spectrum.
    In:
        param: k -- The magnitude of the wavenumber.
        param: n -- The order of the power law.
        param: rand 1 -- A random uniform variable.
        param: rand 2 -- A random uniform variable.
    Out:
        return: A complex number in the fourier plane for the
                given power law.
    """
    
    sigma = np.sqrt(k**power)
    a,b = random.gen_normal_uniform(0,1,rand1,rand2)
    return complex(a,-b)*(1/k**2) *0.5*sigma

def assigment4_a():
    # The value of the scale to integrate to
    a_max = 1/51

    # The values of the density parameters
    omega_m = 0.3
    omega_lambda = 0.7

    # The result
    print('D(a = 1/51) = ', helpers4.calculate_linear_growth(a_max, omega_m, omega_lambda)) 

def assigment4_c():

    # Constants and random number generator
    grid_size = 64
    min_distance = 1
    power = -2

    # Create the hermitan matrix and the ifft. 
    # The ifft is used to plot the backgound
    matrix = misc.generate_matrix_2D(grid_size, min_distance, 
                                     gen_complex, 
                                     random.gen_uniforms(grid_size**2*2),power)
    matrix = misc.make_hermitian2D(matrix)
    
    # Calculate the ifft, used for the background of the movie.
    matrix_ifft = scipy.fftpack.ifft2(matrix).real

    # wavenumbers
    wavenumbers = misc.gen_wavenumbers(grid_size, min_distance)

    matrix_y = matrix * wavenumbers*1J
    matrix_x = matrix * wavenumbers[:, np.newaxis]*1J

    # Fix the symmetry that is broken

    size = matrix_x.shape[0]

    matrix_x[int(size/2),0] = matrix_x[int(size/2),0].real
    matrix_x[int(size/2),int(size/2)] = matrix_x[int(size/2),int(size/2)].real

    matrix_y[0,int(size/2)] = matrix_y[0,int(size/2)].real
    matrix_y[int(size/2),int(size/2)] = matrix_y[int(size/2),int(size/2)].real


    for i in range(int(size/2)+1, size):
        matrix_x[int(size/2),i] *= -1 
        matrix_y[i,int(size/2)] *= -1

    # Generate the componentes of the displacemnet vectors
    # We have to transpose/ switch to get the x and y.
    s_x = scipy.fftpack.ifft2(matrix_y).real *grid_size 
    s_y = scipy.fftpack.ifft2(matrix_x).real *grid_size 

    # Positons of particles 
    pos_x, pos_y = np.meshgrid(range(0,grid_size),range(0,grid_size))

    # Begin simulation
    
    # Constants for integration
    a_min = 0.0025
    scale_factors = np.linspace(a_min, 1, 90)

    # List in which the momentum and position
    # of the first 10 particles is saved.
    pos_top_10 = list()
    momentum_top_10 = list() 

    # Create the plot
    for idx,a in enumerate(scale_factors):
    
        # Calculate da    
        da = 0

        if idx != 0:
            da = scale_factors[idx] - scale_factors[idx-1]
        else:
            da = scale_factors[idx] - a_min

        # Calculate D(a) and \dot(D)
        d = helpers4.calculate_linear_growth(a) 
        ddot = helpers4.calculate_linear_growth_dir(a -da/2)

        # Update position.
        pos_x_new = (pos_x+ s_x*d) % grid_size
        pos_y_new = (pos_y+ s_y*d) % grid_size

        pos_top_10.append(pos_y_new[0:10,0])

        # Update momentum.
        p_x = -(a-da/2)**2 * ddot *s_x
        p_y = -(a-da/2)**2 * ddot*s_y

        momentum_top_10.append(p_y[0:10,0])

        # Create the plot.       
        plt.scatter(pos_x_new, pos_y_new,s=1,c='black')
        plt.imshow(matrix_ifft*grid_size,alpha=0.7)
    
        plt.colorbar()
        plt.title('a=' + str(a))
        plt.xlabel('x [Mpc]')
        plt.ylabel('y [Mpc]')
        plt.savefig('./Plots/4c/4c={0}.png'.format(idx))
        plt.close()

    # Plot the momentum and position
    pos_top_10 = np.array(pos_top_10)
    momentum_top_10 = np.array(momentum_top_10)

    # Momentum
    for i in range(0,10):
        plt.plot(scale_factors, momentum_top_10[:,i],label='particle ' + str(i+1))
        plt.xlabel('a')
        plt.ylabel(r'$P_y$')
    plt.legend()
    plt.savefig('./Plots/4c_momentum.pdf')
    plt.close()
        
    # Position
    for i in range(0,10):
        plt.plot(scale_factors, pos_top_10[:,i],label='particle ' + str(i+1))
        plt.xlabel('a')
        plt.ylabel(r'y')

    plt.legend()
    plt.savefig('./Plots/4c_pos.pdf')
    plt.close()

def assigment4_d():

    # Create the matrix
    grid_size = 64
    min_size = 1
    power = -2

    # Create the matrix and give it the correct symmetry
    matrix = misc.generate_matrix_3D(grid_size, min_size, 
                                     gen_complex, 
                                    random.gen_uniforms(grid_size**3 *2),
                                    power)
    matrix = misc.make_hermitian3D(matrix)

    # Wavenumbers and matrices for positions
    wavenumbers = misc.gen_wavenumbers(grid_size,1)

    kx,ky,kz = np.meshgrid(wavenumbers,wavenumbers,wavenumbers)
    matrix_x = matrix*kx*1J
    matrix_y = matrix*ky*1J
    matrix_z = matrix*kz*1J

    # Fix the broken symmetry

    # Fix matrix_x
    for i in range(matrix.shape[0]):
        for j in range(int(grid_size/2)+1, grid_size):
            matrix_x[i,int(grid_size/2),j] *= -1

        if i > int(grid_size/2):
            matrix_x[i,int(grid_size/2),int(grid_size/2)] *=-1 
            matrix_x[i,int(grid_size/2),0] *= -1 
         
    # Special points
    matrix_x[int(grid_size/2), 0, 0] = matrix_x[int(grid_size/2), 0, 0].imag
    matrix_x[0,int(grid_size/2),0] = matrix_x[0,int(grid_size/2),0].imag
    matrix_x[0,0,int(grid_size/2)] = matrix_x[0,0,int(grid_size/2)].imag

    matrix_x[int(grid_size/2), int(grid_size/2),0] = matrix_x[int(grid_size/2), int(grid_size/2),0].imag
    matrix_x[int(grid_size/2), 0, int(grid_size/2)] = matrix_x[int(grid_size/2),0, int(grid_size/2)].imag
    matrix_x[0, int(grid_size/2), int(grid_size/2)] = matrix_x[0, int(grid_size/2), int(grid_size/2)].imag
    matrix_x[int(grid_size/2), int(grid_size/2), int(grid_size/2)] = matrix_x[int(grid_size/2), int(grid_size/2), int(grid_size/2)].imag

    # Fix matrix_z
    for i in range(matrix.shape[0]):
        for j in range(int(grid_size/2)+1, grid_size):
            matrix_z[i,j,int(grid_size/2)] *= -1

        if i > int(grid_size/2):
            matrix_z[i,int(grid_size/2),int(grid_size/2)] *=-1 
            matrix_z[i,0,int(grid_size/2)] *= -1 
        
    # special points
    matrix_z[int(grid_size/2), 0, 0] = matrix_z[int(grid_size/2), 0, 0].imag
    matrix_z[0,int(grid_size/2),0] = matrix_z[0,int(grid_size/2),0].imag
    matrix_z[0,0,int(grid_size/2)] = matrix_z[0,0,int(grid_size/2)].imag

    matrix_z[int(grid_size/2), int(grid_size/2),0] = matrix_z[int(grid_size/2), int(grid_size/2),0].imag
    matrix_z[int(grid_size/2), 0, int(grid_size/2)] = matrix_z[int(grid_size/2),0, int(grid_size/2)].imag
    matrix_z[0, int(grid_size/2), int(grid_size/2)] = matrix_z[0, int(grid_size/2), int(grid_size/2)].imag
    matrix_z[int(grid_size/2), int(grid_size/2), int(grid_size/2)] = matrix_z[int(grid_size/2), int(grid_size/2), int(grid_size/2)].imag


    # Fix matrix y
    for j in range(grid_size):
        for k in range(grid_size):
             
            if (j == 0 or j == int(grid_size/2)) and k > int(grid_size/2):
                matrix_y[int(grid_size/2),j,k] *= -1
            elif  j > int(grid_size/2):
                matrix_y[int(grid_size/2),j,k] *= -1
        
    # special points
    matrix_y[int(grid_size/2),int(grid_size/2),int(grid_size/2)] = matrix_y[int(grid_size/2),int(grid_size/2),int(grid_size/2)].imag + 0J
    matrix_y[int(grid_size/2),int(grid_size/2),0] = matrix_y[int(grid_size/2),int(grid_size/2),0].imag + 0J
    matrix_y[int(grid_size/2),0,int(grid_size/2)] = matrix_y[int(grid_size/2),0,int(grid_size/2)].imag + 0J
    matrix_y[int(grid_size/2),0,0] = matrix_y[int(grid_size/2),0,0].imag + 0J

    # Generate the componentes of the displacemnet vectors
    s_y = scipy.fftpack.ifftn(matrix_y).real * grid_size**(3/2) 
    s_x = scipy.fftpack.ifftn(matrix_x).real * grid_size**(3/2)
    s_z = scipy.fftpack.ifftn(matrix_z).real * grid_size**(3/2)

    
    # Positions
    pos_x, pos_y, pos_z = np.meshgrid(range(0,grid_size),range(0,grid_size), range(0, grid_size))
   
    # 
    # Begin simulation

    # Constants for integration
    a_min = 1/51
    scale_factors = np.linspace(a_min,1,90,endpoint=False)

    # List in which the momentum and position
    # of the first 10 particles is saved.
    pos_top_10 = list()#.zeros((10,10))
    momentum_top_10 = list() #np.zeros((90,10))

    # Create the plot
    for idx,a in enumerate(scale_factors):
    
        # calculate da    
        da = 0

        if idx != 0:
            da = scale_factors[idx] - scale_factors[idx-1]
        else:
            da = scale_factors[idx] - a_min

        # Calculate D(a)
        # Calculate \dot(D)
        d = helpers4.calculate_linear_growth(a) 
        ddot = helpers4.calculate_linear_growth_dir(a -da/2)

        # Update position
        pos_x_new = (pos_x + s_x*d) % grid_size
        pos_y_new = (pos_y + s_y*d) % grid_size
        pos_z_new = (pos_z + s_z*d) % grid_size

        pos_top_10.append(pos_z_new[0:10,0,0])

        # Update momentum
        p_x = -(a-da/2)**2 * ddot *s_x
        p_y = -(a-da/2)**2 * ddot*s_y
        p_z = -(a-da/2)**2 * ddot*s_y

        momentum_top_10.append(p_z[0:10,0,0])
           
        # slices
        z_mask = np.where(abs(pos_z_new - 32) < 0.5) 
        y_mask = np.where(abs(pos_y_new - 32) < 0.5)
        x_mask = np.where(abs(pos_x_new - 32) < 0.5)

        # Create the plots. 

        # z
        plt.scatter(pos_x_new[z_mask], pos_y_new[z_mask],s=1,c='black')
        plt.title('a=' + str(a))
        plt.xlabel('x [Mpc]')
        plt.ylabel('y [Mpc]')
        plt.savefig('./Plots/4d/xy/4d_xy={0}.png'.format(idx))
        plt.close() 

        # y
        plt.scatter(pos_x_new[y_mask], pos_z_new[y_mask],s=1,c='black')
        plt.title('a=' + str(a))
        plt.xlabel('x [Mpc]')
        plt.ylabel('z [Mpc]')
        plt.savefig('./Plots/4d/xz/4d_xz={0}.png'.format(idx))
        plt.close()

        # x
        plt.scatter(pos_y_new[x_mask], pos_z_new[x_mask],s=1,c='black')
        plt.title('a=' + str(a))
        plt.xlabel('y [Mpc]')
        plt.ylabel('z [Mpc]')
        plt.savefig('./Plots/4d/yz/4d_yz={0}.png'.format(idx))
        plt.close()

    # Plot the momentum and position
    pos_top_10 = np.array(pos_top_10)
    momentum_top_10 = np.array(momentum_top_10)

    # Momentum
    for i in range(0,10):
        plt.plot(scale_factors, momentum_top_10[:,i], label = 'particle ' + str(i+1))
        plt.xlabel('a')
        plt.ylabel(r'$P_z$')
        
    plt.legend()
    plt.savefig('./Plots/4d_momentum.pdf')
    plt.close()

    # Position
    for i in range(0,10):
        plt.plot(scale_factors, pos_top_10[:,i], label = 'particle ' + str(i+1))
        plt.xlabel('a')
        plt.ylabel('z')

    plt.legend()
    plt.savefig('./Plots/4d_pos.pdf')
    plt.close()



if __name__ == "__main__":
    main()