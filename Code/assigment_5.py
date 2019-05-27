import numpy as np
import matplotlib.pyplot as plt
import mathlib.mass as mass
import mathlib.fourier as fourier
import mathlib.misc as misc

def main():
    #assigment_5a()
    #assigment_5b()
    #assigment_5c()
    assigment_5d()
    #Passigment_5e()

def assigment_5a():
    """
        Execute assignment 5.a
    """

    # Create the positions of the particles.
    np.random.seed(121)
    positions = np.random.uniform(low =0, high=16, size=(3,1024))

    # Create the mass grid.
    mass_grid = mass._assign_mass_NGP(positions)

    # Plot the slices
    for z in [4,9,11,14]:
        slice = mass_grid[:,:,z]
        #slice[slice == 0] = np.nan

        plt.imshow(slice)
        plt.colorbar()
        plt.savefig('./Plots/5a_slice_{0}.pdf'.format(z))
        plt.figure() 
      
def assigment_5b():
    """
        Execute assignment 5.b
    """

    # The x positions of the moving particle
    x_values = np.linspace(0, 16, 1000)

    # The y positions of the moving particle for 
    # cell 0 and 4
    y_values_4 = list()
    y_values_0 = list()

    # For each position create the mass grid and get the value in cell 4 and 0
    for x in x_values:
        grid = mass._assign_mass_NGP(np.array([[x],[0],[0]]))
        y_values_0.append(grid[0,0,0])
        y_values_4.append(grid[4,0,0])
   
    # plot the positions
    plt.plot(x_values, y_values_4, label='Cell 4')
    plt.plot(x_values, y_values_0, label='Cell 0')
    plt.xlabel('Positon x')
    plt.ylabel('Mass in terms of particle mass')
    plt.legend()
    plt.savefig('./Plots/5b_cell.pdf')
    plt.figure()

def assigment_5c():
    """
        Execute assigment 5.c
    """

    np.random.seed(121)
    positions = np.random.uniform(low =0, high=16, size=(3,1024))

    # Create the mass grid.
    mass_grid = mass._assign_mass_CIC(positions)

    # Plot the slices.
    for z in [4,9,11,14]:
        slice = mass_grid[:,:,z]
        #slice[slice == 0] = np.nan

        plt.imshow(slice)
        plt.colorbar()
        plt.savefig('./Plots/5c_slice_{0}.pdf'.format(z))
        plt.figure()

    # The x positions of the moving particle
    x_values = np.linspace(0, 16, 1000)

    # The y positions of the moving particle for 
    # cell 0 and 4
    y_values_4 = list()
    y_values_0 = list()

    # For each position create the mass grid and get the value in cell 4 and 0
    for x in x_values:
        grid = mass._assign_mass_CIC(np.array([[x],[0],[0]]))
        y_values_0.append(grid[0,0,0])
        y_values_4.append(grid[4,0,0])
   
    # plot the positions
    plt.plot(x_values, y_values_4, label='Cell 4')
    plt.plot(x_values, y_values_0, label='Cell 0')
    plt.xlabel('Positon x')
    plt.ylabel('Mass in terms of particle mass')
    plt.legend()
    plt.savefig('./Plots/5c_cell.pdf')
    plt.figure()

def assigment_5d():
    """
        Execute assignment 5.d
    """

    # Create the data to fourier transform.
    size = 64

    t = np.linspace(0, size, size)
    f = 5
    y = np.cos(2*np.pi*f*t)
    

    # Execute the FFT
    fft_np = np.fft.fft(y)
    fft_self = fourier.fft(y)

    # Frequencies to plot for.
    freq = misc.gen_wavenumbers(size,1)*size

    plt.plot(freq, abs(fft_np), label='numpy',linestyle=':', zorder=1.1)
    plt.plot(freq,abs(fft_self), label='self',zorder=1.0)
    plt.vlines(-2*np.pi*f,0,max(abs(fft_self))+10,label='Analytical')
    plt.vlines(2*np.pi*f,0,max(abs(fft_self))+10)

    plt.xlabel('Frequence')
    plt.ylabel('Power')
    plt.legend()
    plt.savefig('./Plots/5d_fourier.pdf')
    plt.close()


def assigment_5e():

    # The gaussian and multivariate gaussian for sigma = 1 and mu = 0
    gaus = lambda x : 1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2)
    multivariate = lambda x,y,z: gaus(x)*gaus(y)*gaus(z)


    # Plot the multivariate gaussian
    N = 128
    values = np.arange(0,N)
    values = np.array(values,dtype=complex)
    x,y,z = np.meshgrid(values,values,values)
    
    matrix = multivariate(x,y,z)
    matrix_fft = fourier.fft3(matrix)

    # y-z slice
    plt.imshow(abs(matrix_fft[int(128/2)]))
    plt.xlabel('y')
    plt.ylabel('z')
    plt.show()

    # x-z slice
    plt.imshow(abs(matrix_fft[:,int(128/2),:]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # y-x slice
    plt.imshow(abs(matrix_fft[:,:,int(128/2)]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    #print(multivariate(x,y,z))

    #numpy_fft = np.fft3()
  
    pass

if __name__ == "__main__":
    main()


