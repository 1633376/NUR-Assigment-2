import numpy as np
import matplotlib.pyplot as plt
import mathlib.mass as mass
import mathlib.fourier as fourier
import mathlib.misc as misc

def main():
    assigment_5a()
    assigment_5b()
    assigment_5c()
    assigment_5d()
    assigment_5e()

def assigment_5a():
    """
        Execute assignment 5.a
    """

    # Relevant imports are:
    # (1) import numpy as np
    # (2) import mathlib.mass as mass
    # (3) import matplotlib.pyplot as plt

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
        plt.close() 
      
def assigment_5b():
    """
        Execute assignment 5.b
    """

    # Relevant imports are:
    # (1) import numpy as np
    # (2) import mathlib.mass as mass
    # (3) import matplotlib.pyplot as plt

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
    plt.close()

def assigment_5c():
    """
        Execute assigment 5.c
    """

    # Relevant imports are:
    # (1) import numpy as np
    # (2) import mathlib.mass as mass
    # (3) import matplotlib.pyplot as plt


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
        plt.close()

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
    plt.close()

def assigment_5d():
    """
        Execute assignment 5.d
    """

    # Relevant imports are:
    # (1) import numpy as np
    # (2) import matplotlib.pyplot as plt
    # (3) import mathlib.misc as misc
    # (3) import mathlib.fourier as fourier

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
    """
        Execute assigment 5e
    """

    # Relevant imports are:
    # (1) import numpy as np
    # (2) import matplotlib.pyplot as plt
    # (3) import mathlib.misc as misc
    # (3) import mathlib.fourier as fourier

    # The function to 2D fourier transform
    func_2d = lambda x,y: np.cos(x+y)
    x,y = np.meshgrid(range(0, 64), range(0,64))
    x = np.array(x, dtype=complex)
    y = np.array(y, dtype=complex)

    _, ax = plt.subplots(nrows=1,ncols=2)

    # Self written FFT2
    ax[0].imshow(abs(fourier.fft2(func_2d(x,y))))
    ax[0].set_title("Self written FFT2")
    ax[0].set_xlabel('index')
    ax[0].set_ylabel('index')

    # Numpy FFT
    ax[1].imshow(abs(np.fft.fft2(func_2d(x,y))))
    ax[1].set_title("Numpy version FFT2")
    ax[1].set_xlabel('index')
    ax[1].set_ylabel('index')
    
    plt.savefig('./Plots/5e_2d_fft.pdf',bbox_inches='tight')
    plt.close()

    # 3D
    # The multivariate gaussian.
    # I am fully aware that this line looks ugly in the report.
    multivariate_sigma = lambda x,y,z, sigma:  (1.0/(sigma**2*2*np.pi)**(3/2)) * np.exp(-(((x**2)/2) + ((y**2)/2) + ((z**2)/2))/sigma**2)

    # The multivariate to plot, sigma = 0.5
    multivariate = lambda x,y,z: multivariate_sigma(x,y,z, 0.5)

    # Plot the multivariate gaussian
    N = 64
    values = np.arange(0,N)
    values = np.array(values,dtype=complex)
    x,y,z = np.meshgrid(values,values,values)
    
    matrix = multivariate(x,y,z)
    matrix_fft = fourier.fft3(matrix)

    # y-z slice
    plt.imshow(abs(matrix_fft[int(N/2)]))
    plt.xlabel('y')
    plt.ylabel('z')
    plt.savefig('./Plots/5e_gaussian_yz.pdf')
    plt.close()

    # x-z slice
    plt.imshow(abs(matrix_fft[:,int(N/2),:]))
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig('./Plots/5e_gaussian_xz.pdf')
    plt.close()
    
    # y-x slice
    plt.imshow(abs(matrix_fft[:,:,int(N/2)]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('./Plots/5e_gaussian_xy.pdf')
    plt.close()

if __name__ == "__main__":
    main()


