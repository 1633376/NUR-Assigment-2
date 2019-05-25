import numpy as np
import matplotlib.pyplot as plt

def assigment_5a():

    np.random.seed(121)
    positions = np.random.uniform(low =0, high=16, size=(3,1024))

    # Create the mass grid.
    mass_grid = _assign_mass_NGP(positions)

    # Plot the slices

    for z in [4,9,11,14]:
        slice = mass_grid[:,:,z]
        slice[slice == 0] = np.nan

        plt.matshow(slice,vmin=0)
        plt.colorbar()
        plt.show() 
      
def assigment_5b():


    # x positions of the particles
    x_values = np.linspace(0, 16, 1000)
    # for each position createt he mass grid and get the value in cell 4
    y_values = [_assign_mass_NGP(np.array([[x],[0],[0]]))[4,0,0] for x in x_values]
    y_values_0 =  [_assign_mass_NGP(np.array([[x],[0],[0]]))[0,0,0] for x in x_values]

    # plot the positions
    plt.plot(x_values, y_values, label='Cel 4')
    plt.plot(x_values, y_values_0, label='Cel 5')
    plt.xlabel('Positon')
    plt.ylabel('Mass')
    plt.legend()
    plt.show()

def assigment_5c():

    np.random.seed(121)
    positions = np.random.uniform(low =0, high=16, size=(3,1024))

    # Create the mass grid.
    mass_grid = _assign_mass_CIC(positions)

    # Plot the slices

    for z in [4,9,11,14]:
        slice = mass_grid[:,:,z]
        slice[slice == 0] = np.nan

        plt.matshow(slice,vmin=0)
        plt.colorbar()
        plt.show() 


    # Part 5b
    # x positions of the particles
    x_values = np.linspace(0, 16, 1000)
    y_values = [_assign_mass_CIC(np.array([[x],[0],[0]]))[4,0,0] for x in x_values]
    y_values_0 =  [_assign_mass_CIC(np.array([[x],[0],[0]]))[0,0,0] for x in x_values]

    # plot the positions
    plt.plot(x_values, y_values, label='Cel 4')
    plt.plot(x_values, y_values_0, label='Cel 0')
    plt.xlabel('Positon')
    plt.ylabel('Mass')
    plt.legend()
    plt.show()
    pass


def _assign_mass_NGP(positions):

    mass_grid = np.zeros((16,16,16))
    positions = np.array(positions + 0.5, dtype=np.int32)

    for i in range(0,len(positions[0])):
        mass_grid[positions[0][i] % 16, positions[1][i] % 16,positions[2][i] % 16] += 1

    return mass_grid


def _assign_mass_CIC(positions):

    mass_grid = np.zeros((16,16,16))

    for i in range(0, len(positions[0])):

        # Get the indices of the grid point at the left bottom of 
        # the cel in which the particle is
        center_x = np.int32(positions[0][i]) % 16
        center_y = np.int32(positions[1][i]) % 16
        center_z = np.int32(positions[2][i]) % 16

        # Helper variables to calculate the
        # volume fractions. Mode 16 for circulair boundary conditions
        dx = positions[0][i] % 16 - center_x
        dy = positions[1][i] % 16 - center_y
        dz = positions[2][i] % 16 - center_z

        frac_dx = 1 - dx
        frac_dy = 1 - dy
        frac_dz = 1 - dz

        # Calculate the volume fractions
        #    LTU---------RTY
        #    /|          /|
        #   / |         / |
        # LTD----------RTD|
        #  |  |        |  |
        #  |  LBU-------RBU
        #  | /         |/
        #  |/          /
        # LBD -------RBD

        # Left top down
        mass_grid[center_x][center_y][center_z] += frac_dx*frac_dy*frac_dz
        # Right top down
        mass_grid[center_x][(center_y+1) % 16][center_z] += frac_dx*dy*frac_dz
        # Left top up
        mass_grid[center_x][center_y][(center_z+1) % 16] += frac_dx*frac_dy*dz 
        # Right top up
        mass_grid[center_x][(center_y+1) % 16][(center_z+1) % 16] += frac_dx*dy*dz

        #Left bottom down
        mass_grid[(center_x+1) % 16][center_y][center_z] += dx*frac_dy*frac_dz
        # Right bottom down
        mass_grid[(center_x+1) % 16][(center_y+1) % 16][center_z] += dx*dy*frac_dz
        # Left bottom up
        mass_grid[(center_x+1) % 16][center_y][(center_z+1) % 16] += dx*frac_dy*dz
        # Right bottom up
        mass_grid[(center_x+1) % 16][(center_y+1) % 16][(center_z+1) % 16] += dx*dy*dz


    return mass_grid
        

assigment_5a()
#assigment_5b()
assigment_5c()


