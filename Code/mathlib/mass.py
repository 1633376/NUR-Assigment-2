import numpy as np


def _assign_mass_NGP(positions):
    """
        Create a mass grid of size 16x16 with 
        the Nearest Grid Point shape.
    In:
        param: positions -- The positions of the N particles as 
                            an matrix of size 2xN. 
    Out:
        return: A matrix representing the mass grid.
    """

    # Create the empty grid
    mass_grid = np.zeros((16,16,16))

    # Add 0.5 and cast to an integer. The positions are now
    # the indices in the grid.
    positions = np.array(positions + 0.5, dtype=np.int32)

    # Assign the masses by looping over the positions
    for i in range(0,len(positions[0])):
        # Mod 16 for cirulair boundaries.
        mass_grid[positions[0][i] % 16, positions[1][i] % 16,positions[2][i] % 16] += 1

    # Return the mass.
    return mass_grid


def _assign_mass_CIC(positions):
    """
        Create a mass grid of size 16x16x16 with 
        the Nearest Grid Point shape.
    In:
        param: positions -- The positions of the N particles as 
                            an matrix of size 3xN. 
    Out:
        return: A matrix representing the mass grid.
    """

    mass_grid = np.zeros((16,16,16))

    for i in range(0, len(positions[0])):

        # The cube below represents the cube made by the 8 grid points that
        # enclose a particle. Next we assume that the sides of are side 
        #    LTU---------RTY
        #    /|          /|
        #   / |         / |
        # LTD----------RTD|
        #  |  |        |  |
        #  |  LBU------|RBU
        #  | /         | /
        #  |/          |/
        # LBD -------RBD
        
        # Let $x_{cell},y_{cell},z_{cell}$ be the position of the left top bottom (LTB) edge of the Grid Cube that
        # encloses the particle. Let x,y,z be the position of the particle. The difference between
        # the particle positon and the LTB edge is then given by,
        # dx = x - x_{cell}
        # dy = y - y_{cell}
        # dz = z - z_{cell}
        # The above quenties can be used to calculate the volumne fractions
        # that needs to be assigned to the 8 nearest grid points of the cube
        # that the particle represents.
        # Working this out (by for example a drawing) gives.
        # 
        # LTD = (1 - dx)*(1 - dy)*(1-dz) 
        # RTD = (1 - dx)*dy*(1 - dz)
        # LTU = (1 - dx)*(1 - dy)*dz
        # RTU = (1 - dx)*dy*dz
        # LBD = dx*(1-dy)*dz 
        # RBD = dx*dy*(1-dz)
        # LBU = dx*(1-dy)*dz
        # RBU = dx*dy*dz

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

        # Calculate the volume fractions, assume unit particle mass

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