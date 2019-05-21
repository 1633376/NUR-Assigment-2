import numpy as np
import mathlib.random as rnd


def gen_wavenumbers(size, min_distance):
    """
        Generate the fourier transform sample wavenumbers
        for the discrete fourier transform.
    In:
        param: size -- The size of the matrix.
        param: min_distance -- The distance of a cell/ the sample spacing.
    
    """

    # Array to return.
    ret = np.zeros(size)

    # Positive values
    ret[0:int(size/2)+1] = np.arange(0,int(size/2)+1)

    if size  % 2 == 0: # even
        ret[int(size/2):] = -np.arange(int(size/2),0,-1)
    else: # odd
        ret[int(size/2)+1:] = -np.arange(int(size/2),0,-1)
  

    return ret/(size*min_distance)

def generate_hermitian_fs_2D(size, min_distance, func, random_numbers,power = 2):
    """
        Generate a hermitan matrix in shifted fourier space.
    In:
        param: size -- The size of the matrix.
        param: min_distance -- The sample spacing.
        param: func -- A function that given the magnitude of 
                       the k-vector produces a complex number that
                       is placed inside the matrix.
        param: power -- The power to use for the power law.
        param: random_numers -- An optional array with random numbers.
    Out:
        return: A hemritian symmetric matrix in shifted fourier coordinates
    """           

    # The matrix to return.
    matrix = np.zeros((size,size),dtype=complex)

    # Create the wave numbers for fourier shifted coordinates.
    wave_numbers = gen_wavenumbers(size, min_distance)

    # Counter for random numbers
    random_num_counter = 0

    # Create the matrix
    for row in range(1, int(size/2)+1):

        # Determine the k-value for the edges and the complex number
        k = np.sqrt(2*wave_numbers[row]**2)
        z = func(k,power, random_numbers[random_num_counter],random_numbers[random_num_counter+1])
        random_num_counter+= 2
       
        # Set the value for the first row and column
        matrix[size-row, 0] = z
        matrix[0, size-row] = z

        # Make sure the first row and first colum have the correct symmetry
        matrix[row, 0] = complex(z.real, -z.imag)
        matrix[0, row] = complex(z.real, -z.imag)
        
        # Go over the inner matrix and make sure that it has the right symmetry
        for column in range(1, size):

            # Find the value of k and create the complex number
            k = np.sqrt(wave_numbers[row]**2 + wave_numbers[column]**2)
            z = func(k, power,random_numbers[random_num_counter],random_numbers[random_num_counter+1])
            random_num_counter += 2
            
            # Set the complex value in the inner matrix and make sure that the symmetry is correct.
            matrix[size - row, size -column] = z
            matrix[row,column] = complex(z.real, -z.imag)

    # If the matrix is even, set the imaginary part of the columns correpsonding with a niquest
    # wavenumber to zero.
    if size % 2 == 0:
        for i in range(0,size):
            matrix[int(size/2),i] = matrix[int(size/2),i].real + 0J
            matrix[i,int(size/2)] = matrix[i,int(size/2)].real + 0J

    return matrix


def make_hessian2D(matrix):

    size = matrix.shape[0]

    for row in range(1, int(size/2) +1):
        
        # First row and first column, points (1) and (2)
        # correct first column, point (2)
        matrix[row,0] = complex(matrix[size-row,0].real, - matrix[size-row,0].imag)
        # current first row, point (1)
        matrix[0, row] = complex(matrix[0, size-row].real,- matrix[0,size- row].imag)

        # inner matrix (point 3)
        for column in range(1, size):
            # point (3)
            matrix[row, column] = complex(matrix[size-row, size-column].real, -matrix[size-row, size-column].imag)
            
    # point 4
    if size % 2 == 0:
        for i in range(0,size):
            matrix[int(size/2),i] = matrix[int(size/2),i].real + 0J
            matrix[i,int(size/2)] = matrix[i,int(size/2)].real + 0J
    
    return matrix


def make_hessian3D(tensor):

    size = tensor.shape[0]

    # Inner matrix
    for row in range(1, size): #int(size/2) + size-3):
        for column in range(1, size): #int(size/2)+size-3):
            for depth in range(1,size):               
                tensor[row, column, depth] = complex(tensor[size-row,size-column,size-depth].real, -tensor[size-row, size-column, size-depth].imag)
                
    # Outside top
    for depth in range(1,int(size/2)+1):
        # left top
        tensor[0,0,depth] = complex(tensor[0,0,size-depth].real, -tensor[0,0,size-depth].imag)

        # front top
        tensor[0,depth,0] = complex(tensor[0,size-depth,0].real, -tensor[0,size-depth,0].imag)

        # bottom
        tensor[depth,0,0] = complex(tensor[size-depth,0,0].real, -tensor[size-depth,0,0].imag)


         # outside inner (point 3)
        for column in range(1, size):
            # front
            tensor[depth, column,0] = complex(tensor[size-depth, size-column,0].real, -tensor[size-depth, size-column,0].imag)
            # size 
            tensor[depth,0,column] = complex(tensor[size-depth,0, size-column].real, -tensor[size-depth,0, size-column].imag)
            # top
            tensor[0, depth, column] = complex(tensor[0,size-depth,size-column].real, - tensor[0,size-depth,size-column].imag)


    if size % 2 == 0:
        # Finally fix niquist
        for i in range(0,size):
            for j in range(0,size):
                tensor[int(size/2),i,j] = tensor[int(size/2),i,j].real + 0J
                tensor[i,j,int(size/2)] = tensor[i,j,int(size/2)].real + 0J
                tensor[i,int(size/2),j] = tensor[i,int(size/2),j].real + 0J
    
    return tensor
