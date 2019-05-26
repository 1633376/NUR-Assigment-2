import numpy as np

def gen_wavenumbers(size, min_distance):
    """
        Generate the shifted wavenumbers
        for the discrete fourier transform.
    In:
        param: size -- The size of the matrix.
        param: min_distance -- The distance of a cell/ the sample spacing.
    Out:
        return: An array with shifted wave numbers.
    """
    # Array to return.
    ret = np.zeros(size)

    # Positive values
    ret[0:int(size/2)+1] = np.arange(0,int(size/2)+1)

    if size  % 2 == 0: # even
        ret[int(size/2):] = -np.arange(int(size/2),0,-1)
    else: # odd
        ret[int(size/2)+1:] = -np.arange(int(size/2),0,-1)
  

    return (ret/(size*min_distance))*2*np.pi


def generate_matrix_2D(size, min_distance, func, random_numbers, power):
    """
        Generate a 2D matrix with complex numbers in 
        shifted fourier coordinates using the power spectrum
    In:
        param: size -- The size of the matrix (size x size).
        param: min_distance -- The physical size of 1 cell.
        param: func -- A functiion that takes the power and to random uniform
                      variables to calculate the correct complex number.
        param: random_numbers -- An array with random uniform numbers. 
                                 Must be of atleast size: size x size x 2.
        param: power -- The power of the power spectrum to create the matrix for.
    Out:
        return: A 2D matrix with complex numbers assigned by the power spectrum
                in fourier shifted coordinates.
    """
    
    # Generate the shifted wavenumbers
    wavenumber = gen_wavenumbers(size,min_distance)
    # The matrix to return
    ret = np.zeros((size,size),dtype=complex)

    # A counter for the random uniform variables.
    steps = 0

    # Fill the matrix
    for i in range(size):
        for j in range(size):

            # Element of k_0,k_0 is left zero.
            if i == 0 and j == 0:
                continue
            
            # Calculate the magnitude of the wavenumbers.
            k = np.sqrt(wavenumber[i]**2 + wavenumber[j]**2)
            # Fill the matrix.
            ret[i][j] = func(k, power, 
                            random_numbers[steps], random_numbers[steps+1])
            steps += 2

    # Return the matrix
    return ret

def make_hermitian2D(matrix):
    """
        Give a matrix in shifted fourier coordinates
        the correct hermitian symmetry so that the ifft is real.
    In:
        param: matrix -- The matrix to give the correct symmetry.
    Out:
        return: A matrix with the correct hermitan symmetry so that the 
                ifft is real.
    """

    # The size of the matrix
    size = matrix.shape[0]

    # Loop over the rows
    for row in range(1, int(size/2) +1):
        
        # Give the first column (index 0) has the correct symmetry (see report point A)
        matrix[row,0] = complex(matrix[size-row,0].real,
                              - matrix[size-row,0].imag)
        # Give the first row (index 0) the correct symmetry (see report point B)
        matrix[0, row] = complex(matrix[0, size-row].real,
                             - matrix[0,size- row].imag)

        # Give the inner matrix the correct symmetry (see report point C)
        for column in range(1, size):
            matrix[row, column] = complex(matrix[size-row, size-column].real,
                                         -matrix[size-row, size-column].imag)
    
    # Corrections for even matrix
    if size % 2 == 0:
        matrix[int(size/2), 0] = matrix[int(size/2), 0].real + 0J
        matrix[0, int(size/2)] = matrix[0, int(size/2)].real + 0J
        matrix[int(size/2), int(size/2)] = matrix[int(size/2), int(size/2)].real + 0J
                 
    # Return the matrix.
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
