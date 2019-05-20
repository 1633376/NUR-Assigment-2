import numpy as np



def generate_gaussian_random_field_assigm4(power, size, min_distance, random):
    """
        Generate a gaussian random field in fourier space.
    In:
        param: power -- A function used to evaluate the power for
                        a given wavenumber.
        param: size -- The size of the grid.
    """

    field_matrix = np.zeros((size,size),dtype=complex)
    wave_numbers = _gen_shiffted_wavenumbers((2*np.pi/min_distance), (2*np.pi/(min_distance*size)), size)

    for row in range(1, int(size/2)+1):

        # Determine the k-value for the edges
        k = np.sqrt(2*wave_numbers[row]**2)
        pre_factor = np.sqrt(power(k))
        z = complex(random.gen_normal(0,1),random.gen_normal(0,1))*pre_factor

        # Set the value for the first row and column
        field_matrix[size-row, 0] = z
        field_matrix[0, size-row] = z

        # Make sure the first row and first colum have the correct symmetry
        field_matrix[row, 0] = complex(z.real, -z.imag)
        field_matrix[0, row] = complex(z.real, -z.imag)
        
        # Go over the inner matrix and make sure that it has the right symmetry
        for column in range(1, size):

            # Find the value of k and create the complex number
            k = np.sqrt(2*wave_numbers[row]**2)
            pre_factor = np.sqrt(power(k))
            z = complex(random.gen_normal(0,1),random.gen_normal(0,1))*pre_factor

            # Set the complex value in the inner matrix and make sure that the symmetry is correct.
            field_matrix[size - row, size -column] = z
            field_matrix[row,column] = complex(z.real, -z.imag)

    # If the matrix is even, set the imaginary part of the columns correpsonding with a niquest
    # wavenumber to zero.
    if size % 2 == 0:
        for i in range(0,size):
            field_matrix[int(size/2),i] = field_matrix[int(size/2),i].real + 0J
            field_matrix[i,int(size/2)] = field_matrix[i,int(size/2)].real + 0J

    return field_matrix

    
def _gen_shiffted_wavenumbers(k_max, k_min, size):
    """
        Generate an array with wavenumbers for fourier 
        shifted coordinates by lineair spacing between the maximum 
        and mininimum wavenumber.

        In:
            param: k_max -- The maximum wavenumber.
            param: k_min -- The minimum wavenumber.
            param: size -- The size that the array should have.
        Out:
            return: An array with wavenumbers in fourier shifted coordinates such that
                    the first element corresponds with the minimum wavenumber.

        Example(s):
            k_max = 3, k_min = 1, size = 4
            [1, 2, 3,-2]
            
            k_max = 4, k_min = 1, size = 5 
            [1, 2, 3, -3, -2]
                
    """
    ret = np.zeros(size)

    if size % 2 == 0: # even
        # positive part: k_max should be included, linear space therefore size/2 + 1 
        ret[0:int(size/2)+1] = np.linspace(k_min, k_max, int(size/2)+1)
        # negative part: inverse array , multiply with 1 and select elements to repeat
        ret[int(size/2)+1:] = -ret[::-1][int(size/2):-1]
   
    else: # odd
        # positive part: k_max shouldn't be included, linear space therefore size/2 +2 and skip last element
        ret[0:int(size/2)+1] = np.linspace(k_min, k_max, int(size/2)+2)[:-1]
        # negative p artL inverse array, mulitply with 1 and select correct elements
        ret[int(size/2)+1:] = -ret[::-1][int(size/2):-1]
    
    return ret


def generate_displacements2D(field, min_distance):


    # create a matrix where the wavenumbers are the diogonal.

    wavenumbers = _gen_shiffted_wavenumbers((2*np.pi/min_distance), (2*np.pi/(min_distance*field.shape[0])), field.shape[0])
    diogonal = np.zeros(field.shape)

    for i in range(0, field.shape[0]):
        diogonal[i][i] = wavenumbers[i]

    # The product of the diogonal matrix generates the displacement matrix to ifft when multiplied with the complex part

    final = field*diogonal * (-1J)
    print(np.fft.ifft2(final))

    return np.fft.ifft2(final)


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



size = 20
min_distance = 1
max_distance = size*min_distance
field_matrix = np.zeros((size,size,size),dtype=complex)

wave_numbers  = _gen_shiffted_wavenumbers((2*np.pi/min_distance), (2*np.pi/max_distance), size)

for row, k_x in enumerate(wave_numbers):
    for column, k_y in enumerate(wave_numbers):
        for depth, k_z in enumerate(wave_numbers):
            
            if row == 0 and column == 0 and depth == 0:
                continue # zero variance

            k = np.sqrt(k_x**2 + k_y**2 + k_z**2)
            sigma = np.sqrt(k**(-2))
            field_matrix[row, column, depth] = complex(np.random.normal(0,sigma), np.random.normal(0,sigma))

symmetrix = make_hessian3D(field_matrix*1J)
#print(symmetrix)
print(np.fft.ifftn(symmetrix))