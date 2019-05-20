import numpy as np
import matplotlib.pyplot as plt
import mathlib.random as rnd
import time

# Constants. 
grid_size = 4 # 1024x1024
min_distance = 0.5 # 0.5 Mpc
max_distance = min_distance*grid_size # Mpc

# Create the random number generator
random = rnd.Random(12345678)

def main():
    field = _gen_field(grid_size,-3)
   # print(field)
    print(np.fft.ifft2(field))
   # plt.imshow(np.fft.ifft2(field).real)
   # plt.show()

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


def _gen_field2(size, n):
    a = time.time()
    field_matrix = np.zeros((size,size),dtype=complex)
    wave_numbers = _gen_shiffted_wavenumbers((2*np.pi/min_distance), (2*np.pi/max_distance), size)

    for row in range(1, int(size/2)+1):

        # Determine the k-value for the edges
        k = np.sqrt(2*wave_numbers[row]**2)
        sigma = np.sqrt(_power(k,n))
        z = complex(random.gen_normal(0,sigma),random.gen_normal(0,sigma))#np.random.normal(0,sigma))

        # Set the value for the first row and column
        field_matrix[size-row, 0] = z
        field_matrix[0, size-row] = z

        # Make sure the first row and first colum have the correct symmetry
        field_matrix[row, 0] = complex(z.real, -z.imag)
        field_matrix[0, row] = complex(z.real, -z.imag)
        
        # Go over the inner matrix and make sure that it has the right symmetry
        for column in range(1, size):

            # Find the value of k and create the complex number
            k = np.sqrt(wave_numbers[row]**2 + wave_numbers[column]**2)
            sigma = np.sqrt(_power(k,n))
            z = complex(np.random.normal(0,sigma),np.random.normal(0,sigma))

            # Set the complex value in the inner matrix and make sure that the symmetry is correct.
            field_matrix[size - row, size -column] = z
            field_matrix[row,column] = complex(z.real, -z.imag)

    # If the matrix is even, set the imaginary part of the columns correpsonding with a niquest
    # wavenumber to zero.
    if size % 2 == 0:
        for i in range(0,size):
            field_matrix[int(size/2),i] = field_matrix[int(size/2),i].real + 0J
            field_matrix[i,int(size/2)] = field_matrix[i,int(size/2)].real + 0J
    print(time.time()-a)
    return field_matrix

def _gen_field(size,n):

    field_matrix = np.zeros((size,size),dtype=complex)
    wave_numbers = _gen_shiffted_wavenumbers((2*np.pi/min_distance), (2*np.pi/max_distance), size)

    # Fill the matrix at each position with a random normal value and make sure 
    # the symmetry is correct

    a = time.time()
    for row, k_x in enumerate(wave_numbers):
        for column, k_y in enumerate(wave_numbers):
            
            if row == 0 and column == 0:
                continue # zero variance

            k = np.sqrt(k_x**2 + k_y**2)
            sigma = np.sqrt(_power(k,n))
            field_matrix[row, column] = complex(np.random.normal(0,sigma), np.random.normal(0,sigma))
        
    
    # Make sure that the matrix obese the given symmety: H(k_x,k_y) = conj(H(-k_x, -k_y)).
    # The cel with wavenumbers k-x and k_y should be the conjjugate of the cel with wavenumbers -k_x, -k_y
    # Let z_{ij} denote a complex number in row i and column j of the matrix.
    # Then the required symmetry can be obtained by performing the following operation:
    #
    # (1) Adjusting the first row to be: [z_(0,0), z_(0,1), z_(0,2), conj(z_(0,2)), conj(z_(0,1)]
    # (2) Ajust the first column to be: [z_(0,0), z_(1,0),z_(2,0), conj(z_(2,0), conj(z_(1,0))]
    # (3) Adjusting the inner matrix, this comes down of making z_{1+a, 1+b} = conj(z_{N-a, N-b}) where a is from 0 to N-1 and b from 1 to N
    # (4) For the symmetric case not all elements in the previous steps have pairs, in this case put the comple part to zero.
    #     The values without pair are in column N/2 for all rows and in row N/2 for all columns. This corresponds with the Nequist wavelengts




    for row in range(1, int(size/2) +1):
        
        # First row and first column, points (1) and (2)

        # correct first column, point (2)
        field_matrix[row,0] = complex(field_matrix[size-row,0].real, -field_matrix[size-row,0].imag)
        # current first row, point (1)
        field_matrix[0, row] = complex(field_matrix[0, size-row].real,- field_matrix[0,size- row].imag)

        # inner matrix (point 3)
        for column in range(1, size):

            # point (3)
            field_matrix[row, column] = complex(field_matrix[size-row, size-column].real, -field_matrix[size-row, size-column].imag)
            
    # point 4
    if size % 2 == 0:
        for i in range(0,size):
            field_matrix[int(size/2),i] = field_matrix[int(size/2),i].real + 0J
            field_matrix[i,int(size/2)] = field_matrix[i,int(size/2)].real + 0J
    print(time.time()-a)
    return field_matrix            


def _power(k,n):
    return k**n


if __name__ == "__main__":
    main()

