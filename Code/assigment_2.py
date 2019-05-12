import numpy as np
import matplotlib.pyplot as plt

def _gen_shiffted_wavenumbers(k_max, size):
    

    ret = np.zeros(size)      

    if size % 2 == 0:
        ret[0: int(size/2) + 1] = np.linspace(0,k_max,int(size/2)+1)
        ret[int(size/2)+1:] = np.linspace(-k_max,0,int(size)/2+1)[1:-1] # skip first element
    else:
        ret[0:int(size/2)+1] = np.linspace(0,k_max,int(size/2)+1)
        ret[int(size/2)+1:] = np.linspace(-k_max,0,int(size)/2+1)[:-1]

    
    return ret


def _gen_field(k_max,size,n):

    field_matrix = np.zeros((size,size),dtype=complex)
    wave_numbers = _gen_shiffted_wavenumbers(k_max, size)

    # Fill the matrix at each position with a random normal value

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

    return field_matrix            


def _power(k,n):
    return k**n

np.random.seed(542)
field = _gen_field(100,1024,-2)
print(np.mean(field))
print(field)
print(np.fft.ifft2(field))
#print(np.mean(field))
#print(np.fft.ifft2(field))


plt.imshow(np.fft.ifft2(field).real)
plt.colorbar()
plt.show()