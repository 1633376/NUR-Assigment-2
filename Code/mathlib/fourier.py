import numpy as np

def fft(data):
    """
        Perform the DFT with the cooley-tukey algorithm.
    In:
        param: data -- The array to fourier transform, must be of length 2**N
    Out:
        return: The fft of the data with the zeroth frequence as the first elment.
    """
    
    # Get the length of the data
    n = len(data)
    
    # The array to return. By using this 
    # array the input array wont't require
    # a reverse bitshift. This was mainly done
    # to prevent modifying the input array.
    ret = np.zeros(n, dtype=complex)
    
    # The FFT of 1 element is the element self.
    if n == 1:
        return data

    # Perform the FFT on the even indices and
    # odd indices.
    even = fft(data[::2])
    odd = fft(data[1::2])
    
    # Calculate the recurrent split for the 
    # values of k. Notice here that
    # there is a minus sign, which the result
    # of symmetry that is exploited.
    for k in range(int(n/2)):
        
        # Multiply with the complex exponent 
        odd[k] *= np.exp(-1J*2*np.pi*k/n)

        # Calculate the FFT for the discritized k (or f) values
        ret[k] = even[k] + odd[k]
        # symmetry exploid
        ret[k+int(n/2)] = even[k] - odd[k]

    # return the fourier transform, don't use a normalization
    return ret




def set_bits(number, b1, one):
    """
        Set a specific bit of a n umber (starting) at one
    """

    # Set the bit by b2 at the value of b2.
    if one == 1: # put it at 1
        number |= ( 1 << b1)
    else: # put it at 0
        number &= ~(1 << b1)

    return number

def get_bit(number,b1):
    """
        Get the value of as pecific bit starting at 0
    """

    return (number & (1 << b1-1)) != 0


def reverse2(number,bits):
   return int('{:0{bits}b}').format(number, bits)[::-1], 2)


import matplotlib.pyplot as plt
import scipy.fftpack

x = np.arange(0,64) #linspace(0,4,16)
y = np.sin(x)
z = np.zeros(64,dtype=int)
bits = int(np.log2(max(x)))+1

for i in x:
    z[i] = reverse2(i,bits)


plt.plot(x,abs(scipy.fftpack.rfft(y)),label='np')
plt.plot(x,abs(FFT(np.array(y,dtype=complex))),label='self')
plt.legend()
plt.show()