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
        ret[0] = data[0]
        return ret

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


def fft2(data):
    """
        Calculate the 2D fourier transformation.
    In:
        param: data -- The data to fourier transform
    Out:

    """

    for i in range(data.shape[0]):
        #kx_0,ky_0, kx_0,ky_1, kx_0,ky_2, kx_0,ky_3
        data[i] = fft(data[i])      
    
    # now perform it over the rows (x direction)
    for i in range(data.shape[0]):
        #kx_0,ky_0, kx_1,ky_0, kx_2,ky_0, (store in column)
        data[:,i] = fft(data[:,i])
        
    return data

def fft3(data):
    """
        Perform the 3 dimensionale fourier transform
    In:
        param: -- The data to fourier transform
    Out:
        return: The 3D fourier transform of the data.
    """
    
    # First fourier transform the planes
    for i in range(data.shape[0]):
        data[i,:,:] = fft2(data[i,:,:])
    
    # Finally fourier transform the last axis.
    for i in range(data.shape[2]):
        for j in range(data.shape[1]):
            data[:,i,j] = fft(data[:,i,j])
            
    return data