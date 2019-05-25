import numpy as np

class Random(object):
    """
        A class representing a random number generator (RNG)
    """

    def __init__(self, seed):
        """
            Create a new instance of the random number generator.
        
        In:
            param: seed -- The seed of the random number generator.
                           This must be a positive integer.
        
        """

        # The seed and state of the generator
        self._seed = np.uint64(seed)
        self._state = self._seed 

        # maximum uint32 value
        self._uint32_max = np.uint64(0xFFFFFFFF)

        # The values for the Xor shift.
        self._xor_a1 = np.uint64(20)
        self._xor_a2 = np.uint64(41)
        self._xor_a3 = np.uint64(5)

        # The values for the multiply with carry.
        self._mwc_a = np.uint64(4294957665)
        self._mwc_base = np.uint64(2**32)

    def get_seed(self):
        """ 
            Get the seed that is used to initialize this generator. 

        Out:
            return: The seed used to initalize the generatorr.
        """
        return self._seed

    def get_state(self):
        """
            Get the state of the generator.
        
        Out:
            return: The state of the generator.
        """
        return self._state


    def gen_next_int(self):
        """
            Generate a new random 32-bit unsigned integer.
        Out:
            return: A random 32-bit unsigned integer.
        """

        # The state is at the end updated with mwc.
        # We therefore shouldn't use more than 32 bits to generate
        # the number. 

        return self._update_state() & self._uint32_max

    def gen_uniform(self):
        """
            Generate a random float between 0 and 1.
        Out:
            return: A random float between 0 and 1.
        """

        return self.gen_next_int()*1.0 / self._uint32_max

    def gen_uniforms(self, amount):
        """ 
            Generate multiple random floats
            between 0 and 1.
        In:
            param: amount -- The amount of floats to generate.
        Out:
            return: An array with 'amount' random floats 
                    between 0 and 1.
        """

        samples = np.zeros(amount)

        for i in range(amount):
            samples[i] = self.gen_uniform()

        return samples

    def gen_normal(self, mean, sigma):
        """
            Generate a random normal distributed float.
        
        In:
            param: mean -- The mean of the gaussian distribution.
            param: sigma -- The squareroot of the variance of the distribution.
        Out:
            return: A random float that is drawn from the parameterized normal
                    distribution.
        """

        # Generate two uniform variables.
        u1 = self.gen_uniform()
        u2 = self.gen_uniform()

        # Use the box muller transformation.
        return sigma*np.sqrt(-2* np.log(1-u1))*np.cos(2*np.pi*u2) + mean

    def gen_normal_uniform(self, mean, sigma, u1, u2):
        """
            Generate a random normal distributed float from two provided
            uniform variables.
        
        
        """
        pre_factor= sigma*np.sqrt(-2* np.log(1-u1))

        return pre_factor*np.cos(2*np.pi*u2) + mean, pre_factor*np.sin(2*np.pi*u2) + mean

    def gen_normals(self, mean, sigma, amount):
        """
            Generate multible random normal distributed float.
        
        In:
            param: mean -- The mean of the gaussian distribution.
            param: sigma -- The squareroot of the variance of the distribution.
            param: amount -- The amount of floats to generate.
        Out:
            return: An array with random floats drawn from the parameterized normal
                    distribution.
        """

        # Pre-factors in the box muller transformation.
        square_pre_factor = -2*sigma**2 
        angle_pre_factor = 2*np.pi 

        # With the Box-muller two random normals can be generated for two
        # uniforms. If the amount of requested variables is odd then add
        # one to it and later remove it when returning the result.
        elements = amount if amount % 2 == 0 else amount + 1

        # Array in which the drawn normal distributed variables are stored.
        normal_dist = np.zeros(elements)

        # Apply the box muller transformation to generate the samples.
        for i in range(0, elements, 2):
            
            # Generate the uniforms.
            u1 = self.gen_uniform()
            u2 = self.gen_uniform()
            
            # Calculate common terms.
            pre_fact = np.sqrt(square_pre_factor*np.log(1-u1))

            # Calculate the samples.
            normal_dist[i] = pre_fact*np.cos(angle_pre_factor*u2) + mean
            normal_dist[i+1] = pre_fact*np.sin(angle_pre_factor*u2) + mean
       
        # If amount is odd, don't return the last element.
        return normal_dist[0:amount]
        
    def _update_state(self):
        """
            Update the state of the random number generator.
        
        Out:
            return: The new state of the random number generator.
        """

        self._state = self._xor_shift(self._state) 
        self._state = self._mwc(self._state & self._uint32_max) ^ self._state

        return self._state


    def _xor_shift(self, number):
        """ 
            Execute the XOR-shift algorithm on the 
            input number.
        In:
            param: number -- The number to XOR-shift.
        Out:
            return: The number produced by XOR-shift.
         """

        # Shift to the right and then bitwise xor.
        number ^= (number >> self._xor_a1) 
        # Shift to the left and then bitwise xor.
        number ^= (number << self._xor_a2) 
        # Shift to the right and then bitwise xor.
        number ^= (number >> self._xor_a3) 

        return number

    def _mwc(self, number):
        """
           Perform multiply with carry (MWC) on
           the given input.
        In:
            param: number -- The number to perform MWC on, must be an uint64. 
        Out:
            return: The new number.
        """
        return self._mwc_a * (number & (self._uint32_max -np.uint64(1))) + (number >> np.uint64(32))