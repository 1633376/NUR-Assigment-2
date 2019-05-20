import numpy as np

def trapezoid(function, a, b, num_trap):
    """
        Perform trapezoid integration

    In:
        param: function -- The function to integrate.
        param: a -- The value to start the integration at.
        param: b -- The value to integrate the function to.
        param: num_trap -- The number of trapezoids to use.

    Out:
        return: An approximation of the integral from a to b of 
                the function 'function' 
    """

    # The step size for the integration range.
    dx = (b-a)/num_trap 
    
    # Find trapezoid points.
    x_i = np.arange(a,b+dx,dx) 

    # Determine the area of all trapezoids.
    area = dx*(function(a) + function(b))/2
    area += np.sum(function(x_i[1:num_trap])) *dx

    return area



def romberg(function, a, b, num_trap):
    """
        Perform romberg integration
    
    In:
        param: function -- The function to integrate.
        param: a -- The value to start the integration at.
        param: b -- The value to stop the integrate at.
        param: num_trap -- The maximum number of initial trapezoid 
                           to use to approximate the area.
    
    Out:
        return: An approximation of the integral from a to b of 
                the function 'function' 
    """

    # The array to store the combined results in.
    results = np.zeros(num_trap)
    
    # The current combination:
    # 0 = combine trapezoids, 1 = combine combined trapezoids,
    # 2 = combine the combined combined trapezoids etc.
    for combination in range(0, num_trap-1): 

        # Iterate and combine.
         for j in range(1, num_trap-combination): 
            # Create the initial trapezoids to combine.
            if combination == 0: 
                 # We need two trapezoids to combine the very first time.
                if j == 1:
                    results[j-1] = trapezoid(function, a, b, 1)

                results[j] = trapezoid(function,a,b,2**j)

            # Combine.
            power = 4**(combination+1)
            results[j-1] = (power*results[j] - results[j-1])/(power-1)

    return results[0] 
        



