import numpy as np

def central_diff(function, x, h):
    """
        Use the central difference method to approximate the derivative
        at a point x.
    In:
        param: function -- The function to calculate the derivative of.
        param: x -- The point to approximate the derivative of the function at.
        param: h -- A small value h that in the limit would go to zero.  
    Out:
        return: An approximation of the derivative of the provided
                function at x.    
    """

    #f'(x) approx (f(x+h)-f(x-h))/2h
    return (function(x+h) - function(x-h))/(2*h) 


def ridder(function, x , precision):
    """
        Perform ridders differential method to estimate 
        the derivative at a point x
    
    In:
        param: function -- The function to estimate the derivative for.
        param: x -- The point to estimate the derivative at.
        param: precision -- The precision that must be obtained 
                             in the estimation.
    Out:
        return: An approximation of the derivative.
    """ 

    # The number of initial approximations to use.
    approximations = 0                
    # The return value. 
    ret = 0                           
    # The current precision.
    current_precision = 0xFFFFFFFFFFF 

    # While we didn't reach the request precision with 
    # the current amount of initial approximations, try again 
    # but with more approximations.
    while current_precision > precision: 
        # Increase the amount of initial approximations.
        approximations += 5     
        # Reset the error. 
        current_precision = 0xFFFFFFFFFF 


        # The array to store the combined results in.
        results = np.zeros(approximations)

        # The current combination:
        # 0 = combine initial central_difference evaluations,
        # 1 = combine the combined central difference evaluations
        # 2 = combine the combined combined central difference evaluations etc.
        for combination in range(0, approximations-1): 
            
            # Combine for the current 'combination'.
             for j in range(1, approximations-combination): 

                # Create the initial central difference to combine.
                if combination == 0: 
                    # We need two central difference's to 
                    # combine the very first time.
                    if j == 1:        
                        results[j-1] = central_diff(function, x,1)

                    # Decrease h by a factor of 2 for each next approximation.
                    results[j] = central_diff(function, x, (1/2)**j)  

                # Keep the evaluation of the previous combination that 
                # is getting overwritten temporary in memory to update
                # the current precision.
                previous = results[j-1]

                # Combine
                power = 4**(combination+1)
                results[j-1] = (power*results[j] - results[j-1])/(power-1)

                # Determine the new precision
                precision_tmp = max(abs(results[j-1] - previous), \
                                     abs(results[j-1] - results[j]))
                
                # New precision is smaller-> update
                if precision_tmp < current_precision:  
                    current_precision = precision_tmp
                    ret = results[j-1]

                    # Terminate if requested precision is reached
                    if current_precision < precision: 
                        return ret

                # Abort early if the error of the last combined result is worse 
                # than the previous order by a large amount. 
                if j == (approximations-combination-1) \
                    and abs(ret - previous) > 100*current_precision:
                    
                    return ret                  
    return ret