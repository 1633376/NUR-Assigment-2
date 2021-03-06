import numpy as np
import mathlib.sorting as sorting


def kstest(x, cdf):
    """
        Perform the Kolmogorov-Smirnov test for goodness of fit 
        and return the p-value.
    In:
        param: x   -- An array with value's who's CDF is expected to be
                      the same as the provided CDF. Must be atleast size 4

        param: cdf -- A function that is the expected cdf under the null hypothesis.
    Out:
        return: The p-value obtained by performing the KS-test. 
    """

    # Amount of values in the input array.
    x_size = len(x)

    # Sort the values and evaluate the cdf.
    x_sorted = sorting.merge_sort(x)
    x_sorted_cdf = cdf(x_sorted) 

    # Maximum distance.
    max_dist = 0

    # Value of the emperical cdf at step i-1.
    x_cdf_emperical_previous = 0

    # Find the maximum distance.
    for idx in range(0,x_size):

        # Calculate the emperical cdf.
        x_cdf_emperical = (idx+1)/x_size
        # The true cdf evaluation at the given point.
        x_cdf_true = x_sorted_cdf[idx]

        # Find the distance. The emperical
        # CDF is a step function so there are two distances
        # that need to be checked at each step.

        # Calculate the two distances
        distance_one = abs(x_cdf_emperical - x_cdf_true)
        distance_two = abs(x_cdf_emperical_previous - x_cdf_true)


        # Find the maximum of those two distances and
        # check if it is larger than the current know maximum distance.
        max_dist = max(max_dist, max(distance_one, distance_two))

        # Save the current value of the emperical cdf.
        x_cdf_emperical_previous = x_cdf_emperical

    # Calculate the p-value with the help of the CDF.
    sqrt_elemens = np.sqrt(x_size)
    cdf = _ks_statistic_cdf((sqrt_elemens + 0.12+0.11/sqrt_elemens)*max_dist)
    return 1 - cdf


def kstest2(x1, x2, x2_sorted = None):
    """
        Perform the Kolmogorov-Smirnov test for goodness of fit 
        and return the p-value. 
    In:
        param: x1   -- An array with value's who's CDF is expected to be
                       the same as the CDF of the proviced values.
                       Must be atleast size 4.

        param: x2 -- A discretized pdf of the expected distribution under the null hypothesis.
    Out:
        return: The p-value obtained by performing the KS-test 
    """

    # Amount of values in the input distributions.
    x1_size = len(x1)
    x2_size = len(x2)

    # Sort both arrays.
    x1 = sorting.merge_sort(x1)
    x2 = sorting.merge_sort(x2) if type(x2_sorted) is not None else x2_sorted

    # The maximum distance
    max_dist = 0

    # The iteration values used to determine
    # the emperical pdf's and the max distance.
    x1_i, x2_j = 0,0
  
    # Find the maximum distance by updating the emperical CDFs.
    while x1_i < x1_size and x2_j < x2_size:

        # Update the indices used for the emperical CDF's. 
        
        if x1[x1_i] < x2[x2_j]:           
            x1_i += 1  
        else:
            x2_j += 1

        # Find the max distance
        max_dist = max(abs(x1_i/x1_size-x2_j/x2_size), max_dist)

    sqrt_factor = np.sqrt((x1_size*x2_size)/(x1_size+x2_size))
    cdf = _ks_statistic_cdf((sqrt_factor + 0.12+0.11/sqrt_factor)*max_dist)

    return 1 - cdf

def kuiper_test(x, cdf):
    """
        Perform the Kuiper test for goodness of fit 
        and return the p-value.
    In:
        param: x   -- An array with value's who's CDF is expected to be
                      the same as the provided CDF. Must be atleast size 4

        param: cdf -- A function that is the expected cdf under the null hypothesis.
    Out:
        return: The p-value obtained by performing the kuiper-test 
    """

    # Sort the data in ascending order, calculate the 
    # cdf and the emperical cdf for the sorted values and 
    # save the total amount of  elements we have. 
    x_sorted = sorting.merge_sort(x)
    x_sorted_cdf = cdf(x_sorted) 
    x_elements = len(x)

    # Find the maximum distance above and below
    # the true cdf. 
    max_dist_above = 0   
    max_dist_below = 0

    # Value of the cdf at step i-1.
    x_cdf_emperical_previous = 0 
    

    for idx, x in enumerate(x_sorted):

        # Calculate the emperical cdf.
        x_cdf_emperical = (idx+1)/x_elements
        # Calculate the true cdf.
        x_cdf_true = x_sorted_cdf[idx]

        # Find the maximum distance above and below
        max_dist_above = max(x_cdf_emperical - x_cdf_true, max_dist_above)
        max_dist_below = max(x_cdf_true - x_cdf_emperical_previous, max_dist_below)

        # Update previous cdf
        x_cdf_emperical_previous = x_cdf_emperical

    sqrt_elem = np.sqrt(x_elements)
    v = max_dist_above + max_dist_below
    cdf = _kuiper_statistic_cdf((sqrt_elem + 0.155+0.24/sqrt_elem)*v)

    return 1 - cdf
    

def _ks_statistic_cdf(z):
    """
        An approximation for the cdf of the 
        Kolmogorov-Smirnov (KS) test staistic.
    In:
        param: z -- The value to calculate the cdf at.
    Out:
        return: An approximation of the cdf for the given value.    print(max_dist_above + max_dist_below)
    """

    # Numerical approximation taken from:
    # Numerical methods - The art of scientific computation.
    # Third edition.

    if z < 1.18:
        exponent = np.exp(-np.pi**2/(8*z**2))
        pre_factor = np.sqrt(2*np.pi)/z

        return pre_factor*exponent*(1+ exponent**8)*(1+exponent**16)
    else:
        exponent = np.exp(-2*z**2)
        return 1-2*exponent*(1-exponent**3)*(1-exponent**6)

def _kuiper_statistic_cdf(z):
    """
        An approximation for the cdf of the
        Kuiper test statistic
    In:
        param: z -- The value to calculate the cdf at.
    Out:
        return: An approximation of the cdf for the given value.
    """

    # Value of z is to small, sum will be 1 up to 7 digits
    if z < 0.4:
        return 1 
        
    # Approximateed value of the sum by performing 100 iterations
    
    # The value to return
    ret = 0
    # A term often needed in the sum.
    z_squared = z**2

    # Evaluate the first 100 terms in the sum.
    for j in range(1, 100):
        power = j**2 * z_squared
        ret += (4 * power -1)*np.exp(-2*power)

    return 1- 2*ret



def normal_cdf(x, mean = 0, sigma = 1):
    """
        Evaluate the cummulative normal distribution for
        the given parameters
    In:
        param: x -- The point to evaluate the cdf at or an array of points to evaluate it for.
        param: mean -- The mean of the normal distribution.
        param: sigma -- The square root of the variance for the normal distribution.
    Out:
        return: The cummulative normal distribution evaluated at.
    """

    # Calculate the CDF using the erf function (defined below).
    return 0.5 + 0.5*erf((x-mean)/(np.sqrt(2)*sigma)) 


def normal(x, mean = 0, sigma = 1):
    """
        Evaluate the normal distrbution for the given
        parameters.
    In:
        param: x -- The point to evaulte the distribution at.
        param: mean -- The mean of the distribution.
        param: sigma -- The square root of the variance for the distribution.
    Out:
        return: The value of the parameterized distribution evaluated at the given point.
    """
    
    return 1/((np.sqrt(2*np.pi)*sigma))*np.exp(-0.5*((x - mean)/sigma)**2)

def erf(x):
    """
        Evaluate the erf function for a value of x.
    In:
        param: x -- The value to evaluate the erf function for.
    Out:
        return: The erf function evaluated for the given value of x.
    """

    # Numerical approximation taken from Abramowits and Stegun.

    # Constants for the numerical approximation
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    # Array in which the result is stored
    ret = np.zeros(len(x))
    
    # The approximation functions
    erf_func_t_val = lambda x: 1/(1+ p*x)
    erf_func_approx = lambda t, x : 1 -  t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))*np.exp(-x**2) 

    # Evaluate for both positive and negative
    neg_mask = x < 0
    neg_x = x[neg_mask]
    pos_mask = x >= 0
    pos_x = x[pos_mask]

    ret[neg_mask] = -erf_func_approx(erf_func_t_val(-neg_x), -neg_x)
    ret[pos_mask] = erf_func_approx(erf_func_t_val(pos_x), pos_x)

    return ret


