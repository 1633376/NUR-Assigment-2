import numpy as np
import mathlib.integrate as integrate
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
        return: The p-value obtained by performing the KS-test 
    """

    # Sort the data in ascending order, calculate the 
    # cdf and the emperical cdf for the sorted values and save the total amount of 
    # elements we have. 
    x_sorted = sorting.merge_sort(x)
    x_sorted_cdf = cdf(x_sorted) 
    x_elements = len(x)

    # Find the maximum distance. 
    max_dist = 0    

    # value of the cdf at step i -1
    x_cdf_emperical_previous = 0 
    

    for idx, x in enumerate(x_sorted):

        # Calculate the emperical cdf.
        x_cdf_emperical = (idx+1)/x_elements
        # Calculate the true cdf.
        x_cdf_true = x_sorted_cdf[idx]

        # Find the max distance.
        # TODO: Why also compare with emperical of previous?
        max_dist = max(max(abs(x_cdf_emperical - x_cdf_true),abs(x_cdf_emperical_previous - x_cdf_true)), max_dist)

        x_cdf_emperical_previous = x_cdf_emperical

    sqaure_elem = np.sqrt(x_elements)
    return 1- _ks_statistic_cdf((sqaure_elem + 0.12+0.11/sqaure_elem)*max_dist)

    


def _ks_statistic_cdf(z):
    """
        An approximation for the cdf of the 
        Kolmogorov-Smirnov (KS) test staistic.

    In:
        param: z -- The value to calculate the cdf at.
    Out:
        return: An approximation of the cdf for the given value.
    """

    if z < 1.18:
        exponent = np.exp(-np.pi**2/(8*z**2))
        pre_factor = np.sqrt(2*np.pi)/z

        return pre_factor*exponent*(1+ exponent**8)*(1+exponent**16)
    else:
        exponent = np.exp(-2*z**2)
        return 1-2*exponent*(1-exponent**3)*(1-exponent**6)

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

    # calculate it using the erf function (defined below)

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
        return: The parameterized istribution evaluated at the given point.
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

    # erf function is represented as a taylor series around 0
    # we take up to O(10) terms
    order =30
    erf = np.zeros(x.shape,dtype=object)
    x_pow = x
    
    x_squared = x**2
    sign = 1
    
    for i in range(order):
        erf += ( sign * x_pow)/((1+2*i)*np.math.factorial(i))
        sign *= -1
        x_pow = x_pow * x_squared

    return (2/np.sqrt(np.pi))*erf 
    