import numpy as np
import mathlib.integrate as integrate

def calculate_linear_growth(a_max, omega_m = 0.3, omega_lambda = 0.7):
    """
        Calculate the linear growth factor. 
    In:
        param: a_max -- The value of a to evaluate the linear growth function at.
        param: omega_m -- The mass fraction in the universe.
        param: omega_lambda -- The dark energy fraction in the universe.
    Out:
        return: The value of the linear growth factor for the given parameters.
    """
    

    # The prefactor of the integral
    pre_factor = 0.5*(5*omega_m)*(omega_m*a_max**(-3) + omega_lambda)**(0.5)

    # The function to integrate.
    func = lambda a: a**(-3) /(omega_m * (1/(a**3))+omega_lambda)**(3/2) 

    return pre_factor*integrate.romberg(func, 1e-7, a_max, 15)

def calculate_linear_growth_dir(a, omega_m = 0.3, omega_lambda = 0.7):
    """
        Calculate the derivative of the linear growth factor
    In:
        param: a -- The scale factor for which to calculate it.
        param: omega_m -- The fraction of matter in the universe.
        param: omega_lambda -- The fraction of dark energy in the universe
    Out:
        return: The derivative of the scale factor for the given parameters
    """
    # Hubble constant
    H0 = 7.16e-11

    # The function to integrate, which appears in the second term.
    func = lambda a: a**(-3) /(omega_m * (1/(a**3))+omega_lambda)**(3/2) 


    # The terms in the expression.
    pre_factor = (5*omega_m*H0)/(2*a**2)
    first_bracked_term = 1/(omega_m*a**(-3)+omega_lambda)**0.5
    second_bracked_term = (-3*omega_m*integrate.romberg(func, 1e-7, a, 15))/(2*a)

    return pre_factor*(first_bracked_term + second_bracked_term)


