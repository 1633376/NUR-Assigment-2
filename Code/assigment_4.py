import mathlib.integrate as integrate
import mathlib.misc as misc
import mathlib.random as rnd
def main():
    assigment4_a()
    assigment4_c()
    


def assigment4_a():

    # The value of the scale to integrate to
    a_max = 1/51

    # The values of the density parameters
    omega_m = 0.3
    omega_lambda = 0.7

    # The prefactor of the integral
    pre_fac = 0.5*(5*omega_m)*(omega_m*a_max**(-3) + omega_lambda)

    # The function to integrate.
    # It is not written as lambda so that the scale to the power of 3
    # doesn't have to be calculated twice every evaluation.

    func = lambda a: a**(-3) /(omega_m * (1/(a**3))+omega_lambda)**(3/2) 

    # The result
    print('D(a = 1/51) = ', integrate.romberg(func, 1e-7, a_max, 10)*pre_fac)

def assigment4_b():
    pass

def assigment4_c():
    pass
    



if __name__ == "__main__":
    main()