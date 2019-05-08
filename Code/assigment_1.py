import mathlib.random as random
import matplotlib.pyplot as plt
import numpy as np

def main():

    # initialize the random number generator
    rng = random.Random(0xFADF00D90)

    # run assigment 1.a
    print('Executing assigment 1.a............')
    assigment_1a(rng) 
    print('Executing assigment 1.b............')
    assigment_1b(rng)
    
    pass


def assigment_1a(random):

    # print the seed
    print('Initial seed: ', random.get_seed())

    # generate 1000 numbers
    numbers_1000 = random.gen_uniforms(1000)

    # plot them agianst each other
    plt.scatter(numbers_1000[0:999], numbers_1000[1:], s=2)
    plt.show()

    # plot them agianst the index
    plt.plot(range(0,1000),numbers_1000)
    plt.show()

    # create a histogram for 1e6 points with 20 bins of 0.05 wide
    numbers_mil = random.gen_uniforms(int(1e6))

    plt.hist(numbers_mil, bins=20, range=(0,1),color='orange')
    plt.show()

def assigment_1b(random):

    # Sigma and mean for the distribution
    mean = 3.0
    sigma = 2.4

    # Generate 1000 random normal variables for the given mean and sigma.
    samples = random.gen_normals(mean,sigma,1000) 

    # The true normal distribution for the given mean and sigma,
    gaussian = lambda x: (1/(np.sqrt(2*np.pi)*2.4))*np.exp(-0.5*((x-3)/2.4)**2)
    gaussian_x = np.linspace(-sigma*4 +mean, sigma*4 +mean, 1000)
    gaussian_y = gaussian(gaussian_x)



    # Create histogram.
    plt.hist(samples, bins=25,density=True)
    plt.plot(gaussian_x, gaussian_y, c='orange')

    # Add the sigma lines.
    
    # The hight of the lines.
    lines_height = max(gaussian_y)*1.2

    for i in range(1, 5):
        # Absolute shift from the mean for the given sigma
        shift = i*sigma 
        # Line right of the mean
        plt.vlines(mean + shift,0, lines_height, linestyles='--')
        # Line left of the mean
        plt.vlines(mean - shift,0, lines_height, linestyles='--')   
        
    plt.show()



if __name__ == '__main__':
    main()