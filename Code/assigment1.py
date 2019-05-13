import astropy.stats
import mathlib.random as random
import mathlib.sorting as sorting
import mathlib.statistics as ml_stats
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp_stats

def main():

    # Initialize the random number generator.
    rng = random.Random(0xFEDCBA)

    # Run assigment 1.a.
    assigment_1a(rng) 
    assigment_1b(rng)
    assigment_1c(rng)
    #print('Executing assigment 1.d............')
    #assigment_1d(rng)
    #print('Executing assigment 1.e.............')
    #assigment_1e(rng)


def assigment_1a(random):
    """
        Execute assigment 1.a
    Int:
        param: random -- An initialization of the random number generator.
    """

    # The relevant imports for this piece of code are:
    # (1) matplotlib.pyplot as plt
  
    # Print the seed.
    print('Initial seed: ', random.get_seed())

    # Generate 1000 numbers.
    numbers_1000 = random.gen_uniforms(1000)

    # Plot them agianst each other.
    plt.scatter(numbers_1000[0:999], numbers_1000[1:], s=2)
    plt.ylabel(r'Probability $x_{i+1}$')
    plt.xlabel(r'Probability $x_{i}$')
    plt.savefig('./Plots/1_plot_against.pdf')
    plt.figure()

    # Plot them against the index.
    plt.plot(range(0, 1000), numbers_1000)
    plt.ylabel('Probability')
    plt.xlabel('Index')
    plt.savefig('./Plots/1_plot_index.pdf')
    plt.figure()

    # Create a histogram for 1e6 points with 20 bins of 0.05 wide.
    numbers_mil = random.gen_uniforms(int(1e6))

    plt.hist(numbers_mil, bins=20, range=(0,1), color='orange',edgecolor='black')
    plt.ylabel('Counts')
    plt.xlabel('Generate values')
    plt.savefig('./Plots/1_hist_uniformnes.pdf')
    plt.figure()

def assigment_1b(random):
    """
        Execute assigment 1.b
    Int:
        param: random -- An initialization of the random number generator.
    """
    
    # The relevant imports for this piece of code are:
    # (1) matplotlib.pyplot as plt
    # (2) mathlib.statistics -- This contains the normal distribution is self
    

    # Sigma and mean for the distribution.
    mean = 3.0
    sigma = 2.4

    # Generate 1000 random normal variables for the given mean and sigma.
    samples = random.gen_normals(mean, sigma, 1000) 

    # The true normal distribution for the given mean and sigma.
    gaussian_x = np.linspace(-sigma*4 +mean, sigma*4 +mean, 1000)
    gaussian_y = ml_stats.normal(gaussian_x, mean, sigma)

    # Create a histogram.
    plt.hist(samples, bins=20, density=True, edgecolor='black', 
                      facecolor='orange', zorder=0.1, label='Sampled')
    plt.plot(gaussian_x, gaussian_y, c='red', label='Normal')
    plt.xlim(-sigma*6.5 + mean,sigma*6.5 + mean)
    plt.ylim(0, max(gaussian_y)*1.2)

    # Add the sigma lines.
    
    # The hight of the sigma lines that need to be added.
    lines_height = max(gaussian_y)*1.2

    for i in range(1, 6):
        # Absolute shift from the mean for the given sigma
        shift = i*sigma 
        
        # Sigma right of the mean.
        plt.vlines(mean + shift, 0, lines_height, 
                    linestyles='-', color='black', zorder=0.0)
        plt.text(mean + shift-0.4, lines_height/1.3, str(i) + r'$\sigma$',
                color='black', backgroundcolor='white', fontsize=9)

        # Sigma line left of the mean.
        plt.vlines(mean - shift, 0, lines_height, linestyles='-', zorder=0.0)  
        plt.text(mean - shift -0.4, lines_height/1.3, str(i) + r'$\sigma$',
                            color='black', backgroundcolor='white', fontsize=9)
 
    plt.legend(framealpha=1.0)
    plt.savefig('./Plots/1_hist_gaussian.pdf')
    plt.figure()

def assigment_1c(random):
    """
        Execute assigment 1.c
    Int:
        param: random -- An initialization of the random number generator.
    """
    
    # The relevant imports for this piece of code are:
    # (1) matplotlib.pyplot as plt
    # (2) numpy as np
    # (3) scipy.stats as sp_stats
    # (4) mathlib.sorting as sorting
    # (5) mathlib.statistics as ml_stats 

    # The values to plot point for.
    plot_values = np.array(10**np.arange(1, 5.1, 0.1),dtype=int)

    # An array in which the p-values are stored for the self created.
    # ks-test and the scipy version.
    p_values_self = np.zeros(len(plot_values))
    p_values_scipy = np.zeros(len(plot_values))

    # Generate the maximum amount of needed random numbers.
    random_numbers = random.gen_normals(0, 1, int(1e5))

    # Calculate the p-values with the ks-test.
    for idx, values in enumerate(plot_values):

        # Calculate the value with scipy. 
        p_values_scipy[idx] = sp_stats.kstest(random_numbers[0:values], 'norm')[1]
        p_values_self[idx] = ml_stats.kstest(random_numbers[0:values], ml_stats.normal_cdf)


    # Plot the probabilities for only my own implementation
    plt.plot(plot_values, p_values_self, label = 'self')
    plt.hlines(0.05,0,10**5,colors='red',linestyles='--')
    plt.xscale('log')
    plt.xlabel(r'Log($N_{samples}$)')
    plt.ylabel('Probabillity')
    plt.legend()
    plt.savefig('./Plots/1_plot_ks_test_self.pdf')
    plt.figure()

    # Plot the probabilities for beeing consistent under the null hypothesies.
    plt.plot(plot_values, p_values_scipy, label='scipy', linestyle=':', zorder=1.1)
    plt.plot(plot_values, p_values_self, label='self',zorder=1.0)
    plt.hlines(0.05,0,10**5,colors='red',linestyles='--')
    plt.xscale('log')
    plt.xlabel(r'Log($N_{samples}$)')
    plt.ylabel('Probabillity')
    plt.legend()
    plt.savefig('./Plots/1_plot_ks_test_self_scipy.pdf')
    plt.figure()

def assigment_1d(random):

    # The dex range to create the plots form
    dex_range = np.arange(1, 5.1, 0.1)


    # An array in which the p-values are stored for the self created
    # ks-test and the scipy version.
    p_values_self = np.zeros(len(dex_range))
    p_values_astropy = np.zeros(len(dex_range))

    # Calculate the p-values with the ks-test
    for idx, dex in enumerate(dex_range):
        random_numbers = random.gen_normals(0,1, int(10**dex)) #random.gen_normals(0,1, int(10**dex))

        p_values_self[idx] = stats.kuper_test(random_numbers, stats.normal_cdf) # discard distance
        p_values_astropy[idx] = astropy.stats.kuiper(random_numbers,stats.normal_cdf)[1] # stats.kuper_test(random_numbers,  stats.normal_cdf) #scipy_stats.norm.cdf)

    # Plot the probabiliteis for beeing consistent under the null hypothesies, thus p-values
    plt.plot(dex_range, p_values_self, label = 'self')
    plt.plot(dex_range, p_values_astropy, label='astrop')
    plt.xlabel(r'Log($N_{samples}$) [dex]')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

def assigment_1e(random):

    # Load the data.
    data = np.loadtxt('randomnumbers.txt')
    print(data.shape)
    
if __name__ == '__main__':
    main()