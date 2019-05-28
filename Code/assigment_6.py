import numpy as np
import mathlib.LogisticRegression as lg
import matplotlib.pyplot as plt

# Load the data
# Column 2 = red_shift, 3 = T90
data = np.genfromtxt("GRBs.txt")

# Remove the first 2 columns. These contain 'GRB/XRF' and the number. 
data = data[:,2:]
data = np.array(data)
# The columns are now:
# 0 = red_shift, 1 = T90, 2 = log(M^*/M_{sun})
# 3 = SFR, 4 = log(Z/Z_{sun}), 5 = SSFR, 6 = AV

# Remove all rows without T90
data = data[data[:,1] != -1]

# Replace all missing data with zero's, but be carefull about
# the logarithms. To correct the logarithms we take the 
# column of a logirhtm to the power of  e and then put e^{-1} to zero.

data[:,2] = np.exp(data[:,2])
data[:,4] = np.exp(data[:,4])

# Correct for missing data
for i in range(0,7):
    if i == 2 or i == 4:
        mask = (data[:,i] == np.exp(-1))
        data[:,i][mask] = 0
    else:
        mask = (data[:,i] == -1)
        data[:,i][mask] = 0 


# The chosen columns are:
#  - redshift
#  - log(M^*/M_{sun})
#  - SFR
#
# These are used to create the final data for training the model.
# The final train data consits of the chosen columns plus 
# one additional column that is all 1 (used for bias)

# The columns in the dataset to select.
columns = [0,2,3]

train_data = np.ones((data.shape[0],len(columns)+1))
train_data[:,[0,1,2]] = data[:,columns] 
train_labels = data[:,1] > 10

# Train the model.
logistic_reg = lg.LogisticRegression(len(columns),1e-3,1e-5)
logistic_reg.train(train_data,train_labels)

# Predictions
predictions = logistic_reg.predict(train_data)
predictions = np.array(predictions,dtype=int)

#Create the histogram
bins = np.arange(0,1.1,0.05)

plt.hist(predictions,alpha=0.9,bins=bins, label='Model')
plt.hist(np.array(train_labels, dtype=int),bins=bins, label='True', density=True )
plt.legend()
plt.ylabel('Normalized counts')
plt.savefig('./Plots/6a_hist.pdf')

# Print the accuracy
print('[6] Accuracy: ', 1 - np.sum(predictions - train_labels)/len(train_labels))

