import numpy as np
import mathlib.LogisticRegression as lg


# Load the data
# Column 2 = red_shift, 3 = T90
data = np.genfromtxt("GRBs.txt")

# Remove the first 2 columns. These contain 'GRB/XRF' and the number. 
data = data[:,2:]
data = np.array(data,dtype=np.float128)
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
# (1) - redshift
# (2) - log(M^*/M_{sun})
# (3) - SFR
#
# These are used to create the final data for training the model.
# The final train data consits of the chosen columns plus 
# one additional column that is all 1 (used for bias)

# The columns in the dataset to select.
columns = [0]
columns_set = [0]

train_data = np.ones((data.shape[0],len(columns)+1))
train_data[:,columns_set] = data[:,columns] 
train_labels = data[:,1] > 10

logistic_reg = lg.LogisticRegression(len(columns),1e-3,1e-5)
logistic_reg.train(train_data,train_labels)


print(1 - (((sum(logistic_reg.predict(train_data)) - sum(train_labels)))/len(train_labels)))

