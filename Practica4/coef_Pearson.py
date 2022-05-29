# Normal imports
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# PLS-related
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict


# (1, 2) Load data and give proper variable names
inFile = pkl.load(open('./spectra_from_cuvettes.pkl', 'rb'))
# Load all files
lambdas = inFile['lambdas']
spectra = inFile['spectra']
ink_c = inFile['ink_c']
intralipid_c = inFile['intralipid_c']

# This is not classy but it doesn't matter
X = np.asarray(spectra[0]).T
Y = np.repeat(np.array([ink_c[0], intralipid_c[0]])[:, np.newaxis],
              np.shape(spectra[0])[0], axis=1)

for k in range(1, len(spectra)):
    X = np.concatenate((X, np.asarray(spectra[k]).T), axis=1)
    Y = np.concatenate((Y, np.repeat(np.array([ink_c[k], intralipid_c[k]])[:, np.newaxis],
                                     np.shape(spectra[k])[0],
                                     axis=1)), axis=1)

# First, mean zero and standard deviation 1
X = (X-np.mean(X, axis=0))/np.std(X, axis=0)
Y = (Y-np.mean(Y, axis=0))/np.std(Y, axis=0)

# Another option: use KM approx
X = ((1.0-X)**2.0)/(2.0*X)

Num_a = 11
Pearson_coef = []

for a in range(1,Num_a):
    # This object is the PLS module
    pls = PLSRegression(n_components=a)
    # Fit data to regression
    pls.fit(X.T, Y.T)
    # Get all the matrices
    T = pls.x_scores_
    U = pls.y_scores_
    W = pls.x_weights_
    C = pls.y_weights_
    P = pls.x_loadings_
    Q = pls.y_loadings_
    # Then, get estimates and plot real vs. estimated
    Yhat = pls.predict(X.T).T
    #coeficiente de Person
    Pearson_coef.append(r2_score(Y[0, :],Yhat[0, :]))

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(np.linspace(1, Num_a-1,10), Pearson_coef, 'bo')
ax1.set_xlabel('variables latentes')
ax1.set_ylabel('Coeficiente de Pearson')
ax1.grid(True)


