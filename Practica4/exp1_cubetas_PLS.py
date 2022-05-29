'''
Cuvette experiment with Projection to Latent Structures
Arturo Pardo for Biophotonics, M2022, Universidad de Cantabria
'''

# Normal imports
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# PLS-related
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

'''
This time, it's fairly simple:
    (1) Load the data provided from spectra_from_cuvettes.pkl
    (2) Store into X and Y 
    (3) Use prefab sklearn PLS module 
    (4) Play with the vectors 
'''

# (1, 2) Load data and give proper variable names
inFile = pkl.load(open('./spectra_from_cuvettes.pkl', 'rb'))
for key in inFile.keys():
    print('key: ' + key + ', shape: ' + str(np.shape(inFile[key])))

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

print('X is ' + str(np.shape(X)))
print('Y is ' + str(np.shape(Y)))

# Show on the plot
fig1 = plt.figure(1)
fig1.clf()
ax11 = fig1.add_subplot(1, 1, 1)
ax11.clear()
for k in range(0, len(spectra)):
    ax11.plot(lambdas, np.asarray(spectra[k]).T, c=(ink_c[k], 0.5, 0.5), alpha=0.4)
    ax11.plot(400, 0, c=(ink_c[k], 0.5, 0.5), label=str(np.round(ink_c[k],2)) + ' ink, ' + str(np.round(intralipid_c[k],2)) + ' Intralipid')


ax11.legend(fancybox=True)
ax11.grid(True)
ax11.set_xlabel('Wavelength [nm]')
ax11.set_ylabel('Reflectance')
fig1.tight_layout()
fig1.canvas.draw()
fig1.canvas.flush_events()

''' Projection on Latent Spaces ------------------------------------------------------------------------------------ '''
'''
# First, mean zero and standard deviation 1
X = (X-np.mean(X, axis=0))/np.std(X, axis=0)
Y = (Y-np.mean(Y, axis=0))/np.std(Y, axis=0)

# Another option: use KM approx
X = ((1.0-X)**2.0)/(2.0*X)
'''

# This object is the PLS module
a = [3, 5, 7, 10]
for i in range(4):
    pls = PLSRegression(n_components=a[i])
    # Fit data to regression
    pls.fit(X.T, Y.T)
    # Get all the matrices
    T = pls.x_scores_
    U = pls.y_scores_
    W = pls.x_weights_
    C = pls.y_weights_
    P = pls.x_loadings_
    Q = pls.y_loadings_
    # Plot the weights of X
    fig2 = plt.figure(i+2)
    ax21 = fig2.add_subplot(1, 1, 1)
    ax21.clear()
    for k in range(0, a[i]):
        ax21.plot(lambdas, W[:, k], label='Weight vector ' + str(k))
    ax21.legend(fancybox=True)
    ax21.grid(True)
    fig2.canvas.draw()
    fig2.canvas.flush_events()
    # Then, get estimates and plot real vs. estimated
    Yhat = pls.predict(X.T).T
    fig3 = plt.figure(i+6)
    ax31 = fig3.add_subplot(1, 1, 1)
    ax31.clear()
    ax31.scatter(Y[0, :], Yhat[0, :], label='Ink concentration', marker='.', color='r', edgecolor=None, alpha=0.5)
    ax31.scatter(Y[1, :], Yhat[1, :], label='Intralipid concentration', marker='.', color='g', edgecolor=None, alpha=0.5)
    ax31.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--k')
    ax31.set_xlabel('Real')
    ax31.set_ylabel('Predicted')
    ax31.legend(fancybox=True)
    ax31.grid(True)
    ax31.axis('square')
    fig3.tight_layout()
    fig3.canvas.draw()
    fig3.canvas.flush_events()