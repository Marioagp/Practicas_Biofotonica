'''
Calibración de datos para la práctica 4
Arturo Pardo, 2020
'''

import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from PIL import Image
from color_bar import color_bar

''' Create .pkl file for peppers '''
image_0 = np.asarray(Image.open('./fake_and_real_beers_ms/fake_and_real_beers_ms_' + '01.png')) / 2.0 ** 16.0
width, height = np.size(image_0, 0), np.size(image_0, 1)
l_beer = np.linspace(400, 700, 31)
R_beer = np.zeros([width, height, 31])
for k in range(0, 31):
    img = np.asarray(
        Image.open('./fake_and_real_beers_ms/fake_and_real_beers_ms_' + str(k + 1).zfill(2) + '.png')) / 2.0 ** 16.0
    R_beer[:, :, k] = np.copy(img)

out_dict_peppers = {
    'R': R_beer,
    'l': l_beer
}
pkl.dump(out_dict_peppers, open('./beer.pkl', 'wb'))

'''
Sometimes, it's not possible to perfectly quantify what is on an image. 
This example requires the CAVE dataset peppers sample image (should be within the file). 
'''

# Load the reflectance file (512 x 512 x 31)
data = pkl.load(open('./beer.pkl', 'rb'))
R = data['R']
l = data['l']

# Verify that data has visual integrity
fig1 = plt.figure(1)
ax11 = fig1.add_subplot(1, 1, 1)
ax11.imshow(np.mean(R, axis=2), cmap='gray')  # Reflectancia promedio
ax11.set_title('Reflectancia promedio')

# Find location of regions of interest
R_test = np.copy(np.mean(R, axis=2))
R_test[350:360, 130:140] = 1.0       # Real beer
R_test[350:360, 330:340] = 1.0       # Fake beer
ax11.imshow(R_test)

# Use these ROIs to extract spectra
R_real = np.mean(np.mean(R[350:360, 130:140, :], axis=0), axis=0)
print('R_real is ' + str(np.shape(R_real)))
R_fake = np.mean(np.mean(R[350:360, 330:340, :], axis=0), axis=0)
print('R_fake is ' + str(np.shape(R_fake)))



fig2 = plt.figure(2)
ax21 = fig2.add_subplot(1, 1, 1)
ax21.plot(l, R_real, label='Real beer')
ax21.plot(l, R_fake, label='Fake beer')
ax21.legend(fancybox=True)
plt.grid()

# Those are our vectors of interest.

# Turn image into column vectors by
R_vec = np.reshape(R, (512 ** 2, 31)).T  # Transpose turns (512^2, 31) into (31, 512^2)

# Get matrix of column vectors
W = np.concatenate((R_real[np.newaxis, :], R_fake[np.newaxis, :], np.ones(31)[np.newaxis, :]), axis=0).T

# Do pseudoinverse to extract linear coefficients!
a = np.matmul(np.linalg.pinv(W), R_vec)
a_img = np.reshape(a.T, (512, 512, W.shape[1]))

# Look at the data!

fig3 = plt.figure(3, figsize=(12, 6))
ax31 = fig3.add_subplot(1, 3, 1)
ax32 = fig3.add_subplot(1, 3, 2)
ax33 = fig3.add_subplot(1, 3, 3)

im31 = ax31.imshow(a_img[:, :, 0], cmap='jet')
ax31.set_title('Real fraction')
color_bar(im31, ax31)

ax32.imshow(a_img[:, :, 1], cmap='jet')
ax32.set_title('Fake fraction')
color_bar(im31, ax32)

ax33.imshow(a_img[:, :, 2], cmap='jet')
ax33.set_title('Bias fraction')
color_bar(im31, ax33)

# Finally, do a nice transformation :)
fig4 = plt.figure(4, figsize=(12, 6))
ax41 = fig4.add_subplot(1, 2, 1)
ax42 = fig4.add_subplot(1, 2, 2)
ax41.imshow(np.clip(a_img[:, :, 0], 0, 1), cmap='jet')
ax41.set_title('Real beer detector!')
ax42.imshow(np.clip(a_img[:, :, 1], 0, 1), cmap='jet')
ax42.set_title('Fake beer detector!')

# Let's look at how these entities look:
a_real = np.mean(np.mean(a_img[250:260, 230:240, :], axis=0), axis=0)
a_fake = np.mean(np.mean(a_img[250:260, 230:240, :], axis=0), axis=0)
