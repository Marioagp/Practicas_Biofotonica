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

# ANALISIS CON UNA IMAGEN
data = plt.imread('./fake_and_real_beers_ms/fake_and_real_beers_RGB.bmp')
R = data
list1 = [570, 540, 440]
l = np.array(list1)

# Verify that data has visual integrity
fig21 = plt.figure(1)
ax211 = fig21.add_subplot(1, 1, 1)
ax211.imshow(np.mean(R, axis=2), cmap='gray')  # Reflectancia promedio
ax211.set_title('Reflectancia promedio 2.0')

# Find location of regions of interest (tentatively)
R_test = np.copy(np.mean(R, axis=2))
R_test[350:360, 130:140] = 1.0  # Real beer   SE MODIFICA LA ROI
R_test[350:360, 330:340] = 1.0  # Fake beer
ax211.imshow(R_test)

# Use these ROIs to extract spectra
R_real = np.mean(np.mean(R[350:360, 130:140, :], axis=0), axis=0)
print('R_real is ' + str(np.shape(R_real)))
R_fake = np.mean(np.mean(R[350:360, 330:340, :], axis=0), axis=0)
print('R_fake is ' + str(np.shape(R_fake)))

fig22 = plt.figure(2)
ax221 = fig22.add_subplot(1, 1, 1)
ax221.plot(l, R_real, label='Real beer')
ax221.plot(l, R_fake, label='Fake beer')
ax221.legend(fancybox=True)
plt.grid()


# Turn image into column vectors by
R_vec = np.reshape(R, (512 ** 2, 3)).T  # Transpose turns (512^2, 3) into (3, 512^2)

# Get matrix of column vectors
W = np.concatenate((R_real[np.newaxis, :], R_fake[np.newaxis, :], np.ones(3)[np.newaxis, :]), axis=0).T

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
fig24 = plt.figure(4, figsize=(12, 6))
ax241 = fig24.add_subplot(1, 2, 1)
ax242 = fig24.add_subplot(1, 2, 2)
ax241.imshow(np.clip(a_img[:, :, 0], 0, 1), cmap='jet')
ax241.set_title('Real beer detector!')
ax242.imshow(np.clip(a_img[:, :, 1], 0, 1), cmap='jet')
ax242.set_title('Fake beer detector!')

# Let's look at how these entities look:
a_real = np.mean(np.mean(a_img[250:260, 130:140, :], axis=0), axis=0)
a_fake = np.mean(np.mean(a_img[250:260, 330:340, :], axis=0), axis=0)
