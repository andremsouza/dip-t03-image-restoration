# coding: utf-8
#%%
# Name: André Moreira Souza
# NUSP: 9778985
# Course Code: SCC0251
# Semester: 2019/1
# Assignment: 2 - Image enhancement and filtering


#%%
import numpy as np
import scipy
import imageio
import scipy
import scipy.fftpack

#%% [markdown]
# ## Defining functions

#%%
# F == 1
def adaptive_denoising(g, gamma, k, mode):
	img_final = np.zeros(g.shape, dtype=np.double)
	
	# copying image borders from observed image
	img_final[:k//2, :] = g[:k//2, :]
	img_final[-(k//2):, :] = g[-(k//2):, :]
	img_final[:, :k//2] = g[:, :k//2]
	img_final[:, -(k//2):] = g[:, -(k//2):]
	
	# filtering for each mode
	# 'average' mode
	if mode == 'average':
		# calculating estimated dispersion measure
		disp_h = np.std(g[:g.shape[0]//6, :g.shape[1]//6])
		disp_h = 1 if disp_h == 0 else disp_h
		# applying denoising over each pixel
		for i in range(k//2, g.shape[0] - k//2):
			for j in range(k//2, g.shape[1] - k//2):
				# calculating centrality measure and dispersion measure over kxk neighborhood
				nh = g[i-k//2 : i+k//2 + 1, j-k//2 : j+k//2 + 1] # neighborhood
				centr_l = np.mean(nh)
				disp_l = np.std(nh)
				disp_l = disp_h if disp_l == 0 else disp_l
				img_final[i][j] = g[i][j] - gamma * (disp_h / disp_l) * (g[i][j] - centr_l)
	# 'robust' mode
	elif mode == 'robust':
		# calculating estimated dispersion measure
		percents_h = np.percentile(g[:g.shape[0]//6, :g.shape[1]//6], [25, 75])
		disp_h = percents_h[1] - percents_h[0]
		disp_h = 1 if disp_h == 0 else disp_h
		# applying denoising over each pixel
		for i in range(k//2, g.shape[0] - k//2):
			for j in range(k//2, g.shape[1] - k//2):
				# calculating centrality measure and dispersion measure over kxk neighborhood
				nh = g[i-k//2 : i+k//2 + 1, j-k//2 : j+k//2 + 1] # neighborhood
				percents_l = np.percentile(nh, [25, 50, 75])
				centr_l = percents_l[1]
				disp_l = percents_l[2] - percents_l[0]
				disp_l = disp_h if disp_l == 0 else disp_l
				img_final[i][j] = g[i][j] - gamma * (disp_h / disp_l) * (g[i][j] - centr_l)
	# error case: invalid input
	else: raise ValueError("Unexpected value for mode (should be in ['average', 'robust'])")
	# returning normalized image
	return normalize(img_final, g.max())
	

# F == 2
def constrained_least_squares(g, gamma, k, sigma):
	# img_final = np.zeros(g.shape, dtype=np.double)
	p = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.double) # laplacian operator
	h = gaussian_filter(k, sigma) # gaussian filter
	# padding arrays of invalid shape
	pad_p = [(g.shape[0] - p.shape[0]) // 2, (g.shape[1] - p.shape[1]) // 2]
	pad_h = [(g.shape[0] - h.shape[0]) // 2, (g.shape[1] - h.shape[1]) // 2]
	p = np.pad(p, ((pad_p[0], pad_p[0]), (pad_p[1], pad_p[1])), 'constant', constant_values=0)
	h = np.pad(h, ((pad_h[0], pad_h[0]), (pad_h[1], pad_h[1])), 'constant', constant_values=0)
	print("p: ", p.shape)
	print("h: ", p.shape)
	# calculating 2d-fft of arrays
	G = scipy.fftpack.fft2(g)
	P = scipy.fftpack.fft2(p)
	H = scipy.fftpack.fft2(h)
	H_conj = np.conj(H)

	# calculating final image
	
	img_final = (H_conj / (np.square(np.absolute(H)) + gamma * np.square(np.absolute(P)))) * G

	return img_final

# gaussian_filter function
def gaussian_filter(k=3, sigma=1.0):
	arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
	x, y = np.meshgrid(arx, arx)
	filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
	return filt / np.sum(filt)

# Normalize value of an numpy array between 0 and a given max value
def normalize (arr, maxvalue):
	return (arr-arr.min()) * (maxvalue / (arr.max()-arr.min()))

# root mean squared error (RMSE) function
def rmse (img_g, img_r):
	return np.sqrt(np.sum(np.power(img_g.astype(np.double) - img_r.astype(np.double), 2))/(img_g.shape[0]*img_g.shape[1]))

#%% [markdown]
# ## Main function

#%%
if __name__ == '__main__':
	# get user input
	fpath, gpath = str(input()).strip(), str(input()).strip() # f = reference image, g = degraded image
	f, g = imageio.imread(fpath).astype(np.double), imageio.imread(gpath).astype(np.double) # reading images f and g
	F = int(input()) # F = type of filter for restoration (1 -> denoising; 2 -> deblurring)
	if (F not in [1, 2]): raise ValueError("Unexpected value for F (should be '1' or '2')")
		
	gamma = np.double(input())
	if (gamma < 0 or gamma > 1): raise ValueError("Unexpected value for gamma (should be a float between '0' and '1')")
		
	k = int(input()) # k = size of denoising filter or degradation function (k in [3, 5, 7, 9, 11])
	if (k not in [3, 5, 7, 9, 11]): raise ValueError("Unexpected value for k (should be in [3, 5, 7, 9, 11])")
		
	# get specific inputs and restore image g
	if(F == 1):
		mode = str(input()).strip()
		if (mode not in ['average', 'robust']): raise ValueError("Unexpected value for mode (should be in ['average', 'robust'])")
		f_r = adaptive_denoising(g, gamma, k, mode)
	elif (F == 2):
		sigma = np.double(input())
		if (sigma <= 0): raise ValueError("Unexpected value for sigma (should be a float > 0)")
		f_r = constrained_least_squares(g, gamma, k, sigma)
	print('%.3f' % rmse(f, f_r))


#%%
import matplotlib.pyplot as plt
f_r = constrained_least_squares(g, gamma, k, sigma)
print('%.3f' % rmse(f, f_r))

plt.figure(figsize=(18,6))
plt.subplot(131)
plt.imshow(f, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(132)
plt.imshow(g, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(133)
plt.imshow(f_r, cmap="gray", vmin=0, vmax=255)
plt.axis('off')

# #%%
# f_r = adaptive_denoising(g, gamma, k, mode)
# print('%.3f' % rmse(f, f_r))


# #%%
# plt.figure(figsize=(18,6))
# plt.subplot(131)
# plt.imshow(f, cmap="gray", vmin=0, vmax=255)
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(g, cmap="gray", vmin=0, vmax=255)
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(f_r, cmap="gray", vmin=0, vmax=255)
# plt.axis('off')


# #%%
# def uniform_noise(size, prob=0.1):
#     '''
#     Generates a matrix with uniform noise in the range [0-255] to be added to an image
	
#     :param size: tuple defining the size of the noise matrix 
#     :param prob: probability for the uniform noise generation 
#     :type prob: float
#     :return matrix with uniform noise to be added to image
#     '''
	
#     levels = int((prob * 255) // 2)
#     noise = np.random.randint(-levels, levels, size)
	
#     return noise

# def gaussian_noise(size, mean=0, std=0.01):
#     '''
#     Generates a matrix with Gaussian noise in the range [0-255] to be added to an image
	
#     :param size: tuple defining the size of the noise matrix 
#     :param mean: mean of the Gaussian distribution
#     :param std: standard deviation of the Gaussian distribution, default 0.01
#     :return matrix with Gaussian noise to be added to image
#     '''
#     noise = np.multiply(np.random.normal(mean, std, size), 255)
	
#     return noise

# def impulsive_noise(image, prob=0.1, mode='salt_and_pepper'):
#     '''
#     Returns image with impulsive noise (0 and/or 255) to replace pixels in the image with some probability
	
#     :param image: input image
#     :param prob: probability for the impulsive noise generation 
#     :param mode: type of noise, 'salt', 'pepper' or 'salt_and_pepper' (default)
#     :type prob: float
#     :return noisy image with impulsive noise
#     '''

#     noise = np.array(image, copy=True)
#     for x in np.arange(image.shape[0]):
#         for y in np.arange(image.shape[1]):
#             rnd = np.random.random()
#             if rnd < prob:
#                 rnd = np.random.random()
#                 if rnd > 0.5:
#                     noise[x,y] = 255
#                 else:
#                     noise[x,y] = 0
	
#     return noise


# #%%
# uni_noise = uniform_noise(f.shape, prob=0.15)
# img_uni = np.clip(f.astype(int)+uni_noise, 0, 255)

# hist_img,_ = np.histogram(f, bins=256, range=(0,255))
# hist_uni,_ = np.histogram(img_uni, bins=256, range=(0,255))


# #%%
# f_r = adaptive_denoising(img_uni, gamma, k, mode)
# print(rmse(f, f_r))


# #%%
# plt.figure(figsize=(18,6))
# plt.subplot(131)
# plt.imshow(f, cmap="gray", vmin=0, vmax=255)
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(img_uni, cmap="gray", vmin=0, vmax=255)
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(f_r, cmap="gray", vmin=0, vmax=255)
# plt.axis('off')


# #%%
# gau_noise = gaussian_noise(f.shape, mean=0, std=0.05)
# img_gau = np.clip(f.astype(int)+gau_noise, 0, 255)

# hist_gau,_ = np.histogram(img_gau, bins=256, range=(0,255))


# #%%
# f_r = adaptive_denoising(img_gau, gamma, k, mode)
# print(rmse(f, f_r))


# #%%
# plt.figure(figsize=(18,6))
# plt.subplot(131)
# plt.imshow(f, cmap="gray", vmin=0, vmax=255)
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(img_gau, cmap="gray", vmin=0, vmax=255)
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(f_r, cmap="gray", vmin=0, vmax=255)
# plt.axis('off')


# #%%
# img_imp = impulsive_noise(f, prob=0.1)

# hist_imp,_ = np.histogram(img_imp, bins=256, range=(0,255))


# #%%
# f_r = adaptive_denoising(img_imp, gamma, k, mode)
# print(rmse(f, f_r))


# #%%
# plt.figure(figsize=(18,6))
# plt.subplot(131)
# plt.imshow(f, cmap="gray", vmin=0, vmax=255)
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(img_imp, cmap="gray", vmin=0, vmax=255)
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(f_r, cmap="gray", vmin=0, vmax=255)
# plt.axis('off')




# #%%


#%%
