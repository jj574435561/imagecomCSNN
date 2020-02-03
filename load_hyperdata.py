#import scipy.io
import numpy as np
#import math
import h5py

#mat = scipy.io.loadmat('Indian_pines_corrected.mat')
mat = h5py.File('aviris_sc0cal.mat', 'r')
mdata = mat['X']
mdata = np.array(mdata)
mdata = np.transpose(mdata, (2,1,0))
mdata = mdata.astype(np.double)
(row,col,band) = mdata.shape
Z = np.zeros_like(mdata)
local_sum = np.zeros_like(mdata)

'''
for k in range(0, band):
	for i in range(0, row):
		for j in range(0, col):
			if i == 0 and j == 0:
				local_mean = 0
			elif i == 0 and j > 0:	  # first row
				local_mean = mdata[i,j-1,k]
			elif i > 0 and j == 0:	  # first column except for 1st pixel
				local_mean = (mdata[i-1,j,k] + mdata[i-1,j+1,k])/2
			elif j == col-1 and i > 0:	# Last column except for last pixel in the first row
				local_mean = (mdata[i-1,j-1,k] + mdata[i-1,j,k] + mdata[i,j-1,k])/3
			else:
				local_mean = (mdata[i-1,j-1,k] + mdata[i-1,j,k] + mdata[i,j-1,k] + mdata[i-1,j+1,k])/4
			Z[i,j,k] = mdata[i,j,k] - math.floor(local_mean)
'''
for k in range(0, band):
	for i in range(0, row):
		for j in range(0, col):
			if i == 0 and j > 0:
				local_sum[i,j,k] = 4*mdata[i, j-1, k] 
			elif i > 0 and j == 0:	  
				local_sum[i,j,k] = 2*mdata[i,j-1,k] + mdata[i-1, j+1, k]
			elif i > 0 and j == col-1:	  
				local_sum[i,j,k] = 2*mdata[i-1,j,k] + mdata[i,j-1,k] + mdata[i-1,j-1,k]
			elif i == 0 and j == 0:
				local_sum[i,j,k] = 0
			else:
				local_sum[i,j,k] = mdata[i-1,j-1,k] + mdata[i-1,j,k] + mdata[i,j-1,k] + mdata[i-1,j+1,k]
			Z[i,j,k] = 4*mdata[i,j,k] - local_sum[i,j,k]


context_spatial = np.zeros((3, 4))
context_spectral = np.zeros((1, 4))
ground_truth = np.zeros((1, 1))
data_spatial = []
data_spectral = []
data_label = []

for k in range(0, band):
	if k == 0:
		for i in range(0, row):
			for j in range(0, col):
				if i == 0 and j <= 3: # Frist three pixels
					context_spatial[0, 0:4] = Z[i, j:j+4, k]
					context_spatial[1, 0:4] = Z[i+1, j:j+4, k]
					context_spatial[2, 0:4] = Z[i+2, j:j+4, k]
				elif i == 0 and j > 3:	# first row
					context_spatial[0, 0:4] = Z[i, j-4:j, k]
					context_spatial[1, 0:4] = Z[i+1, j-4:j, k]
					context_spatial[2, 0:4] = Z[i+2, j-4:j, k]
				elif i >= 1 and j == 0:	 # first col
					context_spatial[0, 0:4] = Z[i-1, j:j+4, k]
					context_spatial[1, 0:4] = Z[i, j:j+4, k]
					context_spatial[2, 0:4] = Z[i, j+4:j+8, k]
				elif i >= 1 and i < row-1 and j == col-1: # last col
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1:4] = Z[i-1, j-2:j+1, k]
					context_spatial[1, 0:4] = Z[i, j-5:j-1, k]
					context_spatial[2, 0:4] = Z[i-1, j-6:j-2, k]
				elif i == row-1 and	 1 <= j < col-1: # last row
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1:4] = Z[i-1, j-1:j+2, k]
					context_spatial[1, 0:4] = Z[i-5:i-1, j-1, k]
					context_spatial[2, 0:4] = Z[i-5:i-1, j, k]	
				elif i == row-1 and j == col-1: # last element
					context_spatial[0, 0:2] = Z[i, j-2:j, k]
					context_spatial[0, 2:4] = Z[i-1, j-1:j+1, k]
					context_spatial[1, 0:4] = Z[i-2, j-3:j+1, k]
					context_spatial[2, 0:4] = Z[i-3, j-3:j+1, k]	
				else:	
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1:4] = Z[i-1, j-1:j+2, k]
					context_spatial[1, 0] = Z[i, j-1, k]
					context_spatial[1, 1:4] = Z[i-1, j-1:j+2, k]
					context_spatial[2, 0] = Z[i, j-1, k]
					context_spatial[2, 1:4] = Z[i-1, j-1:j+2, k]
				ground_truth[0] = Z[i, j, k]
				data_spatial.append(np.copy(context_spatial))
				data_spectral.append(np.copy(context_spectral))
				data_label.append(np.copy(ground_truth))	 
				
	elif k == 1:
		for i in range(0, row):
			for j in range(0, col):
				if i == 0 and j <= 3: # Frist three pixels
					context_spatial[0, 0:4] = Z[i, j:j+4, k-1]
					context_spatial[1, 0:4] = Z[i+1, j:j+4, k-1]
					context_spatial[2, 0:4] = Z[i, j+4:j+8, k-1]
				elif i == 0 and j > 3:	# first row
					context_spatial[0, 0:4] = Z[i, j-4:j, k]
					context_spatial[1, 0:4] = Z[i, j-4:j, k-1]
					context_spatial[2, 0:4] = Z[i+1, j-4:j, k-1]
				elif i >= 1 and j == 0:	 # first col
					context_spatial[0, 0:4] = Z[i-1, j:j+4, k]
					context_spatial[1, 0:4] = Z[i-1, j:j+4, k-1]
					context_spatial[2, 0:4] = Z[i, j:j+4, k-1]
				elif i >= 1 and i < row-1 and j == col-1: # last col
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1] = Z[i-1, j-2, k]
					context_spatial[0, 2] = Z[i-1, j-1, k]	
					context_spatial[0, 3] = Z[i-1, j, k]	
					context_spatial[1, 0] = Z[i, j-1, k-1]
					context_spatial[1, 1] = Z[i, j, k-1]
					context_spatial[1, 2] = Z[i-1, j-1, k-1]	
					context_spatial[1, 3] = Z[i-1, j, k-1]
					context_spatial[2, 0] = Z[i, j-2, k-1]	
					context_spatial[2, 1:4] = Z[i+1, j-2:j+1, k-1]
				elif i == row-1 and	 1 <= j < col-1: # last row
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1:4] = Z[i-1, j-1:j+2, k]
					context_spatial[1, 0:2] = Z[i, j-1:j+1, k-1]
					context_spatial[1, 2:4] = Z[i-1, j-1:j+1, k-1]
					context_spatial[2, 0] = Z[i, j+1, k-1]	
					context_spatial[2, 1:4] = Z[i-2, j-1:j+2, k-1]
				elif i == row-1 and j == col-1: # last element
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1:4] = Z[i-1, j-2:j+1, k]
					context_spatial[1, 0:2] = Z[i, j-1:j+1, k-1]
					context_spatial[1, 2:4] = Z[i-1, j-1:j+1, k-1]
					context_spatial[2, 0] = Z[i-1, j-2, k-1]	
					context_spatial[2, 1:4] = Z[i-2, j-2:j+1, k-1]
				else:	
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1:4] = Z[i-1, j-1:j+2, k]
					context_spatial[1, 0:2] = Z[i, j-1:j+1, k-1]
					context_spatial[1, 2:4] = Z[i-1, j-1:j+1, k-1]
					context_spatial[2, 0] = Z[i, j+1, k-1]
					context_spatial[2, 1:4] = Z[i+1, j-1:j+2, k-1]
				context_spectral[0,0] = Z[i, j, k-1]
				ground_truth[0] = Z[i, j, k]
				data_spatial.append(np.copy(context_spatial))
				data_spectral.append(np.copy(context_spectral))
				data_label.append(np.copy(ground_truth))
				
	elif k == 2 or k == 3:
		for i in range(0, row):
			for j in range(0, col):
				if i == 0 and j <= 3:
					context_spatial[0, 0:4] = Z[i, j:j+4, k-1]
					context_spatial[1, 0:4] = Z[i+1, j:j+4, k-1]
					context_spatial[2, 0:4] = Z[i, j:j+4, k-2]
				elif i == 0 and j > 3:	# first row
					context_spatial[0, 0:4] = Z[i, j-4:j, k]
					context_spatial[1, 0:4] = Z[i, j-4:j, k-1]
					context_spatial[2, 0:4] = Z[i, j-4:j, k-2]
				elif i > 0 and j == 0:	# first col
					context_spatial[0, 0:4] = Z[i-1, j:j+4, k]
					context_spatial[1, 0:4] = Z[i-1, j:j+4, k-1]
					context_spatial[2, 0:4] = Z[i-1, j:j+4, k-2]
				elif i == 1 and j == col-1: # last col
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1] = Z[i-1, j-2, k]
					context_spatial[0, 2] = Z[i-1, j-1, k]	
					context_spatial[0, 3] = Z[i-1, j, k]	
					context_spatial[1, 0] = Z[i, j-1, k-1]
					context_spatial[1, 1] = Z[i, j, k-1]
					context_spatial[1, 2] = Z[i-1, j-1, k-1]	
					context_spatial[1, 3] = Z[i-1, j, k-1]
					context_spatial[2, 0] = Z[i, j-1, k-2]
					context_spatial[2, 1] = Z[i, j, k-2]
					context_spatial[2, 2] = Z[i-1, j-1, k-2]	
					context_spatial[2, 3] = Z[i-1, j, k-2]
				elif i > 1 and j == col-1: # last col
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1] = Z[i-1, j-1, k]
					context_spatial[0, 2] = Z[i-1, j, k]	
					context_spatial[0, 3] = Z[i-2, j, k]
					context_spatial[1, 0] = Z[i, j-1, k-1]
					context_spatial[1, 1] = Z[i, j, k-1]
					context_spatial[1, 2] = Z[i-1, j-1, k-1]
					context_spatial[1, 3] = Z[i-1, j, k-1]	
					context_spatial[2, 0] = Z[i, j-1, k-2]
					context_spatial[2, 1] = Z[i, j, k-2]
					context_spatial[2, 2] = Z[i-1, j-1, k-2]
					context_spatial[2, 3] = Z[i-1, j, k-2]						
				else:	
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1] = Z[i-1, j-1, k]
					context_spatial[0, 2] = Z[i-1, j, k]
					context_spatial[0, 3] = Z[i-1, j+1, k]
					context_spatial[1, 0] = Z[i, j-1, k-1]
					context_spatial[1, 1] = Z[i, j, k-1]
					context_spatial[1, 2] = Z[i-1, j-1, k-1]
					context_spatial[1, 3] = Z[i-1, j, k-1]
					context_spatial[2, 0] = Z[i, j-1, k-2]
					context_spatial[2, 1] = Z[i, j, k-2]
					context_spatial[2, 2] = Z[i-1, j-1, k-2]
					context_spatial[2, 3] = Z[i-1, j, k-2]
				if k == 2:
					context_spectral[0, 0:2] = Z[i, j, k-2:k]
				if k == 3:
					context_spectral[0, 0:3] = Z[i, j, k-3:k]
				ground_truth[0] = Z[i, j, k]
				data_spatial.append(np.copy(context_spatial))
				data_spectral.append(np.copy(context_spectral))
				data_label.append(np.copy(ground_truth))

	else:
		for i in range(0, row):
			for j in range(0, col):
				if i == 0 and j <= 3:
					context_spatial[0, 0:4] = Z[i, j:j+4, k-1]
					context_spatial[1, 0:4] = Z[i+1, j:j+4, k-1]
					context_spatial[2, 0:4] = Z[i, j:j+4, k-2]
				elif i == 0 and j > 3:	# first row
					context_spatial[0, 0:4] = Z[i, j-4:j, k]
					context_spatial[1, 0:4] = Z[i, j-4:j, k-1]
					context_spatial[2, 0:4] = Z[i, j-4:j, k-2]
				elif i > 0 and j == 0:	# first col
					context_spatial[0, 0:4] = Z[i-1, j:j+4, k]
					context_spatial[1, 0:4] = Z[i-1, j:j+4, k-1]
					context_spatial[2, 0:4] = Z[i-1, j:j+4, k-2]
				elif i == 1 and j == col-1: # last col
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1] = Z[i-1, j-2, k]
					context_spatial[0, 2] = Z[i-1, j-1, k]	
					context_spatial[0, 3] = Z[i-1, j, k]	
					context_spatial[1, 0] = Z[i, j-1, k-1]
					context_spatial[1, 1] = Z[i-1, j-2, k-1]
					context_spatial[1, 2] = Z[i-1, j-1, k-1]	
					context_spatial[1, 3] = Z[i-1, j, k-1]
					context_spatial[2, 0] = Z[i, j-1, k-2]
					context_spatial[2, 1] = Z[i-1, j-2, k-2]
					context_spatial[2, 2] = Z[i-1, j-1, k-2]	
					context_spatial[2, 3] = Z[i-1, j, k-2]
				elif i > 1 and j == col-1: # last col
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1] = Z[i-1, j-1, k]
					context_spatial[0, 2] = Z[i-1, j, k]	
					context_spatial[0, 3] = Z[i-2, j, k]
					context_spatial[1, 0] = Z[i, j-1, k-1]
					context_spatial[1, 1] = Z[i-1, j-1, k-1]
					context_spatial[1, 2] = Z[i-1, j, k-1]	
					context_spatial[1, 3] = Z[i-2, j, k-1]
					context_spatial[2, 0] = Z[i, j-1, k-2]
					context_spatial[2, 1] = Z[i-1, j-1, k-2]
					context_spatial[2, 2] = Z[i-1, j, k-2]	
					context_spatial[2, 3] = Z[i-2, j, k-2]	
				else:	
					context_spatial[0, 0] = Z[i, j-1, k]
					context_spatial[0, 1] = Z[i-1, j-1, k]
					context_spatial[0, 2] = Z[i-1, j, k]
					context_spatial[0, 3] = Z[i-1, j+1, k]
					context_spatial[1, 0] = Z[i, j-1, k-1]
					context_spatial[1, 1] = Z[i-1, j-1, k-1]
					context_spatial[1, 2] = Z[i-1, j, k-1]
					context_spatial[1, 3] = Z[i-1, j+1, k-1]
					context_spatial[2, 0] = Z[i, j-1, k-2]
					context_spatial[2, 1] = Z[i-1, j-1, k-2]
					context_spatial[2, 2] = Z[i-1, j, k-2]
					context_spatial[2, 3] = Z[i-1, j+1, k-2]
				ground_truth[0] = Z[i, j, k]
				context_spectral[0,0:4] = Z[i, j, k-4:k]
				data_spatial.append(np.copy(context_spatial))
				data_spectral.append(np.copy(context_spectral))
				data_label.append(np.copy(ground_truth))

data_label = np.array(data_label)
data_label = np.squeeze(data_label)				
data_spatial = np.array(data_spatial)
data_spectral = np.array(data_spectral)
		
np.savez('data_aviris_sc0cal_fl.npz', data_spatial = data_spatial, data_spectral = data_spectral, data_label = data_label)

