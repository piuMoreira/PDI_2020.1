import numpy as np
import loader

def rgb2yiq(img_in : np.array):
	mat_yiq = np.array(
        [[0.299, 0.587, 0.114], [0.59590059, -0.27455667, -0.32134392], [0.21153661, -0.52273617, 0.31119955]])
	img_out = np.copy(img_in).astype(np.float32)
	
	for i in range(len(img_in)):
		for j in range(len(img_in[0])):
			pixelrgb = img_in[i][j]
			pixelyiq = np.dot(mat_yiq, pixelrgb)
			pixelyiq /= 255
			img_out[i, j] = pixelyiq
			
	return img_out

def yiq2rgb(img_in : np.array):
	mat_rgb = np.array(
        [[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.106, 1.703]])
	img_out = np.copy(img_in)
	
	for i in range(len(img_in)):
		for j in range(len(img_in[0])):
			pixelyiq = img_in[i][j]
			pixelyiq *= 255
			pixelrgb = np.dot(mat_rgb, pixelyiq)
			img_out[i, j] = pixelrgb
	
	return img_out.astype(np.uint8)