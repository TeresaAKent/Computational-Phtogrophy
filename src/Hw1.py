# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 20:44:44 2021

@author: tkent
"""

import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from skimage.color import rgb2gray

    
im1= skio.imread("campus.tiff")
H,W=np.shape(im1)
H2=int(H/2)
W2=int(W/2)
print (H,W)
im2=im1.astype(np.float64)
im2=np.clip((im2-150)/(4095),0,1)

imB1=np.zeros((H2,W2,3))
# Location of r,g,g,b 1,2,3,4 locations see write up
BP=np.array([0,1,2,3]).astype(int)

#Assign the red channel
imB1[:,:,0]=im2[BP[0]//2::2,BP[0]%2::2]
# Assign the Blue Channel
imB1[:,:,2]=im2[BP[3]//2::2,BP[3]%2::2]
# Assign the Green Channel
imB1[:,:,1]=1/2*(im2[BP[1]//2::2,BP[1]%2::2]+im2[BP[2]//2::2,BP[2]%2::2])
plt.imshow(imB1)

# White Balencing
Max=np.amax(imB1,(0,1))
WhiteBalence=imB1/Max*Max[0]
plt.figure()
ax=plt.subplot(131)
ax.set_title("White Balencing")
plt.imshow(WhiteBalence)

# Gray Balencing
Average=np.average(imB1,(0,1))
GrayBalence=imB1/Average*Average[1]
ax=plt.subplot(132)
ax.set_title("Grey Balencing")
plt.imshow(GrayBalence)

# Other Balencing
ax=plt.subplot(133)
ax.set_title("Other Balencing")
multipler=np.array([2.393118,1.000000,1.223981])
OtherBalence=imB1*multipler
plt.imshow(OtherBalence)

# Demosaicing
im3=np.zeros_like(GrayBalence)
y = np.arange(H2).reshape(-1,1).flatten()
x= np.arange(W2).reshape(-1,1).flatten()
X,Y=np.meshgrid(x,y)
r=interp2d(x,y,GrayBalence[:,:,0])
g=interp2d(x,y,GrayBalence[:,:,1])
b=interp2d(x,y,GrayBalence[:,:,2])
plt.figure()
zx=plt.subplot(121)
plt.imshow(GrayBalence)
zx.set_title("Before Denoise")
im3[:,:,0]=r(x,y)
im3[:,:,1]=g(x,y)
im3[:,:,2]=b(x,y)
zx=plt.subplot(122)
plt.imshow(im3)
zx.set_title("After Denoise")

#Color Space Correction
plt.figure()
zx=plt.subplot(121)
plt.imshow(im3)
zx.set_title("Before Color Correction")
MSRGBXYZ=np.array([[.4124564, .3575761, .1804375],[.2126719,.7151522,.0721750],[.0193339,.1191920,.9503041]])
MXYZcam=np.array([6988,-1384,-714,-5631,13410,2447,-1485,2204,7318])/10000
MXYZcam=MXYZcam.reshape((3,3),order='F')
MSRGBcam=MXYZcam*MSRGBXYZ
MSRGBcam=MSRGBcam/np.sum(MSRGBcam,0)
im4=np.zeros_like(GrayBalence)
for i in range (H2):
    for j in range (W2):
        imagePixel=im3[i,j,:].reshape(3,1)
        im4[i,j,:]=np.matmul(np.linalg.inv(MSRGBcam),imagePixel).T
zx=plt.subplot(122)
plt.imshow(im4)
zx.set_title("After Color Correction")

# Brightness adjustment and gamma encoding
gray=rgb2gray(im4)
plt.figure()
for i in range(15):
    num=np.round(i/15,2)
    im5a=np.clip(im4/(np.average(gray)/num),0,1)
    ax=plt.subplot(3, 5, i+1)
    plt.imshow(im5a)
    ax.set_title("Scaling Number{}".format(num))    
im5=np.clip(im4/(np.average(gray)/.38),0,1)
plt.figure()
plt.imshow(im5)

# Tonal repreduction
im6=np.zeros_like(im5)
im6[im5<.0031308]=12.92*im5[im5<.0031308]
im6[im5>.0031308]=(1+.055)*im5[im5>.0031308]**(1/2.4)-.055
plt.figure()
plt.imshow(im6)

# Compression
skio.imsave("Uncompressed.png",im6)
skio.imsave("Compressed95.jpg",im6, quality=95)
skio.imsave("Compressed90.jpg",im6, quality=90)
skio.imsave("Compressed85.jpg",im6, quality=85)
skio.imsave("Compressed80.jpg",im6, quality=80)
skio.imsave("Compressed75.jpg",im6, quality=75)
skio.imsave("Compressed70.jpg",im6, quality=70)
skio.imsave("Compressed65.jpg",im6, quality=65)
skio.imsave("Compressed60.jpg",im6, quality=60)
skio.imsave("Compressed50.jpg",im6, quality=50)

# Manual White Balencing
plt.figure()
averages=np.average(im6[2117:2213,147:226,:],(0,1))
Divisor=[averages[0]/averages[1],1,averages[2]/averages[1]]
im7=im6/Divisor
plt.imshow(im7)