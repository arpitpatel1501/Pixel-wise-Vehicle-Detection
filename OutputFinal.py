import cv2
import numpy as np

import random
import scipy
import scipy.io
import os

import csv

from matplotlib import pyplot as plt
"""# **Read Image**"""

def readImage(filename):
    """
     Read in an image file, errors out if we can't find the file
    :param filename:
    :return: Img object if filename is found
    """
    img = cv2.imread(filename, 0)
    img = cv2.resize(img, (120, 120), interpolation = cv2.INTER_AREA)
    if img is None:
        print('Invalid image:' + filename)
        return None
    else:
        print('Image successfully read...')
        print(img.shape)
        #plt.imshow(img)
        #plt.show()
        
    return img

def findVehicle_S(img, window_size):

    height = img.shape[0]
    width = img.shape[1]
    
    
    offset = int(window_size/2)
    print("offset: ",offset)

    newFrame = np.random.random((height-2*offset,width-2*offset))
    
    print("offset: ", offset)
    print ("Window start...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            #Calculate sum of squares
            windowIxx = img[y-offset:y+offset+1, x-offset:x+offset+1]
            
            window = np.where(windowIxx>100,1,0)
            
            windowSum = window.sum()
            newFrame[y-offset][x-offset] = windowSum/(window_size**2)
            
    return newFrame


#######################################################################################

def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""

    plt.imshow(image)
    y, x = np.transpose(filtered_coords)
    plt.plot(x, y, 'b.')
    plt.axis('off')


def findVehicle_E(img, window_size):
    height = img.shape[0]
    width = img.shape[1]
    
    
    offset = int(window_size/2)
    print("offset: ",offset)
    newFrame = np.random.random((height-2*offset,width-2*offset))
    
    print("offset: ", offset)
    #Loop through image and find our corners
    print ("Window start...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            #Calculate sum of squares
            windowIxx = img[y-offset:y+offset+1, x-offset:x+offset+1]
            
            window = np.where(windowIxx>125,1,0)
            
            windowSum = window.sum()
            newFrame[y-offset][x-offset] = windowSum/(window_size**2)
    return newFrame

#######################################################################################

def findVehicle_C(img, window_size):
    height = img.shape[0]
    width = img.shape[1]
    
    offset = int(window_size/2)
    print("offset: ",offset)
    newFrame = np.random.random((height-2*offset,width-2*offset))
    
    print("offset: ", offset)
    #Loop through image and find our corners
    print ("Window start...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            #Calculate sum of squares
            windowIxx = img[y-offset:y+offset+1, x-offset:x+offset+1]
            
            windowSum = windowIxx.sum()
            newFrame[x-offset][y-offset] = windowSum/(window_size**2)
    return newFrame


#######################################################################################

#img = readImage("/content/drive/My Drive/PGM_Project/Frames/frame208.jpg")
image = readImage("frame276.jpg")


######################## for S ################################
img=image
print(img)
newFrame = findVehicle_S(img, 7)
print("New Frame: ", newFrame)
#plt.imshow(newFrame)
#plt.show()
print("SizeNewFrame: ",newFrame.shape)
print(newFrame.max())



newFrame = np.where(newFrame>0.5, 1, 0);

#plt.imshow(newFrame*255)
#plt.show()

result_s = newFrame.ravel()
print("result: ", result_s)
print("result_size: ",result_s.shape)
'''
current_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_dir, "s_output/" + 's_result.mat')
scipy.io.savemat(path, {'result_s': result_s})

path = os.path.join(current_dir, "final_output/" + 's_result.mat')
scipy.io.savemat(path, {'result_s': result_s})
'''
######################## for E ################################

img=image

img = cv2.resize(img, (120, 120), interpolation = cv2.INTER_AREA)
img = np.where(img>100,255,0)
print(img.dtype)
img = np.uint8(img)
print(img.dtype)
print("\nType: ", img)

#plt.imshow(img)
#plt.show()


edges = cv2.Canny(img,0,255,L2gradient=True)

print(img.shape)

pixel= img[119, 119]
print(edges)
#plt.imshow(edges,cmap = 'gray')
#plt.show()


newFrame = findVehicle_E(edges, 7)

newFrame = np.where(newFrame>0.28, 1, 0)

#plt.imshow(newFrame*255)
#plt.show()

print("New Frame: ", newFrame)
print("max in  new: ", newFrame.max())

result_edge = newFrame.ravel()
print("result: ", result_edge)
print("result_size: ",result_edge.shape)
'''
current_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_dir, "edge_output/" + 'edge_result.mat')
scipy.io.savemat(path, {'result_edge': result_edge})

path = os.path.join(current_dir, "final_output/" + 'edge_result.mat')
scipy.io.savemat(path, {'result_edge': result_edge})
'''
corner = np.float32(img) 

dest = cv2.cornerHarris(img, 7, 7, 0.04)
#print(dest)
dest = cv2.dilate(dest, None) 

'''
plt.subplot(121),plt.imshow(newFrame*255,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
'''
######################## for C ################################



img = image
    #img = cv2.imread(filename, 0)
img = cv2.resize(img, (120, 120), interpolation = cv2.INTER_AREA)
img1=img
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)


corners = cv2.goodFeaturesToTrack(img1, 150, 0.5, 10)
corners = np.int0(corners)

corner_frame = np.zeros((120,120))

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    print("\nx: "+str(x)+ " y: "+str(y))
    corner_frame[x][y] = 1
    
print("\ncheck_sum", np.sum(corner_frame))
#print(img)
y=img-img1
#print(y[10])
#cv2.imshow("img", img)

#plt.imshow(img)
#plt.show()

newFrame = findVehicle_C(corner_frame, 7)
print("New Frame: ", newFrame)

print("max in  new: ", newFrame.max())

#plt.imshow(np.real(resize_img))
#plt.show()
print("SizeNewFrame: ",newFrame.shape)


newFrame = np.where(newFrame>=0.0203,1,0)

#plt.imshow(newFrame*255)
#plt.show()

result_corner = newFrame.ravel()
print("result: ", result_corner)
print("result_size: ",result_corner.shape)
'''
current_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_dir, "corner_output/" + 'corner_result.mat')
scipy.io.savemat(path, {'result_corner': result_corner})

path = os.path.join(current_dir, "final_output/" + 'corner_result.mat')
scipy.io.savemat(path, {'result_corner': result_corner})
'''
cv2.waitKey(0)

#print(img.max())

cv2.destroyAllWindows()
####################################################################
#image = readImage("Label_276.jpg")
img = cv2.imread("Label_276.png",0)
img = cv2.resize(img, (120, 120), interpolation = cv2.INTER_AREA)

#plt.imshow(img*255)
#plt.show()

print("shape: ", img.shape)
print("\nmax: ", np.max(img))
print("\nsum: ", np.sum(img))

result_ground = newFrame.ravel()
print("result: ", result_ground)
print("result_size: ",result_ground.shape)

########################################################################
cnt=0
with open('mycsv_reduce_1.csv', 'w', newline='') as f:
	mywriter = csv.writer(f)
	mywriter.writerow(['S', 'E', 'C', 'Ground'])
	
	for i in range(len(result_ground)):
		if not any ([result_s[i] , result_edge[i] , result_corner[i] , result_ground[i] , cnt>100]):
			mywriter.writerow([result_s[i], result_edge[i], result_corner[i], result_ground[i]])
			cnt=cnt+1
		if any([result_s[i] , result_edge[i] , result_corner[i] , result_ground[i]]):
			mywriter.writerow([result_s[i], result_edge[i], result_corner[i], result_ground[i]])
	
