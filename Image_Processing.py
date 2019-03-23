import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import cv2

def img_read(image):
    return cv2.imread(image)

def img_show(title,image, height, width):
    
    #plt.figure(figsize=(12,8))
    image = cv2.resize(image,(round(height/5),round(width/5)))
    cv2.imshow(title,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    
    '''b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()'''    
    
img = img_read(r'D:\Programming\Python\Udemy\Freelancer_profile_work\Image_Processing\Fuse__133.jpg') 
img = img[459:2220,217:3251]
img = img[0:, 0:450]
height, width = img.shape[:2] 

# gray Image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

'''plt.figure()
plt.subplot(1,2,1)
plt.hist(img_gray.ravel(),256,[0,256])
plt.subplot(1,2,2)
plt.imshow()
plt.show()'''

# Filter for smoothening
# img_blur = cv2.GaussianBlur(img, (5,5), 0)
# img_blur = cv2.medianBlur(img_gray,5)
img_blur = cv2.bilateralFilter(img_gray,5,250,250)

# Binarization (Thresholding)
# ret,img_thre = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY) # Simple Threshold Image
# img_adt_thre = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11 ,2) # Adaptive Threshold Image
# img_adt_thre = cv2.adaptiveThreshold(img_gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
ret, img_ots_thre = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu's Binarization

# Geomatric Transformation 
'Used to chnge the geomatri of image'

'''rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(cols,rows))

plt.figure(figsize=(10,6))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()'''

# Morphological Transformation
"""

kernel = np.ones((5,5), np.int8) 
'''
# Kernel can be foud by structuring element of any shape like circular, rectangle, cross etc
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) # for rectangular kernel
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) # for Ellipse kernel
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) # for Croos kernel

'''
img_mor_ero = cv2.erode(img_ots_thre, kernel, iterations = 1)
img_mor_dia = cv2.dilate(img_ots_thre, kernel, iterations = 1)
img_mor_ope = cv2.morphologyEx(img_ots_thre, cv2.MORPH_OPEN, kernel)
img_mor_clo = cv2.morphologyEx(img_ots_thre, cv2.MORPH_CLOSE, kernel)
img_mor_grd = cv2.morphologyEx(img_ots_thre, cv2.MORPH_GRADIENT, kernel) # Black Hat and Top Hat also there to perform

plt.figure(figsize=(15,12))
plt.subplot(321), plt.imshow(img_ots_thre), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(322), plt.imshow(img_mor_ero), plt.title('Erosion Image'), plt.xticks([]), plt.yticks([])
plt.subplot(323), plt.imshow(img_mor_dia), plt.title('Dilation Image'), plt.xticks([]), plt.yticks([])
plt.subplot(324), plt.imshow(img_mor_ope), plt.title('Opening Image'), plt.xticks([]), plt.yticks([])
plt.subplot(325), plt.imshow(img_mor_clo), plt.title('Closing Image'), plt.xticks([]), plt.yticks([])
plt.subplot(326), plt.imshow(img_mor_grd), plt.title('Morpholigivcal gradient Image'), plt.xticks([]), plt.yticks([])
#plt.suptitle('Morphological Trasformation on Binary Image', fontsize = 16)
plt.tight_layout()
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)
plt.show()
"""

# Image Gradients
'''
It detects edges of an image
'''

"""'''laplacian = cv2.Laplacian(img_ots_thre,cv2.CV_64F)
sobelx = cv2.Sobel(img_ots_thre,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img_ots_thre,cv2.CV_64F,0,1,ksize=5)'''


# Important point White to black edge is missed by it. So we need to convert to absoulute value.
laplacian = np.int8(np.absolute(cv2.Laplacian(img_ots_thre,cv2.CV_64F)))
sobelx = np.int8(np.absolute(cv2.Sobel(img_ots_thre,cv2.CV_64F,1,0,ksize=5)))
sobely = np.int8(np.absolute(cv2.Sobel(img_ots_thre,cv2.CV_64F,0,1,ksize=5)))

plt.figure(figsize=(15,12))
plt.subplot(221), plt.imshow(img_ots_thre), plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(laplacian), plt.title('laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(sobelx), plt.title('sobelx'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(sobely), plt.title('sobely'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
"""
# Canny Endge Detection

canny = cv2.Canny(img_gray, 100, 200, 3, L2gradient=True)

# To find Contours

contours, _ = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
mask = img
for cnt in contours:
    '''
    # Rotated Rectangle
    #print(cnt)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.array(box).reshape((-1,1,2)).astype(np.int32)
    img_cont = cv2.drawContours(mask, [box], 0, (0,255,0), 3)
    #img_show(img)
    '''

    '''
    # Straight Bounding Rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    '''

    # Minimum Enclosing Circle
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    img = cv2.circle(img,center,radius,(0,255,0),2)

plt.figure(figsize=(12,9))
plt.subplot(121), plt.imshow(img), plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img_cont), plt.title('Recognized'), plt.xticks([]), plt.yticks([])
plt.show()


"""# image Ploting
title = ['Original Gray Image', 'Simple Trehsold Image', \
        'Adaptive Threshold Iamge', "Otsu's Binarization Image" ]
img_plot = [img_gray, img_thre, img_adt_thre, img_ots_thre]

'''plt.imshow(img_plot[1])
plt.show()'''

plt.figure(figsize=(12,15))
# f.suptitle('Blured Image', fontsize = 16)

for i in range(4):
    # print(i)
    plt.subplot(2, 2, i + 1), plt.imshow(img_plot[i])
    plt.title(title[i]), plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # temp_img = img_plot[i+1]
    '''plt.subplot(3, 2, i + 2), plt.imshow(img_plot[i +1])
    plt.title(titel[i + 1]), plt.xticks([]), plt.yticks([])
    '''
plt.suptitle('Blured(GaussianBlur) before Gray', fontsize = 16)
plt.show()

# img_show('Gray Image', img_gray, height, width) 
#contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 
# print(contours.shape, _.shape)
# print(contours[301])

'''
len(contours) 
contours[1]
reat = cv2.minAreaRect(contours[1])
box = cv2.boxPoints(rect) 
box = np.int0(box)
'''

'''
for cnt in contours:
    #print(cnt)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.array(box)
    img_with_cntr = cv2.drawContours(img_gray, box.astype(int), 0, (0,255,0), -1)
    #img_show(img)

# img = cv2.drawContours(img, [box], 0, (0, 255, 0), 3) 

img_show('Contours Image',img_with_cntr, height, width)
'''"""