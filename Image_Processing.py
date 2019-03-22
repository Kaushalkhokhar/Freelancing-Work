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
    
img = img_read(r'D:\Programming\Python\Udemy\Freelancer_profile_work\Anu_Car_Cycle.jpg') 
height, width = img.shape[:2] 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
contours = cv2.findContours(img_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1] 

img_show('Gray Image', img_gray, height, width) 

'''
len(contours) 
contours[1]
reat = cv2.minAreaRect(contours[1])
box = cv2.boxPoints(rect) 
box = np.int0(box)
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

