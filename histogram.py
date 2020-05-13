import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
image = np.asarray(Image.open('/Users/jindeshubham/Desktop/handwritten_recognition/image-data/P583-Fg002-R-C01-R01-binarized.jpg'))
img_height, img_width = image.shape
hist=[]
for i in range(0,img_height):
    count=0
    for j in range(0,img_width):
        if(image[i][j] == 0):
            count = count + 1
    hist.append(count)

previous=0
min=0
i=0
accumalated=0
start=[]
descend=0
potential_peak=0
peak=[]

def check_highest(i,hist,range):
    img_height_tmp = hist[i:i+range]
    idx=img_height_tmp.index(max(img_height_tmp))
    return idx

while(i<img_height):
    print("I is ",i)
    if(hist[i]==0):
        i=i+1
        continue
    else:
        if(hist[i]>previous):    ##Peak ascending or start of the peak
          if(accumalated==0):
             accumalated=1
             start.append(i)
             i=i+1

          else:
              accumalated=accumalated+1
              i=i+1

          previous=hist[i]
          continue
        elif(hist[i]<=previous):
              if(accumalated<20):      ###Previous was not the peak as it might some pixalation issue
                previous=hist[i]
                accumalated=accumalated+1
                i=i+1
                continue
              else:
                idx=check_highest(i,hist,20)       ###This can be potential peak we will check for some more values.
                i=i + idx
                peak.append(i)
                accumalated=0
                continue


print("start ",start)
print("peak  ",peak)







