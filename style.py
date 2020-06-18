import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics
import sys
import time
import cv2
import os
from scipy.signal import argrelextrema

class Character:
    def __init__(self, pic, label):
        self.data = pic
        self.label = label
        self.archaic = 0
        self.hasmonean = 0
        self.herodian = 0

class Label:
    def __init__(self, pics, label):
        self.data = pics
        self.label = label
        

def makePerspHist(img):
    norm_size = 64
    up_filter = PIL.Image.NEAREST
    img = Image.fromarray(img)
    img_norm = img.resize((norm_size,norm_size),up_filter)
    img_norm = np.asarray(img_norm)
    #with np.printoptions(threshold=np.inf):
     #   print(img_norm)
    #print(img)
    # this needs to be done for 4 different perspectives
    ph = np.zeros([4,(norm_size*2)-1], dtype=int)
    # Horizontal  (sum x)
    for y in range(norm_size):
        for x in range(norm_size):
            if(img_norm[y,x] == 0):
                ph[0,y] = ph[0,y]+1
    # Vertical (sum y)
    for x in range(norm_size):
        for y in range(norm_size):
            if(img_norm[y,x] == 0):
                ph[1,x] = ph[1,x]+1
    # Diagonal 1 and 2
    d = 0
    img_norm_flip = np.fliplr(img_norm) # need to flip so the other diagonal is used as well // np.diagonal sucks
    for i in range(-(norm_size-1), norm_size-1):
        for t in np.diagonal(img_norm,i,0,1):
            if t == 0:
                ph[2,d] = ph[2,d]+1
        for t in np.diagonal(img_norm_flip,i,0,1):
            if t == 0:
                ph[3,d] = ph[3,d]+1
        
        #ph[2,d] = sum(np.diagonal(img_norm,i,0,1))
        #ph[3,d] = sum(np.diagonal(img_norm,i,1,0))
        d = d+1
    
    return ph

###############################
#testing    
test = np.zeros([4,4], dtype=int)
test[0,0] = 1    
test[0,1] = 2
test[1,0] = 3 
test[1,1] = 4 
#test = Image.fromarray(test)
#test_1 = test.resize((2,2),PIL.Image.NEAREST)
ph = makePerspHist(test)
print(ph)
###############################


# Assumption get pictures and labels from calssification

# First get style labels
path = 'c:/Users/nikee/Desktop/hwr_style/train_data/'
style_type = os.listdir(path)
i= 0
letter_type = []
for style in style_type:
    letter_type.append(os.listdir(path + style))
    i = i+1

images = []
i = 0
for style in style_type:
    image_per_style = []
    for letter in letter_type[i]:
        image_names =[]
        for image in os.listdir(path + style + '/'+ letter):
            if image.endswith(".png"):
                image_names.append(image)
        image_per_style.append(image_names)
    i = i+1
    images.append(image_per_style)



# test if there are any unique labels in each class (they would be pretty useless):
# this test can be excluded from the final version of the code
unique_list = []    
count_list = []    
for x in range(1,3):
    for y in letter_type[x]:
        if y not in unique_list:
            temp = y
            unique_list.append(temp)
        else:
            if y not in count_list:
                count_list.append(y)
if len(unique_list) != len(count_list):
    print("There are style unique classes")
else:
    print("All classes are there at least twice (YAAHY)")


# next import characters from file
char_list = []


i = 0
for style in images:
    c = 0
    for letter in style:
        pic_list = []
        for pic in letter:
            img = np.asarray(Image.open(path+style_type[i]+'/'+letter_type[i][c]+'/'+pic))
            img_list = makePerspHist(img)
            pic_list.append(img_list)
        char_list.append(Label(pic_list,style))
        c = c+1
    i = i+1


# normalize size for all input files

#  Create PHs (perspective Histograms) from 4 perspectives
# This could be expanded, or replaced with different feature extractors eg SIFT

# implement KNN


# Summ results up


folder=[string1+s for s in folders]

image = np.asarray(Image.open('/Users/nikee/Desktop/hwr_style




