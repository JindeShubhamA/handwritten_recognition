import PIL
from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import os
from random import randrange
from collections import Counter


class Label:
    def __init__(self, pics, label,style):
        self.data = pics
        self.label = label
        self.style = style
        self.dist = np.inf
    def __lt__(self,other): # needs to be defined to be able to sort the list with labels
        return self.dist < other.dist
    
    
# copied stuff (including some compability changes)#
####################################################

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = folds
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = row
			test_set.append(row_copy)
			row_copy.style = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row.style for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores


### The next 3 functions are actually used in KNN, rest is evaluation###
def euclidean_distance(r1, r2):
    dist = 0.0
    for i in range(len(r1)-1):
        dist = dist + (r1[i] - r2[i])**2
    dist = math.sqrt(dist)
    return dist

def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row.data)
        train_row.dist = dist
        distances.append(train_row)
    distances.sort()
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i])
    return neighbors
 

def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row.style for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
#######################################################################



def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row.data, num_neighbors)
		predictions.append(output)
	return(predictions)


# copied stuff end#################################
###################################################

def makePerspHist(img):
    norm_size = 64
    up_filter = PIL.Image.NEAREST
    img = Image.fromarray(img)
    img_norm = img.resize((norm_size,norm_size),up_filter)
    img_norm = np.asarray(img_norm)
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
        d = d+1
    ph = np.concatenate(ph)
    return ph

def showResult(res):
    result = Counter(res)
    for styles in result:
        print(str(styles) + ': ' + str(result[styles]))
#########################################################################
###### Main Code ########################################################
#########################################################################

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


# next import characters from file and transform them into histograms
char_list = []

i = 0
for style in images:
    c = 0
    for letter in style:
        pic_list = []
        for pic in letter:
            img = np.asarray(Image.open(path+style_type[i]+'/'+letter_type[i][c]+'/'+pic))
            img_list = makePerspHist(img)
            #pic_list.append(img_list)
            char_list.append(Label(img_list,pic, style_type[i]))
        c = c+1
    i = i+1

### Import of evaluation data would happen here:    

import_path = 'c:/Users/nikee/Desktop/hwr_style/test_data/'  

import_data = os.listdir(import_path)
test_data = []
for picture in import_data:
    img = np.asarray(Image.open(import_path+'/'+picture))
    img = cv2.bitwise_not(img[:,:,2]) # invert colors + make the array 1 dimensional
    test_data.append(makePerspHist(img))


# Uses a train data row to test prediction
#test_data = char_list[3].data


# implement KNN
num_neighbors = 5
#n_folds = 5
## Current KNN test.
#label = predict_classification(char_list, test_data, num_neighbors)
#print(label)

#evaluation = evaluate_algorithm(char_list, k_nearest_neighbors, n_folds, num_neighbors)
#not sure how this actually works, but evaluation results are not overwhemling (=Very bad)
#however this might be because incomplete sample sets results in very big problems during cross validation


# Now this would run for a set of characters
labels = []

for chars in test_data:
    labels.append(predict_classification(char_list, chars, num_neighbors))
#Summ results up
showResult(result)
