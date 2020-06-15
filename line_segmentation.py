from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics
import sys
import os

import time
from scipy.signal import argrelextrema

#### Helper Function to retrieve peaks from Projection Profile ####

def check_peak(idx, histogram, filter_size):
    if (idx - (filter_size/2) <=0):
        start = 0
    else:
        start = int(idx - (filter_size/2))
    if ((idx + filter_size/2) >= len(histogram)):
        end = len(histogram)
    else:
        end = int(idx + (filter_size/2))
    curr_val =  histogram[idx]
    histogram[idx] = histogram[idx]*-1
    window  = histogram[start:end]
    max_value = max(window)
    histogram[idx] = histogram[idx]*-1
    if (curr_val > max_value):
        return 1
    return 0

def return_min_idx(start,end,histogram):
    hist = histogram[start:end]
    idx=hist.index(min(hist))
    return start + idx

def return_max_idx(start,end,histogram):
    hist = histogram[start+50:end-50]
    idx=hist.index(max(hist))
    return start + idx

#### A* Algorithm + Helper functions ####

class Point:
    def  __init__(self, x, y):
        self.x = x
        self.y = y

class Node:
    def __init__(self, point, parent, f_score, g_score):
        self.cordinates = point
        self.f_score = f_score
        self.g_score = g_score
        self.parent  = parent
        self.visited = True


def transverable(Point,grid):  
    ### idea: let's first cut through based on A* star heuristic, save the positions of line crossing
    ### and cut them again  for whatever gives the best match up with our pilot model to then exchange involved
    ### letters in the data
    if(grid[Point.x][Point.y] > 200):    
        return True
    return False

def g_score_cal(direction):
    return float(math.sqrt(direction[0]*direction[0] + direction[1]*direction[1]))

def heuristics(Point1, Point2):
    return (((Point1.x - Point2.x)**2 + (Point1.y - Point2.y)**2)*1.1)

def check_in_lst(pnt, lst):
    for elmnt in lst:
        if (elmnt.cordinates.x == pnt.x and elmnt.cordinates.y == pnt.y):
            return True, elmnt
    return False, None


def a_star(start ,end, image, win_h): #grid_width,grid_height):
    im_hgt, im_wid = image.shape
    open_lst = []
    close_lst = []
    g_score = 0
    h_score = heuristics(start,end)
    f_score = g_score + h_score
    strt_node = Node(start,None, f_score, g_score)
    print(str(start.x) + ' : ' + str(start.y) + '\n')
    directions = [[0,1],[1,0],[1,1],[-1,1],[-1,0]] # right, top, top_right, bottom_right, bottom
    open_lst.append(strt_node)
    counter = 0
    while(len(open_lst) > 0):
        counter += 1
        #print('Length of open_list:' + str(len(open_lst)) + '\n')
        f_score_min = math.inf
        for node_elemnt in open_lst:
            
            # find the element with lowest f_value in open list
            if(node_elemnt.f_score < f_score_min): #ordered list instead of traversing it every iteration
                current = node_elemnt
                f_score_min = node_elemnt.f_score

        open_lst.remove(current)
        close_lst.append(current)

        if (counter == 5000):
            print('more than 5000 steps explored \n')
        
        #if we are at the goal then return
        if(current.cordinates.x == end.x and current.cordinates.y == end.y): #.x and current.cordinates.y == end.y):
            return current

        for direction in directions:
            x = current.cordinates.x + direction[0]
            y = current.cordinates.y + direction[1]
            neighbr = Point(x, y)
            
            #Checking whether we are still in the window and neighbour is white. Maybe win_h * 1.5 or based on maxima
            if ((neighbr.x > min(im_hgt, start.x+win_h/2)) or (neighbr.x < max(0, start.x-win_h/2)) or (neighbr.y < 0) or (neighbr.y > end.y)):
                continue

            
            #Already in the closed list?
            cls_lst_cordinates = [cls.cordinates for cls in close_lst] #do not rebuild it all the time
            exist_flag = 0 
            for cls_lst_elmnt in cls_lst_cordinates:
                if(cls_lst_elmnt.x == neighbr.x and cls_lst_elmnt.y == neighbr.y):
                    exist_flag=1
                    break
            if(exist_flag):
                 #print("Present in the closed list")
                 continue

            #f_score computation
            g_scr = g_score_cal(direction) + current.g_score 
            #g_scr = new_g_score_cal(direction, current.cordinates, image, win_h) #+ current.g_score 
            h_score = heuristics(neighbr,end) #+ min_dist(current.cordinates, image, win_h)
            if not transverable(neighbr, image): #needs to be able to cut through blacks eventually
                # print('before: ' + str(h_score) + '\n')    
                h_score += 80000 
                # print('after: ' + str(h_score) + '\n')
            f_scr = h_score + g_scr #g_scr + h_scr + current.g_score
            #if it is already in the open list, keep only the one with the lower f-score otgerwise just save it
            flag, elemt=check_in_lst(neighbr, open_lst)

            if(flag):
                if f_scr < elemt.f_score: #is this even a possibility in un weighted graph like this?
                    open_lst.remove(elemt)
                    new_elmnt = Node(neighbr, current, f_scr, g_scr)
                    open_lst.append(new_elmnt)
                    #print('Replacing element in open list [' + str(neighbr.x) + ', ' + str(neighbr.y) + ']\n' )
                else:
                    continue
            else:
                new_elmnt = Node(neighbr, current, f_scr, g_scr)
                open_lst.append(new_elmnt)
    print("Error: A* found no viable path")
    sys.exit()
   

### Legacy in A*
def min_dist(position, image, win_h):
    C = 1000
    min_d = 0
    x= position.x
    y= position.y
    while(min_d < win_h/2 and image[x+min_d,y] > 200 and image[x-min_d,y] > 200):
        min_d += 1  
    return C/(1+min_d)

def new_g_score_cal(direction, position, image, win_h): #did not work as intended
    return float(math.sqrt(direction[0]*direction[0] + direction[1]*direction[1])) + min_dist(position, image, win_h)
    
#### END A* #####
   
#### Read in Input ####

#P123-Fg001-R-C01-R01-binarized
#P342-Fg001-R-C01-R01-binarized
#P123-Fg002-R-C01-R01-binarized
file = "P123-Fg002-R-C01-R01-binarized"
image = np.asarray(Image.open('/home/basti/Desktop/HWR/binarized/'+ file + ".jpg"))
img_height, img_width = image.shape
hist=[]
for i in range(0,img_height):
    count=0
    for j in range(0,img_width):
        if(image[i][j] == 0):
            count = count + 1
    hist.append(count)


plt.plot(hist)
plt.savefig("projection_profile.jpg")
filter_size = 75 ################################ one of the parameters so should be moved
peaks=[]
valleys = []
#####Get the peaks
for i in range(0,len(hist)):
    flag = check_peak(i,hist,filter_size)
    if(flag):
        if(hist[i]>100):
          peaks.append(i)

for i in range(0,len(peaks)-1):
    start = peaks[i]
    end   = peaks[i+1]
    if (end - start > 300):
        idx = return_max_idx(start,end,hist)
        #print("Inserting another potential peak at ", idx)
        peaks.insert(i+1,idx)

####Get the valleys
for i in range(0,len(peaks)-1):
    start = peaks[i]
    end   = peaks[i+1]
    idx = return_min_idx(start,end,hist)
    valleys.append(idx)      

#Add first and last line- start point based on average line spacing
        
frst_ln = valleys[0]
lst_ln = valleys[-1]
avg = 0
for i in range(0,len(valleys)-1):
    avg += valleys[i+1]-valleys[i]
avg = avg/(len(valleys)-1)

valleys.insert(0,max(0, frst_ln-round(1.5*avg)))
valleys.append(min(img_height, lst_ln+round(1.5*avg)))


###### LINE SEGMENTATION #####

def turn_over_y(image):
    h, w = image.shape
    turned = np.zeros((h,w))
    for r in range(h):
        for c in range(w):
            turned[r][w-1-c] = image[r][c]
    return turned

save_img = image.copy()  

save_img = turn_over_y(save_img)

copy = valleys[2:4]

lines = []
for i in valleys:
    end_node = a_star(Point(i,0), Point(i,img_width-1), save_img, avg)
    line = []
    while(end_node!=None):
        #save_img[end_node.cordinates.x][end_node.cordinates.y] = 100 #draw in the path
        line.append([end_node.cordinates.x, end_node.cordinates.y])
        end_node = end_node.parent
    line.reverse()
    lines.append(line) 

#### SAVING RESULTS ####


def extract_lines(image, upper, lower):
    min_row = np.min(upper[:, 0])
    height = np.max(lower[:, 0]) - min_row
    width = image.shape[1]
    segment = np.ones([height, width])*255
    rest, indeces = np.unique(upper[:,1], return_index = True, axis=0)
    upper_u = upper[indeces]
    rest, indeces = np.unique(lower[:,1], return_index = True, axis=0)
    lower_u = lower[indeces]
    for c in range(width-1):
        for r in range(upper_u[c][0], lower_u[c][0]):
            segment[r-min_row][c] = image[r][c]
    return segment

for i in range(len(lines)-1):
    line_img_array = extract_lines(save_img, np.array(lines[i]), np.array(lines[i+1]))
    line_img_array = turn_over_y(line_img_array)
    line_img = Image.fromarray(line_img_array)
    
    if not os.path.exists(file):
        os.makedirs(file)
    if line_img.mode != 'RGB':
        line_img = line_img.convert('RGB')
    line_img.save(file + "/Line_" + str(i) + ".jpg") #later add number of image aswell

#### Print out cutting lines ####
### (requires to uncomment line in main while loop, about line 250) ##

# save_img_inv =  np.zeros((img_height,img_width))  #turned_over_y
# for i in range(0,img_height):
#     for j in range(0,img_width):
#         save_img_inv[i][(img_width-1) -j] = save_img[i][j]
        
# im = Image.fromarray(save_img_inv)

# if im.mode != 'RGB':
#     im = im.convert('RGB')
# im.save("Cutting_Lines.jpg")
# print("Done")


