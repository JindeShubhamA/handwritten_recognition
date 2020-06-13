from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics
from collections import OrderedDict
import os
from character_segmentation import character_seg
import sys

#Global Variables
filter_size = 50
peaks=[]
valleys = []


'''
   Data-Structure to store information about points to be 
   traversed using A* algorithm
'''
class Point:
    def  __init__(self, x, y):
        self.x = x
        self.y = y

class Node:
    def __init__(self, point, parent, f_score, g_score):
        self.cordinates = point
        self.f_score = f_score
        self.g_score = g_score
        # self.h_score = h_score
        self.parent  = parent
        self.visited = True

'''
Segment Words
@:param sentence_img - Segmented Text Line
@:returns - Input Text Line with words segmentation
TODO Logic can be further improved
'''

def segment_words(sentence_img):
    sentence_img_cp = np.copy(sentence_img)
    sent_hgt, sent_wid = sentence_img_cp.shape
    for i in range(0, sent_hgt):
        for j in range(0, sent_wid):
            if(sentence_img_cp[i][j] < 150):
                sentence_img_cp[i][j] = 0
            else:
                sentence_img_cp[i][j] = 1
    vertical_projection = []
    for i in range(0,sent_wid):
        sum = 1
        for j in range(0,sent_hgt):
            sum = sum + sentence_img_cp[j][i]
        vertical_projection.append(sum)
    plt.plot(vertical_projection)
    gap_lnt = 0
    gap_str = []
    for i in range(0, sent_wid):
        if (vertical_projection[i] == sent_hgt):
            gap_lnt = gap_lnt + 1
        elif (vertical_projection[i] != sent_hgt):
            if (gap_lnt != 0):
                gap_str.append(gap_lnt)
            gap_lnt = 0
    print("Gap Length ", gap_lnt)
    average_length = statistics.mean(gap_str)
    threshold_lgt = int(average_length/3)

    gap_lnt = 0
    word_cut = []
    for i in range(0, sent_wid):
        if (vertical_projection[i] == sent_hgt):
            gap_lnt = gap_lnt + 1
        else:
            if (gap_lnt != 0):
                if (gap_lnt >= threshold_lgt):
                    word_cut.append(i)
            gap_lnt = 0
    for i in range(0, len(word_cut)):
        sentence_img[0:sent_hgt, word_cut[i] - 5] = 100
    return sentence_img


'''
Check for potential peaks in the lst of histogram. 
@:param idx - Index location to check for potential peak
@:param histogram -  Lst of Horizontal/vertical Projection
@:param filter_size - Range to check
@:returns - 0 - No peak
            1 - Peak   
'''
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
    if (curr_val >= max_value):
        return 1
    return 0

'''
Return the idx with min value in the range 
@:param start - Starting position of the list
@:param end   - End position of the list
@:param histogram - List to search in
@:return - Idx of min position  
'''
def return_min_idx(start,end,histogram):
    hist = histogram[start:end]
    idx=hist.index(min(hist))
    return start + idx

'''
Return the idx with max value in the range 
@:param start - Starting position of the list
@:param end   - End position of the list
@:param histogram - List to search in
@:return - Idx of max position  
'''
def return_max_idx(start,end,histogram):
    hist = histogram[start+50:end-50]
    idx=hist.index(max(hist))
    return start + idx


'''
Checks if the direct path exists (straight line with no road blocks) 
@:param start - Starting position of the list
@:param end   - Ending position of the list
@:search_arr  - Array to search 
@:return  False - No direct path exist
          True  - Direct path    exist
'''
def straight_path_exist(start,end,search_arr):
    if end>img_width:
        end = img_width
    search_arr_sub = search_arr[start:end]
    for element in search_arr_sub:
        if(element==0):
            return False
    return True

'''
Eucledian heuristics for A* algorithm
'''
def heuristics(Point1, Point2):
    return ((Point1.x - Point2.x)**2) + ((Point1.y - Point2.y)**2)

'''
Checks if the given point is traversable
'''
def transverable(Point,grid):
    # print("Coordinate \n", grid[Point.x][Point.y])
    # print("Point \n",Point.x, Point.y)
    if(grid[Point.x][Point.y] > 200):

        return True

    return False


'''
G score calculation
'''
def g_score_cal(direction):
    return float(math.sqrt(direction[0]*direction[0] + direction[1]*direction[1]))

'''
Checks if the pnt exist in the lst
'''
def check_in_lst(pnt, lst):
    for elmnt in lst:
        if elmnt.cordinates == pnt:
            return True, elmnt
    return False, None

'''
Return all the blocks in the 1 dimensional grid
'''
def blocks_on_road(grid):
    lst = []
    for i in range(0,len(grid)):
        if grid[i] < 100:
            lst.append(i)
    return lst

# def check_in_grid(row,Point):
#     if abs(Point.x - row)> 20
#         return False

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


'''
Standard A* algorithm
'''

# def a_star(start_x,end, row, grid,grid_width,grid_height,win_h):
#     open_lst = []
#     close_lst = []
#     start_pt = Point(row, start_x)
#     end_pt = Point(row, end)
#     g_score = 0
#     h_score = heuristics(start_pt,end_pt)
#     f_score = g_score + h_score
#     strt_node = Node(start_pt,None, f_score, g_score, h_score)
#     directions = [[0,1],[1,1],[-1,1],[1,0],[-1,0]]
#     open_lst.append(strt_node)
#     while(1):
#         f_score_min = 40000 ** 2
#         for node_elemnt in open_lst:
#             if(node_elemnt.f_score < f_score_min):
#                 current = node_elemnt
#                 f_score_min = node_elemnt.f_score
#         open_lst.remove(current)
#         close_lst.append(current)
#         if(current.cordinates.x == row and current.cordinates.y == end-1):
#             return current
#         for direction in directions:
#             x_pt = current.cordinates.x + direction[0]
#             y_pt = current.cordinates.y + direction[1]
#             neighbr_pt = Point(x_pt, y_pt)
#             if ((neighbr_pt.x > min(grid_height, start.x + win_h / 2)) or (neighbr_pt.x < max(0, start.x - win_h / 2)) or (
#                     neighbr_pt.y < 0) or (neighbr_pt.y > grid_width)):
#                 continue
#
#             if not transverable(neighbr_pt, grid):
#                 continue
#             cls_lst_cordinates = [cls.cordinates for cls in close_lst]
#             exist_flag = 0
#             for cls_lst_elmnt in cls_lst_cordinates:
#                 if(cls_lst_elmnt.x == neighbr_pt.x and cls_lst_elmnt.y == neighbr_pt.y):
#                     exist_flag=1
#                     break
#             if(exist_flag):
#                  #print("Present in the closed list")
#                  continue
#             g_scr = g_score_cal(direction)
#             # g_scr = new_g_score_cal(direction, current.cordinates, image, win_h) #+ current.g_score
#             h_scr = heuristics(neighbr_pt,end_pt)
#             f_scr = g_scr + h_scr + current.g_score
#             flag, elemt=check_in_lst(neighbr_pt, open_lst)
#             if(flag):
#                 if f_scr < elemt.f_score:
#                     open_lst.remove(elemt)
#                     new_elmnt = Node(neighbr_pt, current, f_scr, g_scr, h_scr)
#                     open_lst.append(new_elmnt)
#                 else:
#                     continue
#             else:
#                 new_elmnt = Node(neighbr_pt, current, f_scr, g_scr, h_scr)
#                 open_lst.append(new_elmnt)
def a_star(start, end, image, win_h):  # grid_width,grid_height):
    im_hgt, im_wid = image.shape
    open_lst = []
    close_lst = []
    g_score = 0
    h_score = heuristics(start, end)
    f_score = g_score + h_score
    strt_node = Node(start, None, f_score, g_score)
    print(str(start.x) + ' : ' + str(start.y) + '\n')
    directions = [[0, 1], [1, 0], [1, 1], [-1, 1], [-1, 0]]  # right, top, top_right, bottom_right, bottom
    open_lst.append(strt_node)
    counter = 0
    while (len(open_lst) > 0):
        counter += 1
        # print('Length of open_list:' + str(len(open_lst)) + '\n')
        f_score_min = math.inf
        for node_elemnt in open_lst:

            # find the element with lowest f_value in open list
            if (node_elemnt.f_score < f_score_min):  # ordered list instead of traversing it every iteration
                current = node_elemnt
                f_score_min = node_elemnt.f_score

        open_lst.remove(current)
        close_lst.append(current)

        if (counter == 5000):
            print('more than 5000 steps explored \n')

        # if we are at the goal then return
        if (current.cordinates.x == end.x and current.cordinates.y == end.y):  # .x and current.cordinates.y == end.y):
            return current

        for direction in directions:
            x = current.cordinates.x + direction[0]
            y = current.cordinates.y + direction[1]
            neighbr = Point(x, y)

            # Checking whether we are still in the window and neighbour is white. Maybe win_h * 1.5 or based on maxima
            if ((neighbr.x > min(im_hgt, start.x + win_h / 2)) or (neighbr.x < max(0, start.x - win_h / 2)) or (
                    neighbr.y < 0) or (neighbr.y > end.y)):
                continue

            # Already in the closed list?
            cls_lst_cordinates = [cls.cordinates for cls in close_lst]  # do not rebuild it all the time
            exist_flag = 0
            for cls_lst_elmnt in cls_lst_cordinates:
                if (cls_lst_elmnt.x == neighbr.x and cls_lst_elmnt.y == neighbr.y):
                    exist_flag = 1
                    break
            if (exist_flag):
                # print("Present in the closed list")
                continue

            # f_score computation
            g_scr = g_score_cal(direction) + current.g_score
            # g_scr = new_g_score_cal(direction, current.cordinates, image, win_h) #+ current.g_score
            h_score = heuristics(neighbr, end)  # + min_dist(current.cordinates, image, win_h)
            if not transverable(neighbr, image):  # needs to be able to cut through blacks eventually
                # print('before: ' + str(h_score) + '\n')
                h_score += 80000
                # print('after: ' + str(h_score) + '\n')
            f_scr = h_score + g_scr  # g_scr + h_scr + current.g_score
            # if it is already in the open list, keep only the one with the lower f-score otgerwise just save it
            flag, elemt = check_in_lst(neighbr, open_lst)

            if (flag):
                if f_scr < elemt.f_score:  # is this even a possibility in un weighted graph like this?
                    open_lst.remove(elemt)
                    new_elmnt = Node(neighbr, current, f_scr, g_scr)
                    open_lst.append(new_elmnt)
                    # print('Replacing element in open list [' + str(neighbr.x) + ', ' + str(neighbr.y) + ']\n' )
                else:
                    continue
            else:
                new_elmnt = Node(neighbr, current, f_scr, g_scr)
                open_lst.append(new_elmnt)
    print("Error: A* found no viable path")
    sys.exit()

file = "P106-Fg002-R-C01-R01-binarized"
image = np.asarray(Image.open('/Users/jindeshubham/Desktop/handwritten_recognition/image-data/'+file+'.jpg'))
img_height, img_width = image.shape
hist=[]

#Horizonal Projection of the image
for i in range(0,img_height):
    count=0
    for j in range(0,img_width):
        if(image[i][j] == 0):
            count = count + 1
    hist.append(count)

plt.plot(hist)
#plt.savefig("test1.jpg")



#####Get the peaks
for i in range(0,len(hist)):
    flag = check_peak(i,hist,filter_size)
    if(flag):
        if(hist[i]>100):
           peaks.append(i)

#Checks if any potental peak can be added. Logic is that if the difference between two peaks is large
#we add a peak which is the maximum of point between two peaks.
insert_peak_lst = []
for i in range(0,len(peaks)-1):
    start = peaks[i]
    end   = peaks[i+1]
    if (end - start > 150):
        idx = return_max_idx(start,end,hist)
        #print("Inserting another potential peak at ", id
        if(hist[idx]>0):
           insert_peak_lst.append(idx)

for p in insert_peak_lst:
    peaks.append(p)

peaks.sort()
peaks = list(OrderedDict.fromkeys(peaks))

ptential_peaks = []
for i in range(0,len(peaks)-1):
    frst = peaks[i]
    snd  = peaks[i+1]
    if (snd - frst) < 30:
        ptential_peaks.append(max(peaks[i],peaks[i+1]))
    else:
        ptential_peaks.append(peaks[i])
peak =ptential_peaks
peaks = list(OrderedDict.fromkeys(peaks))

####Get the valleys
for i in range(0,len(peaks)-1):
    start = peaks[i]
    end   = peaks[i+1]
    idx = return_min_idx(start,end,hist)
    valleys.append(idx)

####Potential small words
save_img =  np.zeros((img_height,img_width))

for i in range(0,img_height):
    for j in range(0,img_width):
        save_img[i][j] = image[i][j]
frst_ln = valleys[0]
lst_ln = valleys[-1]

avg_val_dis = 0
for i in range(0,len(valleys)-1):
    avg_val_dis += valleys[i+1]-valleys[i]
avg_val_dis = avg_val_dis/(len(valleys)-1)
####First and the last line.
valleys.insert(0,max(int(frst_ln-(avg_val_dis*1.5)),10))
valleys.insert(len(valleys),min(int(lst_ln+(avg_val_dis*1.5)),img_height-10))

# print("Peaks ",peaks)
# print("Valleys ",valleys)
###Remove the blocks

# for i in range(0,len(valleys)):
#     block_col =blocks_on_road(save_img[valleys][:])



# for i in range(0,img_width):
#     save_img[valleys[0]][i] = 100

'''
lowr_lines_max = []
upr_line_min = []
line_str = []

for i in range(0,len(valleys)):
    print("Valley is",valleys[i])
    end_nd = a_star(0, img_width, valleys[i], save_img,img_width-1,img_height-1)
    lower_line_max = 0
    upper_line_min = 99999
    line = []
    while(end_nd!=None):
        save_img[end_nd.cordinates.x][end_nd.cordinates.y] = 100
        line.append(end_nd.cordinates.x)
        if(end_nd.cordinates.x > lower_line_max):
            # lower_line = end_nd.cordinates.x
            lower_line_max = end_nd.cordinates.x
        if(end_nd.cordinates.x < upper_line_min):
            upper_line_min = end_nd.cordinates.x
            #upper_line = end_nd.cordinates.x
        end_nd = end_nd.parent
    line.reverse()
    line_str.append(line)
    upr_line_min.append(upper_line_min)
    lowr_lines_max.append(lower_line_max)

for iter in range(1,len(valleys)):
    upper_line = upr_line_min[iter-1]
    lower_line = lowr_lines_max[iter]
    img_cp = np.copy(save_img)
    print("Img width",img_width)
    for i in range(0,img_width):
        print("I is ",i)
        img_cp[0:line_str[iter-1][i],i] = 255
        img_cp[line_str[iter][i]:img_height,i] = 255

    img_cp = img_cp[upper_line:lower_line,:]
    # img_cp = segment_words(img_cp)
    save_ln = Image.fromarray(img_cp)
    if save_ln !='RGB':
         save_ln = save_ln.convert('RGB')
    save_ln.save(str(iter)+".jpg")


im = Image.fromarray(save_img)

if im.mode != 'RGB':
    im = im.convert('RGB')
im.save("Line_segmentation_result.jpg")
print("Done")
'''
###### LINE SEGMENTATION #####
save_img = image.copy()

copy = valleys[1:]
lines = []
for i in copy:
    end_node = a_star(Point(i, 0), Point(i, img_width - 1), save_img, avg_val_dis)
    line = []
    while (end_node != None):
        # save_img[end_node.cordinates.x][end_node.cordinates.y] = 100 #draw in the path
        line.append([end_node.cordinates.x, end_node.cordinates.y])
        end_node = end_node.parent
    line.reverse()
    lines.append(line)


#### SAVING RESULTS ####

def turn_over_y(image):
    h, w = image.shape
    turned = np.zeros((h, w))
    for r in range(h):
        for c in range(w):
            turned[r][w - 1 - c] = image[r][c]
    return turned


def extract_lines(image, upper, lower):
    min_row = np.min(upper[:, 0])
    height = np.max(lower[:, 0]) - min_row
    width = image.shape[1]
    segment = np.ones([height, width]) * 255
    rest, indeces = np.unique(upper[:, 1], return_index=True, axis=0)
    upper_u = upper[indeces]
    rest, indeces = np.unique(lower[:, 1], return_index=True, axis=0)
    lower_u = lower[indeces]
    for c in range(width - 1):
        for r in range(upper_u[c][0], lower_u[c][0]):
            segment[r - min_row][c] = image[r][c]
    return segment


for i in range(len(lines) - 1):
    line_img_array = extract_lines(save_img, np.array(lines[i]), np.array(lines[i + 1]))
    line_img = turn_over_y(line_img_array)
    line_img_sv = Image.fromarray(line_img_array)
    print(type(line_img))
    print(line_img.shape)

    # if not os.path.exists(file):
    #     os.makedirs(file)
    if line_img_sv.mode != 'RGB':
        line_img_sv = line_img_sv.convert('RGB')
    line_img_sv.save(file + "/Line_" + str(i) + ".jpg")# later add number of image aswell
    print("Image is ",str(i))
    character_seg(line_img, i)



























































