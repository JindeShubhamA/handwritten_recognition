from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
class Point:
    def  __init__(self, x, y):
        self.x = x
        self.y = y

class Node:
    def __init__(self, point, parent, f_score, g_score, h_score):
        self.cordinates = point
        self.f_score = f_score
        self.g_score = g_score
        self.h_score = h_score
        self.parent  = parent
        self.visited = True



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

def straight_path_exist(start,end,search_arr):
    if end>img_width:
        end = img_width
    search_arr_sub = search_arr[start:end]
    for element in search_arr_sub:
        if(element==0):
            return False
    return True

def heuristics(Point1, Point2):
    return ((Point1.x - Point2.x)**2) + ((Point1.y - Point2.y)**2)

def transverable(Point,grid):
    print("Coordinate \n", grid[Point.x][Point.y])
    print("Point \n",Point.x, Point.y)
    if(grid[Point.x][Point.y] > 200):

        return True

    return False

def g_score_cal(direction):
    return float(math.sqrt(direction[0]*direction[0] + direction[1]*direction[1]))


def check_in_lst(pnt, lst):
    for elmnt in lst:
        if elmnt.cordinates == pnt:
            return True, elmnt

    return False, None

# def check_in_grid(row,Point):
#     if abs(Point.x - row)> 20
#         return False

def a_star(start_x,end, row, grid,grid_width,grid_height):
    #col_cnt = np.zeros()
    open_lst = []
    close_lst = []
    start_pt = Point(row, start_x)
    end_pt = Point(row, end)

    g_score = 0
    h_score = heuristics(start_pt,end_pt)
    f_score = g_score + h_score
    strt_node = Node(start_pt,None, f_score, g_score, h_score)

    directions = [[0,1],[1,1],[-1,1],[1,0],[-1,0]]
    open_lst.append(strt_node)

    while(1):
        f_score_min = 40000 ** 2

        for node_elemnt in open_lst:

            if(node_elemnt.f_score < f_score_min):
                current = node_elemnt
                f_score_min = node_elemnt.f_score

        open_lst.remove(current)
        close_lst.append(current)


        if(current.cordinates.x == row and current.cordinates.y == end-1):
            return current

        for direction in directions:
            x_pt = current.cordinates.x + direction[0]
            y_pt = current.cordinates.y + direction[1]



            neighbr_pt = Point(x_pt, y_pt)

            if ((neighbr_pt.x > grid_height) or (neighbr_pt.x < 0) or (neighbr_pt.y > grid_width) or (neighbr_pt.y < 0)):
                continue

            if not transverable(neighbr_pt, grid):
                continue

            cls_lst_cordinates = [cls.cordinates for cls in close_lst]
            exist_flag = 0
            for cls_lst_elmnt in cls_lst_cordinates:
                if(cls_lst_elmnt.x == neighbr_pt.x and cls_lst_elmnt.y == neighbr_pt.y):
                    exist_flag=1
                    break

            if(exist_flag):
                 print("Present in the closed list")
                 continue


            g_scr = g_score_cal(direction)
            h_scr = heuristics(neighbr_pt,end_pt)
            f_scr = g_scr + h_scr + current.g_score

            flag, elemt=check_in_lst(neighbr_pt, open_lst)

            if(flag):
                if f_scr < elemt.f_score:
                    open_lst.remove(elemt)
                    new_elmnt = Node(neighbr_pt, current, f_scr, g_scr, h_scr)
                    open_lst.append(new_elmnt)
                else:
                    continue
            else:
                new_elmnt = Node(neighbr_pt, current, f_scr, g_scr, h_scr)
                open_lst.append(new_elmnt)



image = np.asarray(Image.open('/Users/jindeshubham/Desktop/handwritten_recognition/image-data/P342-Fg001-R-C01-R01-binarized.jpg'))
img_height, img_width = image.shape
hist=[]
for i in range(0,img_height):
    count=0
    for j in range(0,img_width):
        if(image[i][j] == 0):
            count = count + 1
    hist.append(count)

plt.plot(hist)
plt.savefig("test1.jpg")
filter_size = 75
peaks=[]
valleys = []


#####Get the peaks
for i in range(0,len(hist)):
    flag = check_peak(i,hist,filter_size)
    if(flag):
        if(hist[i]>100):
          peaks.append(i)

####Get the valleys
for i in range(0,len(peaks)-1):
    start = peaks[i]
    end   = peaks[i+1]
    idx = return_min_idx(start,end,hist)
    valleys.append(idx)


print("Peaks ",peaks)
print("Valleys ",valleys)

save_img =  np.zeros((img_height,img_width))

for i in range(0,img_height):
    for j in range(0,img_width):
        save_img[i][j] = image[i][j]

# for i in range(0,img_width):
#     save_img[valleys[0]][i] = 100

for i in range(0,len(valleys)):
    end_nd = a_star(0, img_width, valleys[i], save_img,img_width-1,img_height-1)
    while(end_nd!=None):
        save_img[end_nd.cordinates.x][end_nd.cordinates.y] = 100
        end_nd = end_nd.parent
print("Done updating \n")
im = Image.fromarray(save_img)

if im.mode != 'RGB':
    im = im.convert('RGB')
im.save("Test3.jpg")
print("Done")


























































