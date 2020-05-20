
import matplotlib.pyplot as plt
import statistics
import numpy as np
from PIL import Image
# class Point:
#     def  __init__(self, x, y):
#         self.x = x
#         self.y = y
#
# class Node:
#     def __init__(self, point, parent, f_score, g_score, h_score):
#         self.cordinates = point
#         self.f_score = f_score
#         self.g_score = g_score
#         self.h_score = h_score
#         self.parent  = parent
#         self.visited = True
#
#
# node_lst =[]
# pt = Point(0,0)
# a = Node(pt,None,0,0,0)
# node_lst.append(a)
# for i in range(1,3):
#     pt = Point(i,i)
#     b = Node(pt,a,i,i,i)
#     a = b
#     node_lst.append(a)
#
# print("here")
# elemnt = node_lst[-1]
#
# while (elemnt !=None):
#     print("X cordinate, y cordinate, fscore\n",elemnt.cordinates.x, elemnt.cordinates.y, elemnt.f_score)
#     elemnt = elemnt.parent


# def blocks_on_road(grid):
#     lst = []
#     for i in range(0,len(grid)):
#         if grid[i] < 100:
#             lst.append(i)
#     return lst
#
#
# grid_struct = [[0,0,255,0,255],
#                [0,255,0,255,0],
#                [255,255,0,0,255],
#                [255,255,0,0,255],
#                [255,255,0,255,255]]
#
# valley = [0,1,2,3,4]
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


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
    plt.savefig("Vertical_projection.jpg")
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

    gap_lnt = 0
    word_cut = []
    for i in range(0, sent_wid):
        if (vertical_projection[i] == sent_hgt):
            gap_lnt = gap_lnt + 1
        else:
            if (gap_lnt != 0):
                if (gap_lnt >= average_length):
                    word_cut.append(i)
            gap_lnt = 0
    for i in range(0, len(word_cut)):
        sentence_img[0:sent_hgt, word_cut[i]] = 100
    return sentence_img


image = np.asarray(Image.open('6.jpg'))
image = rgb2gray(image)


print(type(image))
img = segment_words(image)

img =  Image.fromarray(img)
if img != 'RGB':
    img = img.convert('RGB')
img.save("66.jpg")
#end_nd = a_star(0,4,2,grid_struct)

'''
while (end_nd!=None):
    print("\n Coordinates are " ,end_nd.cordinates.x,end_nd.cordinates.y)
    end_nd = end_nd.parent
'''