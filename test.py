

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


def blocks_on_road(grid):
    lst = []
    for i in range(0,len(grid)):
        if grid[i] < 100:
            lst.append(i)
    return lst


grid_struct = [[0,0,255,0,255],
               [0,255,0,255,0],
               [255,255,0,0,255],
               [255,255,0,0,255],
               [255,255,0,255,255]]

valley = [0,1,2,3,4]






#end_nd = a_star(0,4,2,grid_struct)

'''
while (end_nd!=None):
    print("\n Coordinates are " ,end_nd.cordinates.x,end_nd.cordinates.y)
    end_nd = end_nd.parent
'''