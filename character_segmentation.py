from __future__ import division
from collections import defaultdict
import hashlib
import math
import os
import time
#from urllib2 import urlopen
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.sparse, scipy.spatial

t0 = time.clock()

diagnostics = True


class SWTScrubber(object):
    @classmethod
    def scrub(cls, line_img):
        """
        Apply Stroke-Width Transform to image.

        :param filepath: relative or absolute filepath to source image
        :return: numpy array representing result of transform
        """
        canny, sobelx, sobely, theta = cls._create_derivative(line_img)
        swt = cls._swt(theta, canny, sobelx, sobely)
        shapes = cls._connect_components(swt)
        swts, heights, widths, topleft_pts, images = cls._find_letters(swt, shapes)
        word_images = cls._find_words(swts, heights, widths, topleft_pts, images)

        if  word_images == None:
            return None, None
        final_mask = np.zeros(swt.shape)
        for word in word_images:
            final_mask += word
        return final_mask, 1

    @classmethod
    def _create_derivative(cls, line_img):
        # img = cv2.imread(filepath,0)
        line_img=np.uint8(line_img)
        edges = cv2.Canny(line_img, 175, 320, apertureSize=3)
        # Create gradient map using Sobel
        sobelx64f = cv2.Sobel(line_img,cv2.CV_64F,1,0,ksize=-1)
        sobely64f = cv2.Sobel(line_img,cv2.CV_64F,0,1,ksize=-1)

        theta = np.arctan2(sobely64f, sobelx64f)
        # if diagnostics:
        #     cv2.imwrite('edges.jpg',edges)
        #     cv2.imwrite('sobelx64f.jpg', np.absolute(sobelx64f))
        #     cv2.imwrite('sobely64f.jpg', np.absolute(sobely64f))
        #     # amplify theta for visual inspection
        #     theta_visible = (theta + np.pi)*255/(2*np.pi)
        #     cv2.imwrite('theta.jpg', theta_visible)
        return (edges, sobelx64f, sobely64f, theta)

    @classmethod
    def _swt(self, theta, edges, sobelx64f, sobely64f): #stroke width transform
        # create empty image, initialized to infinity
        swt = np.empty(theta.shape)
        swt[:] = np.Infinity
        rays = []

        print (time.clock() - t0)

        # now iterate over pixels in image, checking Canny to see if we're on an edge.
        # if we are, follow a normal a ray to either the next edge or image border
        # edgesSparse = scipy.sparse.coo_matrix(edges)
        step_x_g = -1 * sobelx64f
        step_y_g = -1 * sobely64f
        mag_g = np.sqrt( step_x_g * step_x_g + step_y_g * step_y_g )
        grad_x_g = step_x_g / mag_g
        grad_y_g = step_y_g / mag_g

        for x in range(edges.shape[1]):
            for y in range(edges.shape[0]):
                if edges[y, x] > 0:
                    step_x = step_x_g[y, x]
                    step_y = step_y_g[y, x]
                    mag = mag_g[y, x]
                    grad_x = grad_x_g[y, x]
                    grad_y = grad_y_g[y, x]
                    ray = []
                    ray.append((x, y))
                    prev_x, prev_y, i = x, y, 0
                    while True:
                        i += 1
                        cur_x = math.floor(x + grad_x * i)
                        cur_y = math.floor(y + grad_y * i)

                        if cur_x != prev_x or cur_y != prev_y:
                            # we have moved to the next pixel!
                            try:
                                if edges[cur_y, cur_x] > 0:
                                    # found edge,
                                    ray.append((cur_x, cur_y))
                                    theta_point = theta[y, x]
                                    alpha = theta[cur_y, cur_x]

                                    inv_cos = grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]
                                    if abs(inv_cos) > 1:
                                        continue

                                    if math.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                                        thickness = math.sqrt( (cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y) )
                                        for (rp_x, rp_y) in ray:
                                            swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                                        rays.append(ray)
                                    break
                                # this is positioned at end to ensure we don't add a point beyond image boundary
                                ray.append((cur_x, cur_y))
                            except IndexError:
                                # reached image boundary
                                break
                            prev_x = cur_x
                            prev_y = cur_y

        # Compute median SWT
        for ray in rays:
            median = np.median([swt[y, x] for (x, y) in ray])
            for (x, y) in ray:
                swt[y, x] = min(median, swt[y, x])
        if diagnostics:
            cv2.imwrite('swt.jpg', swt * 100)

        return swt

    @classmethod
    def _connect_components(cls, swt):
        # STEP: Compute distinct connected components
        # Implementation of disjoint-set
        class Label(object):
            def __init__(self, value):
                self.value = value
                self.parent = self
                self.rank = 0
            def __eq__(self, other):
                if type(other) is type(self):
                    return self.value == other.value
                else:
                    return False
            def __ne__(self, other):
                return not self.__eq__(other)

        ld = {}

        def MakeSet(x):
            try:
                return ld[x]
            except KeyError:
                item = Label(x)
                ld[x] = item
                return item

        def Find(item):
            # item = ld[x]
            if item.parent != item:
                item.parent = Find(item.parent)
            return item.parent

        def Union(x, y):
            """
            :param x:
            :param y:
            :return: root node of new union tree
            """
            x_root = Find(x)
            y_root = Find(y)
            if x_root == y_root:
                return x_root

            if x_root.rank < y_root.rank:
                x_root.parent = y_root
                return y_root
            elif x_root.rank > y_root.rank:
                y_root.parent = x_root
                return x_root
            else:
                y_root.parent = x_root
                x_root.rank += 1
                return x_root

        # apply Connected Component algorithm, comparing SWT values.
        # components with a SWT ratio less extreme than 1:3 are assumed to be
        # connected. Apply twice, once for each ray direction/orientation, to
        # allow for dark-on-light and light-on-dark texts
        trees = {}
        # Assumption: we'll never have more than 65535-1 unique components
        label_map = np.zeros(shape=swt.shape, dtype=np.uint16)
        next_label = 1
        # First Pass, raster scan-style
        swt_ratio_threshold = 3.0 ########################################################## parameter
        for y in range(swt.shape[0]):
            for x in range(swt.shape[1]):
                sw_point = swt[y, x]
                if sw_point < np.Infinity and sw_point > 0:
                    neighbors = [(y, x-1),   # west
                                 (y-1, x-1), # northwest
                                 (y-1, x),   # north
                                 (y-1, x+1)] # northeast
                    connected_neighbors = None
                    neighborvals = []

                    for neighbor in neighbors:
                        # west
                        try:
                            sw_n = swt[neighbor]
                            label_n = label_map[neighbor]
                        except IndexError:
                            continue
                        if label_n > 0 and sw_n / sw_point < swt_ratio_threshold and sw_point / sw_n < swt_ratio_threshold:
                            neighborvals.append(label_n)
                            if connected_neighbors:
                                connected_neighbors = Union(connected_neighbors, MakeSet(label_n))
                            else:
                                connected_neighbors = MakeSet(label_n)

                    if not connected_neighbors:
                        # We don't see any connections to North/West
                        trees[next_label] = (MakeSet(next_label))
                        label_map[y, x] = next_label
                        next_label += 1
                    else:
                        # We have at least one connection to North/West
                        label_map[y, x] = min(neighborvals)
                        # For each neighbor, make note that their respective connected_neighbors are connected
                        # for label in connected_neighbors. @todo: do I need to loop at all neighbor trees?
                        trees[connected_neighbors.value] = Union(trees[connected_neighbors.value], connected_neighbors)

        # Second pass. re-base all labeling with representative label for each connected tree
        layers = {}
        contours = defaultdict(list)
        for x in range(swt.shape[1]):
            for y in range(swt.shape[0]):
                if label_map[y, x] > 0:
                    item = ld[label_map[y, x]]
                    common_label = Find(item).value
                    label_map[y, x] = common_label
                    contours[common_label].append([x, y])
                    try:
                        layer = layers[common_label]
                    except KeyError:
                        layers[common_label] = np.zeros(shape=swt.shape, dtype=np.uint16)
                        layer = layers[common_label]

                    layer[y, x] = 1
        return layers

    @classmethod
    def _find_letters(cls, swt, shapes):
        # STEP: Discard shapes that are probably not letters
        swts = []
        heights = []
        widths = []
        topleft_pts = []
        images = []

        for label,layer in shapes.items():
            (nz_y, nz_x) = np.nonzero(layer)
            east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
            width, height = east - west, south - north

            if width < 8 or height < 8:
                continue

            if width / height > 10 or height / width > 10:
                continue

            diameter = math.sqrt(width * width + height * height)
            median_swt = np.median(swt[(nz_y, nz_x)])
            if diameter / median_swt > 10:
                continue

            if width / layer.shape[1] > 0.4 or height / layer.shape[0] > 0.4:
                continue

            #if diagnostics:
                # print (" written to image.")
                # cv2.imwrite('layer'+ str(label) +'.jpg', layer * 255)

            # we use log_base_2 so we can do linear distance comparison later using k-d tree
            # ie, if log2(x) - log2(y) > 1, we know that x > 2*y
            # Assumption: we've eliminated anything with median_swt == 1
            swts.append([math.log(median_swt, 2)])
            heights.append([math.log(height, 2)])
            topleft_pts.append(np.asarray([north, west]))
            widths.append(width)
            images.append(layer)

        return swts, heights, widths, topleft_pts, images

    @classmethod
    def _find_words(cls, swts, heights, widths, topleft_pts, images):
        # Find all shape pairs that have similar median stroke widths
        print ('SWTS')
        print (swts)
        print ('DONESWTS')
        swt_tree = scipy.spatial.KDTree(np.asarray(swts))
        stp = swt_tree.query_pairs(1)

        # Find all shape pairs that have similar heights
        height_tree = scipy.spatial.KDTree(np.asarray(heights))
        htp = height_tree.query_pairs(1)

        # Intersection of valid pairings
        isect = htp.intersection(stp)

        chains = []
        pairs = []
        pair_angles = []
        for pair in isect:
            left = pair[0]
            right = pair[1]
            widest = max(widths[left], widths[right])
            distance = np.linalg.norm(topleft_pts[left] - topleft_pts[right])
            if distance < widest * 3:
                delta_yx = topleft_pts[left] - topleft_pts[right]
                angle = np.arctan2(delta_yx[0], delta_yx[1])
                if angle < 0:
                    angle += np.pi

                pairs.append(pair)
                pair_angles.append(np.asarray([angle]))

        try:
            angle_tree = scipy.spatial.KDTree(np.asarray(pair_angles))

        except:
            return None
        atp = angle_tree.query_pairs(np.pi/12)

        for pair_idx in atp:
            pair_a = pairs[pair_idx[0]]
            pair_b = pairs[pair_idx[1]]
            left_a = pair_a[0]
            right_a = pair_a[1]
            left_b = pair_b[0]
            right_b = pair_b[1]

            # @todo - this is O(n^2) or similar, extremely naive. Use a search tree.
            added = False
            for chain in chains:
                if left_a in chain:
                    chain.add(right_a)
                    added = True
                elif right_a in chain:
                    chain.add(left_a)
                    added = True
            if not added:
                chains.append(set([left_a, right_a]))
            added = False
            for chain in chains:
                if left_b in chain:
                    chain.add(right_b)
                    added = True
                elif right_b in chain:
                    chain.add(left_b)
                    added = True
            if not added:
                chains.append(set([left_b, right_b]))

        word_images = []
        for chain in [c for c in chains if len(c) > 3]:
            for idx in chain:
                word_images.append(images[idx])
                # cv2.imwrite('keeper'+ str(idx) +'.jpg', images[idx] * 255)
                # final += images[idx]

        return word_images

def character_seg(line_img,cnt):

    # file_url = image_name
    #
    # orgnl_img = cv2.imread(file_url,0)

    rows, cols = line_img.shape
    org_count=0
    for i in range(0,cols):
        for j in range(0,rows):
            org_count = org_count + line_img[j][i]

    try:
        #s3_response = urlopen(file_url)
        #with open(local_filename, 'wb+') as destination:
        #    while True:
                # read file in 4kB chunks
        #        chunk = s3_response.read(4096)
        #        if not chunk: break
        #        destination.write(chunk)
        #final_mask = SWTScrubber.scrub('wallstreetsmd.jpeg')
        
        final_mask, flag = SWTScrubber.scrub(line_img)
        if flag == None:
            return
        # final_mask = cv2.GaussianBlur(final_mask, (1, 3), 0)
        # cv2.GaussianBlur(sobelx64f, (3, 3), 0)
        cv2.imwrite('final.jpg', final_mask * 255)
        print (time.clock() - t0)
    finally:
        #s3_response.close()
        print("Done")
    '''
    img = cv2.imread('swt.jpg',0)
    print(type(img))
    row, cols = img.shape
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite('closing.jpg', closing * 255)
    hist=[]
    for i in range(0,cols):
        count=0
        for j in range(0,row):
            if (closing[j][i] == 0):
                count = count + 1
        hist.append(count)
    plt.plot(hist)
    plt.savefig("test1.jpg")
    swt_count =0

    for i in range(0,cols):
        for j in range(0,rows):
            swt_count = swt_count + img[j][i]

    threshold = swt_count/(rows*cols)

    print("Threshold is ",threshold)
    ptential_valleys = []
    print(min(hist))
    for i in range(0,len(hist)):
        if(hist[i]<threshold/2):
            print(i)
    '''
    try:
        img = cv2.imread('swt.jpg')
        #img = cv2.bitwise_not(np.asarray(Image.open('/home/basti/Desktop/HWR/Sebastian/'+ folder + "/Line_" + str(cnt) + ".jpg")))
        mser = cv2.MSER_create()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converting to GrayScale
        gray_img = img.copy()
        hulls = []
        regions,_ = mser.detectRegions(gray)

        reg_cor = {"xmin":"","ymin":"","xmax":"","ymax":""}
        cordinates_lst = []
        stepper = 0
        rect_ls = []
        for p in regions:
            
             p = np.array(p)
             reg_cor["xmin"] = min(p[:,0])
             reg_cor["xmax"] = max(p[:,0])
             reg_cor["ymin"] = min(p[:,1])
             reg_cor["ymax"] = max(p[:,1])
             bl = (reg_cor["xmin"],reg_cor["ymin"]) # bl = bottom left
             tr = (reg_cor["xmax"],reg_cor["ymax"]) # tr = top right
             rect_ls.append((bl,tr))
             diff = np.subtract(tr, bl)
             if (diff[0]*diff[1]) > 1000: # if the rectangle is reasonable large enough (should we make this relative to reduce the amount of parameters we have to set?!)
                 ''' In case we want to exclude overlaping rectangles
                     def intersects(self, other):
                         return not (self.top_right.x < other.bottom_left.x or self.bottom_left.x > other.top_right.x or self.top_right.y < other.bottom_left.y or self.bottom_left.y > other.top_right.y)

                 '''
                 letter = gray_img[min(p[:,1]):max(p[:,1]),min(p[:,0]):max(p[:,0]),:]
                 
                 kernel = np.ones((2,2),np.uint8)
                 letter = cv2.morphologyEx(letter, cv2.MORPH_CLOSE, kernel)
                 
                 cv2.imwrite('/home/basti/Desktop/HWR/Sebastian/letters/'+ str(stepper)+'.jpg', letter) ### with both this and the next line enabled we'll have the green lines in our letters
                
             #cv2.rectangle(gray_img, (min(p[:,0]), min(p[:,1])), (max(p[:,0]),  max(p[:,1])), (0, 255, 0), 1)
             #cordinates_lst.append(reg_cor)
             stepper += 1

        
        #cv2.imwrite('/Users/jindeshubham/Desktop/handwritten_recognition/mser'+str(cnt)+'.jpg', gray_img) #Saving
        cv2.imwrite('/home/basti/Desktop/HWR/Sebastian/'+ str(cnt)+'.jpg', gray_img) #Saving
    except:
        print("FAILED TO FIND")



#### Read in Input ####

from PIL import Image
cnt = 1
folder = "P123-Fg002-R-C01-R01-binarized"
image = np.asarray(Image.open('/home/basti/Desktop/HWR/Sebastian/'+ folder + "/Line_" + str(cnt) + ".jpg").convert('L'))
img = image.copy()

### Preprocessing ###
preprocessing = 0
if preprocessing == 1:
    
    # First invert colors
    
    img = cv2.bitwise_not(img)
    
    ## Erosion
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    
    ## Dilation
    #dilation = cv2.dilate(img,kernel,iterations = 1)
    
    ## Opening 
    #opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    ## Closing
    #closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
    img = cv2.bitwise_not(erosion)
    
#### MAIN ####

character_seg(img, cnt)

