import math
import time
import random
import cv2
import sys
import numpy as np
import os
import scipy
caffe_root = './caffe/'  # Change this to your caffe path.
sys.path.insert(0, caffe_root + 'python')
import caffe

target_imsize = (224, 224)
verbose = 1 # 0 or 1 or 2

# CNN
pycaffe_dir = caffe_root + 'python/'
center_only = True
image_dims = [224, 224]
model_def = 'deploy.prototxt'
pretrained_model = 'goselo_visible.caffemodel'
caffe.set_mode_gpu()
classifier = caffe.Classifier(model_def, pretrained_model, image_dims=image_dims, mean=None, input_scale=1.0, raw_scale=255.0, channel_swap=range(6))

# direction information
n_dirs = 8 # number of possible directions to move on the map
dx = [1, 1, 0, -1, -1, -1, 0, 1]
dy = [0, 1, 1, 1, 0, -1, -1, -1]
avoid_relative_dirs = np.array([1, 7, 2, 6, 3, 5, 4])
curr_dir = 0 # i.e., dx=1 and dy=0
    
# load map information
im_map = cv2.imread( sys.argv[ 1 ] )
im_map[ im_map > 100 ] = 255 # change gray to white
n = im_map.shape[ 1 ] # horizontal size of the map
m = im_map.shape[ 0 ] # vertical size of the map
the_map = []
the_map_pathlog = []
the_map_expand = []
row = [0] * n
row2 = [0] * n
row3 = [0] * n
for i in range(m): # create empty map
    the_map.append(list(row))
    the_map_pathlog.append(list(row2))
    the_map_expand.append(list(row3))

# set timeout
max_move_count = int(1e+6)

################################   
# set start and goal locations #
################################   
while True:
    xA = random.randint(0,n-1)
    yA = random.randint(0,m-1)
    xB = random.randint(0,n-1)
    yB = random.randint(0,m-1)
    if (im_map[ yA ][ xA ][ 0 ] > 100) and (im_map[ yB ][ xB ][ 0 ] > 100):
        break
start = (xA, yA)
goal  = (xB, yB)

if verbose == 1:
    print 'Map size (X,Y): ', n, m
    print 'Start: ', xA, yA
    print 'Finish: ', xB, yB

route_final = []
for move_count in range(max_move_count):

    # break if the agent is inside an obstacle
    if im_map[ yA ][ xA ][ 0 ] < 100:
        break

    #######################
    # record path history #
    #######################
    the_map_pathlog[ yA ][ xA ] += 1

    ##################
    # expand the map #
    ##################
    im_map_expand = cv2.erode( im_map, np.ones((2,2),np.uint8), 1 )

    ret, the_map = cv2.threshold( im_map, 100, 1, cv2.THRESH_BINARY_INV )
    the_map = the_map[:, :, 0] 
    ret, the_map_expand = cv2.threshold( im_map_expand, 100, 1, cv2.THRESH_BINARY_INV )
    the_map_expand = the_map_expand[:, :, 0] 

    ###############
    # convert img #
    ###############
    orig_map = np.zeros( (4, m, n), dtype=np.float )
    orig_map[0] = np.array(the_map) #/ 255.
    orig_map[1][yA][xA] = 1 #/ 255.
    orig_map[2][yB][xB] = 1 #/ 255.
    orig_map[3] = np.array(the_map_pathlog)
    orig_map = orig_map.transpose((1,2,0))

    sx = xA; sy = yA; gx = xB; gy = yB
    mx = (sx + gx) / 2.
    my = (sy + gy) / 2.
    dx_ = max( mx, orig_map.shape[ 1 ] - mx )
    dy_ = max( my, orig_map.shape[ 0 ] - my )
    im2 = np.zeros( (int(dy_ * 2), int(dx_ * 2), orig_map.shape[ 2 ]), dtype=np.float )
    im2[ int(dy_-my):int(dy_-my)+orig_map.shape[ 0 ] , int(dx_-mx):int(dx_-mx)+orig_map.shape[ 1 ] ] = orig_map
    im2[ im2 == 1 ] = 2
    if gx == sx:
        if gy > sy:
            theta = 90
        else:
            theta = 270
    else:
        theta = math.atan( float(gy-sy) / float(gx-sx) ) * 180 / math.pi
        if gx-sx < 0:
            theta = theta + 180
    im2 = scipy.ndimage.interpolation.rotate( im2, 90+theta )
    im2 = im2.transpose( (2,0,1) )
    L = int(np.sqrt( (gx-sx)*(gx-sx) + (gy-sy)*(gy-sy) ))
    im2 = [im2[0], im2[3]]
    im3 = np.zeros( (target_imsize[0], target_imsize[1], 6), dtype=np.uint8 )
    im3 = im3.transpose( (2,0,1) )
    l = (L+4, 4*L, 8*L)
    for n_ in range(2):
        for i in range(3):
            im2_ = np.zeros( (l[ i ], l[ i ]), dtype=np.uint8 )
            y1 = max( 0,  (im2[ n_ ].shape[0]-l[i])/2 )
            y2 = max( 0, -(im2[ n_ ].shape[0]-l[i])/2 )
            x1 = max( 0,  (im2[ n_ ].shape[1]-l[i])/2 )
            x2 = max( 0, -(im2[ n_ ].shape[1]-l[i])/2 )
            dy_ = min( l[i], im2[ n_ ].shape[ 0 ] )
            dx_ = min( l[i], im2[ n_ ].shape[ 1 ] )
            im2_[ y2:y2+dy_, x2:x2+dx_ ] = im2[ n_ ][ y1:y1+dy_, x1:x1+dx_ ]
            im3[ i + n_ * 3 ] = cv2.resize( im2_, im3[ i + n_ * 3 ].shape, interpolation = cv2.INTER_AREA )
            t = time.time()
    im3 = im3 * 0.5
    im3[(im3 > 0)*(im3 <= 1)] = 1

    caffe_input_map = im3.transpose( (1, 2, 0) )
    caffe_input_map = caffe_input_map / 255.

    ######################
    # CNN classification # 
    ######################
    t = time.time()
    predictions = classifier.predict([caffe_input_map], not center_only)
    dir_src = np.argmax( predictions )
    ang = 360 - 45 * dir_src - theta - 90
    while ang < 0:
        ang = ang + 360
    dir_dst = 8 - int( round( (ang % 360) / 45. ) )
    if dir_dst == 8:
        dir_dst = 0
    route = [dir_dst]
    if verbose == 2:
        print 'Time to generate the route (seconds): ', time.time() - t
        print 'Route (', move_count, '):', route

    #######################################################
    # force avoidance (change direction if it is invalid) #
    #######################################################
    free_space = n_dirs
    for d in range(n_dirs):
        free_space = free_space - the_map_expand[ yA+dy[ d ] ][ xA+dx[ d ] ]
    xA_ = xA + dx[ int(route[0]) ]
    yA_ = yA + dy[ int(route[0]) ]
    if (free_space > 0) and (the_map_expand[ yA+dy[ int(route[0]) ] ][ xA+dx[ int(route[0]) ] ] == 1) and ((abs(xA_-xB) > 3) or (abs(yA_-yB) > 3)):
        cand_dirs = avoid_relative_dirs + dir_dst
        for c in range(len(cand_dirs)):
            if cand_dirs[ c ] >= n_dirs:
                cand_dirs[ c ] -= n_dirs
        cand_dirs = [cd for cd in cand_dirs if not (the_map_expand[ yA+dy[ cd ] ][ xA+dx[ cd ] ] == 1) ]
        num_passed = [ the_map_pathlog[ yA+dy[ cd ] ][ xA+dx[ cd ] ] for cd in cand_dirs ]
        route = [ cand_dirs[ np.argmin( num_passed ) ] ]
        if verbose == 1:
            print 'Object Avoidance! Route (', move_count, '):', route

    ###############
    # display map #
    ###############
    im_map_vis = im_map.copy()
    cv2.circle( im_map_vis, (xA, yA), 5, (0, 0, 255), -1 )
    cv2.circle( im_map_vis, (xB, yB), 5, (255, 0, 0), -1 )
    cv2.imshow( 'map', im_map_vis )
    cv2.waitKey(3)

    ####################
    # one step forward #
    ####################
    xA += dx[ int(route[0]) ]
    yA += dy[ int(route[0]) ]
    curr_dir = int(route[0])
    route_final.append( int(route[0]) )

    # break if reached goal
    if (abs(xA-xB) < 2) & (abs(yA-yB) < 2):
        break


# mark the route on the map
route = route_final
x = start[ 0 ]
y = start[ 1 ]
cv2.circle( im_map_vis, (x, y), 2, (0, 0, 255), -1 )
for i in range(len(route)):
    j = int(route[i])
    x += dx[j]
    y += dy[j]
    cv2.circle( im_map_vis, (x, y), 2, (255, 0, 0), -1 )
cv2.circle( im_map_vis, (x, y), 2, (0, 0, 255), -1 )
cv2.imwrite('result.png',im_map_vis)
cv2.imshow( 'map', im_map_vis )
cv2.waitKey()
