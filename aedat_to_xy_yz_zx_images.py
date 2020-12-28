from dv import AedatFile
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt



env = "rain"
date = "2020_07_30_17_51_53"
aedat_file = "../dataset/KH/"+env+"/raw_aedat/"+date+".aedat4"
with AedatFile(aedat_file) as f:
    events = np.hstack([packet for packet in f['events'].numpy()])
    _t_, x, y, p = events['timestamp'], events['x'], events['y'], events['polarity']

x_resolution = 320
y_resolution = 240
t_resolution = 1000 #resize to 250
t_bin = 10
t_step = 250
xy_bin = 50
ti = _t_[0]
print(ti)
t = (_t_-ti)//1000
N = len(t)
print(t[6626], t[6627])
print(len(t), len(x), len(y))

# ground_read = open('~/hyeongilee/slayerPytorch/example/dataset/KH/'+env+'/utm.txt','r')
# ground_file = open('../dataset/KH/'+env+'/proj_image_'+str(t_resolution)+'ms/ground.txt','w')


def binary_search(target, lst):
    length = len(lst)
    left = 0 
    right = length-1

    while left<=right:
        mid = (left+right)//2
        if lst[mid] == target:
            while(mid != -1 and lst[mid] == target):
                mid = mid-1
            return mid + 1
        elif lst[mid]>target:
            right = mid-1
        elif lst[mid]<target:
            left = mid+1
    return -1

def numpy_to_image(np_array, typ, idx):
    # im = Image.fromarray(A)
    # im.save("your_file.jpeg")
    if typ == 'xy':
        sm = np.sum(np_array)
        nz = np.count_nonzero(np_array)
        avg = sm/float(nz)

        mx = 2*avg
    else:
        mx = np.max(np_array)
    mn = np.min(np_array)
    # print(mx,mn)
    for i in range(np_array.shape[0]):
        for j in range(np_array.shape[1]):
            np_array[i][j] = min(int((np_array[i][j]-mn)*(255.0/(mx-mn))), 255)
    # plt.imsave('/media/rcvkhu1/770a08ae-0a7d-4b62-bbf4-b81fdd9baf2a/'+env+'/'+date+'/proj_image_'+str(t_resolution)+'ms_'+str(t_bin)+'ms/dis_pol'+str(idx)+'.png', np_array)
    # img = Image.fromarray(np_array)
    # img.save('../dataset/KH/'+env+'/proj_image_'+str(t_resolution)+'ms/'+use_pol+'/'+typ+'/'+str(idx)+'.png')
    cv2.imwrite('/media/rcvkhu1/770a08ae-0a7d-4b62-bbf4-b81fdd9baf2a/proj_beta/'+env+'/'+date+'/proj_image_'+str(t_resolution)+'ms_'+str(t_step)+'ms/'+typ+'/'+str(idx)+'.png', np_array)


cnt = 0
noise_set = set()

for i in range(0, t[-1]-t_resolution+1, t_step):
    j = binary_search(i, t)
    k = binary_search(i+t_resolution, t)-1
    if (j == -1 or k == -2):
        continue
    img_xt = np.zeros((x_resolution, t_resolution//t_bin))
    img_yt = np.zeros((y_resolution, t_resolution//t_bin))
    img_xy = np.zeros((y_resolution, x_resolution))
    for z in range(j, k+1):
        _x = x[z]    #0~319
        _y = y[z]    #0~239
        _t = (t[z]-t[j])//t_bin    #0~249
        
        # if((_x == 201 and _y == 45) or (_x == 306 and _y == 0) or (_x == 163 and _y == 11)):
        #     continue
        if((_y,_x) in noise_set):
            continue
        img_xt[_x][_t] += 1
        img_yt[_y][_t] += 1
        if(t[z]-t[j] < xy_bin):
            img_xy[_y][_x] += 1

    if(cnt == 0):
        for i in range(y_resolution):
            for j in range(x_resolution):
                if(img_xy[i][j] > (0.01*t_resolution)):
                    noise_set.add((i,j))
                    img_xy[i][j] = 0

    # # print(len(noise_set))
    # # print(noise_set)
    # for (i,j) in noise_set:
    #     img_xy[i][j] = 0
    cnt += 1
    numpy_to_image(img_xt, 'xt', cnt)
    numpy_to_image(img_yt, 'yt', cnt)
    numpy_to_image(img_xy, 'xy', cnt)
    
    



    
# ground_read.close()


# ground_file.close()

