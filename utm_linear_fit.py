from dv import AedatFile
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os


env = 'rain'
date = '2020_07_30_17_51_53'
t_step = 250 #ms
utm_raw = open('../dataset/KH/'+env+'/utm.txt')
utm = list()

while True:
    line = utm_raw.readline()
    if not line:
        break
    line_sp = line.split()
    utm.append([int(line_sp[0]), float(line_sp[1]), float(line_sp[2])])

utm_raw.close()

aedat_file = "../dataset/KH/"+env+"/raw_aedat/"+date+".aedat4"
with AedatFile(aedat_file) as f:
    events = np.hstack([packet for packet in f['events'].numpy()])
    _t_, x, y, p = events['timestamp'], events['x'], events['y'], events['polarity']

slice_n = 1000//t_step
ti = _t_[0]
it = 0

while utm[it][0] < ti:
    it += 1

left = utm[it-1][0]
right = utm[it][0]
print(ti - left)
bin_location = (ti-left) // (t_step*1000)
print(bin_location)

i = it-1
f = it

count = 0
p = "/media/rcvkhu1/770a08ae-0a7d-4b62-bbf4-b81fdd9baf2a/proj_beta/"+env+"/"+date+"/proj_image_1000ms_"+str(t_step)+"ms/"
d = "/media/rcvkhu1/770a08ae-0a7d-4b62-bbf4-b81fdd9baf2a/proj_beta/"+env+"/"+date+"/proj_image_1000ms_"+str(t_step)+"ms/xt"
for path in os.listdir(d):
    if os.path.isfile(os.path.join(d, path)):
        count += 1
print(count)

ground_output = open(p+"utm.txt",'w')

n = 1
while n <= count:
    x_i = utm[i][1]
    y_i = utm[i][2]
    x_f = utm[f][1]
    y_f = utm[f][2]
    x = x_i + bin_location*((x_f-x_i)/slice_n)
    y = y_i + bin_location*((y_f-y_i)/slice_n)
    bin_location = bin_location+1
    if(bin_location == slice_n):
        bin_location = 0
        f = f + 1
        i = i + 1
    ground_output.write(str(n) + '\t' + str(x) + '\t' + str(y) + '\n')
    n = n + 1

ground_output.close()


