from PIL import Image
import os
from os import listdir
from os.path import isfile, join

filepath = '/media/rcvkhu1/770a08ae-0a7d-4b62-bbf4-b81fdd9baf2a/'+'LA_train'+'/'+'LA_4'+'/'+'yt'+'/'
onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]


print(len(onlyfiles))

for i in range(len(onlyfiles)):
    id = i + 1
    im1 = Image.open(filepath+'/'+str(id)+'.png')
    im1.save(filepath+'/'+str(id)+'.jpg')
    os.remove(filepath+'/'+str(id)+'.png')