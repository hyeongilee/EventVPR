import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
# vis_file = np.load("/media/rcvkhu1/770a08ae-0a7d-4b62-bbf4-b81fdd9baf2a/train_dataset_100ms_aug_3/1059700595.npy")
vis_file = np.load("/media/rcvkhu1/770a08ae-0a7d-4b62-bbf4-b81fdd9baf2a/train_dataset_100ms_aug_3/1059700595.npy")

print(vis_file)
x = []
y = []
t = []
for i in range(vis_file.shape[0]):
    if(vis_file[i][2] == 1):
        x.append(vis_file[i][0])
        y.append(vis_file[i][1])
        t.append(vis_file[i][3])


ax.scatter(x, y, t, alpha=0.6, marker='.')
plt.show()