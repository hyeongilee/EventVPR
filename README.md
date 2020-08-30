# EventVPR
### aedat_to_numpy.ipynb
* convert .aedat file to numpy file(.npy)
* ms : time length for one numpy file (int)
* resize_n : resize factor for spatio resolution (int)
### aedat_to_text.ipynb
* convert .aedat file to text file(.txt)
* uses when apply e2vid.
* ms : time length for one numpy file (int)
* resize_n : resize factor for spatio resolution (int)
### classify.ipynb
* classify the utm coordinate using gps data
* ms : time length for one numpy file (int)
### classify_grid.ipynb
* classify the train-test grid# using gps data and database
* ms : time length for one numpy file (int)
* resolution : grid resolution (north, east)
### delta_x.ipynb
* update odometry info
### demo.ipynb
* train and test SNN
* parameters in network.yaml
### demo_grid.ipynb
* train and test SNN in grid map
* parameters in network_grid.yaml
### event_augmentation.ipynb
* augment event data with proposed algorithm
* a_f : augment ratio factor (augmented event#/real event#)
### hdf5_to_numpy.ipynb
* convert .hdf5 file to numpy file(.npy)
### make_aug_train_set_with_database.ipynb
* make augmented training set and testing set only with database
* ms : time length for one numpy file (int)
* threshold : upper distance limit for true data.
* threshold2 : lower distance limit for false data.
* aug : list for augment factor
### make_train_dataset.ipynb
* make training set and testing set with database and query
* ms : time length for one numpy file (int)
* threshold : upper distance limit for true data.
* threshold2 : lower distance limit for false data.
### make_train_dataset_grid.ipynb
* make training set and testing sest for grid map classification
* ms : time length for one numpy file (int)
* aug : list for augment factor
### make_train_set with_database
* make raw training set and testing set only with database
* ms : time length for one numpy file (int)
### visualize.ipynb
* visualize numpy event file
### utc_to_unix.ipynb
* convert gps utc time to unix time
