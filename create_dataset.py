import os
keras = tf.keras
import random
import math
import numpy as np
import shutil
DATA = "monkbrill/"
subfolders = [ f.path for f in os.scandir(DATA) if f.is_dir() ]
count = -1
project_dir = "/Users/jindeshubham/Desktop/handwritten_recognition/"
per_class_count = 60


for subfolder in subfolders:
    folder_name = subfolder.split("/")[1]
    os.mkdir(project_dir+"training_new/" + folder_name)
    os.mkdir(project_dir+"testing_new/" + folder_name)
    subfolder = project_dir + subfolder
    count=count+1
    train_list=[]
    test_list =[]
    files = []
    for (dirpath, dirnames, filenames) in os.walk(subfolder):
        file_cnt = len(filenames)
        if(file_cnt<=per_class_count):
            reps = math.ceil(per_class_count/file_cnt)
        else:
            reps=1
            idx=random.sample(range(file_cnt), per_class_count)
            filenames = [filenames[id] for id in idx]
        class_count = 0
        for file in filenames:
            if(class_count==per_class_count):
                break
            if(class_count+reps>=per_class_count):
                reps = per_class_count - class_count
            if np.random.rand(1) < 0.8:
               for rep in range(0,reps):
                 file_jpg = file.split(".")[0]
                 file_jpg = file_jpg + "_" + str(rep) + ".jpg"
                 shutil.copy(subfolder + '/' + file,'training_new/' + folder_name + "/" + file_jpg)
                 train_list.append(str(count))
                 class_count = class_count + 1
            else:
               for rep in range(0, reps):
                 file_jpg = file.split(".")[0]
                 file_jpg = file_jpg + "_" + str(rep) + ".jpg"
                 shutil.copy(subfolder + '/' + file, 'testing_new/' + folder_name + "/" + file_jpg)
                 test_list.append(str(count))
                 class_count=class_count+1
        with open("training_new.txt", "a+") as f:
             f.write("\n")
             f.write('\n'.join(train_list))
        with open("testing_new.txt", "a+") as f:
             f.write("\n")
             f.write('\n'.join(test_list))

