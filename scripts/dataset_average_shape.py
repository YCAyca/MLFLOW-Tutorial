import os
import cv2

root_dir = "dataset_few" 

photo_folders = os.listdir(root_dir)

height = 0
width = 0
count = 0

for folder in photo_folders:
    images = os.listdir(os.path.join(root_dir, folder))
    for img_name in images:
        img = cv2.imread(os.path.join(root_dir, folder, img_name)) #BGR
        h,w,c = img.shape # h,w,c
        height += h
        width += w 
        count += 1


print("average height:", height/count)
print("average width:", width/count)
