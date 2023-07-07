import json
import numpy as np
import cv2
import os
# read json file
path1 = "C:/Users/Avon/Desktop/label/"
path2 = "C:/Users/Avon/Desktop/jiangmask/"
path3 = "C:/Users/Avon/Desktop/jiangimg/"
files = os.listdir(path1)
for file in files:
    with open(path1+file, "r") as f:
        data = f.read()
        data = json.loads(data)
        points = data["shapes"][0]["points"]
        points = np.array(points, dtype=np.int32)  # tips: points location must be int32
        name, _ = file.split('.')
        image = cv2.imread(path3+name+'.png')
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        cv2.imwrite(path2+name+".png", mask)
