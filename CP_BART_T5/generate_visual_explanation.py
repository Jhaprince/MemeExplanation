import pandas as pd
import numpy as np
from PIL import Image
import os

visual_coord = pd.read_excel("/home/krishanu_2021cs19/ACL2023/VLCybebully/Meme_Explanability_FINAL.xlsx","Boxes")


dictionary = {}

rewidth = 224
reheight = 224

for i in range(len(visual_coord)):
    image_name = visual_coord["image_name"][i]
    dictionary[image_name] = []
    


for i in range(len(visual_coord)):
    image_name = visual_coord["image_name"][i]
    image_width = visual_coord["image_width"][i]
    image_height = visual_coord["image_height"][i]
    bbox_x = visual_coord["bbox_x"][i]
    bbox_y = visual_coord["bbox_y"][i]
    bbox_width = visual_coord["bbox_width"][i]
    bbox_height = visual_coord["bbox_height"][i]
    
    bbox_feature = []
    
    x_new = int(bbox_x*rewidth/image_width)
    y_new = int(bbox_y*reheight/image_height)
    w_new = int(bbox_width*rewidth/image_width)
    h_new = int(bbox_height*reheight/image_height)
    
    bbox_feature.append(x_new)
    bbox_feature.append(y_new)
    bbox_feature.append(w_new)
    bbox_feature.append(h_new)
    
    dictionary[image_name].append(bbox_feature)
    
target_array = "/home/krishanu_2021cs19/ACL2023/CMEx/VisualExplanations"
target_segment = "/home/krishanu_2021cs19/ACL2023/CMEx/VisualSegment"
    
for item in dictionary.keys():
    image_explanation = np.zeros((rewidth,reheight))
    
    for i in range(len(dictionary[item])):
        x = dictionary[item][i][0]
        y = dictionary[item][i][1]
        w = dictionary[item][i][2]
        h = dictionary[item][i][3]
        
        for j in range(x,x+w):
            for k in range(y,y+h):
                image_explanation[j][k]=1
    
    
    
    image_explanation_t = np.transpose(image_explanation)
    target_vx = os.path.join(target_array,item+".npy")
    
    np.save(target_vx,image_explanation_t)
    
    image_explanation = np.zeros((rewidth,reheight),dtype=np.uint8)
    
    for i in range(len(dictionary[item])):
        x = dictionary[item][i][0]
        y = dictionary[item][i][1]
        w = dictionary[item][i][2]
        h = dictionary[item][i][3]
        
        for j in range(x,x+w):
            for k in range(y,y+h):
                image_explanation[j][k]=225
    
    
    
    image_explanation_t = np.transpose(image_explanation)
    
    img = Image.fromarray(image_explanation_t)
    img.save(os.path.join(target_segment,item))
    
    
    
    
    