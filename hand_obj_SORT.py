
from sort import *
import pickle
import numpy as np
import os

#create instance of SORT
mot_tracker = Sort() 

# get detections
detections_path = "D:\\GithubLocal\\sort\\6745e6c6-27d3-433e-85f9-ecebbc991868_A00__img_hand_obj_info.pickle"

with open(detections_path, 'rb') as f:
    detection_data = pickle.load(f)
img_path = sorted(list(detection_data.keys())) 
#print (img_path)

hand_obj_score = []


for i in range (0,len(img_path),1):
    
    current_frame_detections = []
    
    if not np.any(detection_data[img_path[i]]["hand_dets"]) and not np.any(detection_data[img_path[i]]["obj_dets"]): # no hand or object found
        hand_obj_score.append([[0,0,0,0,0]])
        
    else:
        
        if np.any(detection_data[img_path[i]]["hand_dets"]): # at least one hand got detected
            hand_detections = []       
            hand_detections = detection_data[img_path[i]]["hand_dets"]
            
            for itr in range (0,len(hand_detections),1):
                current_frame_detections.append(hand_detections[itr][:5])
                
        if np.any(detection_data[img_path[i]]["obj_dets"]): # at least one obj got detected
            obj_detections = []       
            obj_detections = detection_data[img_path[i]]["obj_dets"]
            for itr in range (0,len(obj_detections),1):
                current_frame_detections.append(obj_detections[itr][:5])
        
        hand_obj_score.append(current_frame_detections)

    
#     print("frame Number: ", i)
#     print("main: ", img_path[i], detection_data[img_path[i]])
#     print("extracted: ", hand_obj_score[i])
# print(hand_obj_score[len(img_path)-1])

img_folder_path = "E:\\temp\\6745e6c6-27d3-433e-85f9-ecebbc991868_A00\\"

colours = np.random.rand(32, 3)
display = True

if(display):
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

for i in range(0,len(hand_obj_score),1):
    
    hos = hand_obj_score[i]
    np_hos = np.array(hos) # converting to np array from list
    track_bbs_ids = mot_tracker.update(np_hos)
    
    
    head_tail = os.path.split(img_path[i])
    new_img_path = os.path.join(img_folder_path,head_tail[1])

    if(display):
        im = io.imread(new_img_path)
        ax1.imshow(im)
        plt.title(head_tail[1] + ' Tracked Targets')
    
    for d in track_bbs_ids:
         #print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))
         if(display):
             d = d.astype(np.int32)
             ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

    if(display):
        fig.canvas.flush_events()
        plt.draw()
        ax1.cla()
    
    
    print(new_img_path, track_bbs_ids)
    
# print("end")    
# update SORT
#track_bbs_ids = mot_tracker.update(detections)