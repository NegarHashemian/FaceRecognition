#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:00:36 2019

@author: nhashemian
"""

from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import inception_blocks_v2 as ib
import utils as us
from matplotlib import pyplot as plt



def face_verification(image_path, identity, database, model):
    
    
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding-database[identity])
    
    img0 = cv2.imread(image_path)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img0)
    ax[0].set_title('New Image')
    ax[0].axis('off')
    
    img1 = cv2.imread("images/Negar1.jpg")
    ax[1].imshow(img1)
    ax[1].set_title('Ref is '+identity)
    ax[1].axis('off')
    
    plt.show()
    
    
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist <0.7:
        print("The new images is " + str(identity) )

    else:
        print("The new images is not " + str(identity))       
        
    return dist



def face_recognition(image_path, database, model):
        
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path,model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist<min_dist:
            min_dist = dist
            identity = name

    img0 = cv2.imread(image_path)
    plt.imshow(img0)
    plt.axis('off')
    
    
    plt.show()
    
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("The image is " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity



def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    encoding = model.predict_on_batch(x_train)
    return encoding

# Main Code

FRmodel = ib.faceRecoModel(input_shape=(3, 93, 93))

us.load_weights_from_FaceNet(FRmodel)

database = {}
database["Negar"] = img_to_encoding("images/Negar1.jpg", FRmodel)
database["MySister"] = img_to_encoding("images/Nazanin.jpg", FRmodel)
print()
print('Verify if the new image is Negar:')
face_verification("images/Negar4.jpg", "Negar", database, FRmodel)
print('*********************************')
face_verification("images/Nazanin.jpg", "Negar", database, FRmodel)


print('*********************************')
print('Recognize who is in the picture:')

face_recognition("images/Negar4.jpg", database, FRmodel)
