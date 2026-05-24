import os
import numpy as np
import operator
from distances import *

from tqdm import tqdm


def move_request_images(data_dir):
    request_images = ["0_1_BMW_X3_156", "0_0_BMW_Serie3Berline_45", "0_2_BMW_i8_300", "2_0_Volkswagen_Touareg_2839", 
                      "2_4_Volkswagen_Polo_3471", "2_9_Volkswagen_T-Roc_4233",
                      "4_2_Opel_vivarofourgon_5982", "4_4_Opel_Insignatourer_6351", "4_9_Opel_zafiralife_6850",
                      "6_0_Hyundai_Nexo_8305", "6_3_Hyundai_i10_8736", "6_5_Hyundai_i30_9029",
                      "8_1_Ford_Puma_11198", "8_5_Ford_Explorer_11897", "8_6_Ford_Focus_11936"]
    
    for img in request_images:
        img = img + ".jpg"
        if os.path.exists(os.path.join(data_dir, "dataset", img)):
            os.rename(os.path.join(data_dir, "dataset", img), os.path.join(data_dir, "request_images", img))

def getkVoisins(lfeatures, req, k, distanceName) : 
    ldistances = [] 
    print("Calculating distances...")
    for i in tqdm(range(len(lfeatures))): 
        dist = distance_f(req, lfeatures[i][1], distanceName)
        ldistances.append((lfeatures[i][0], lfeatures[i][1], dist)) 
    if distanceName in ["Correlation","Intersection"]:
        ordre=True
    else:
        ordre=False
    ldistances.sort(key=operator.itemgetter(2),reverse=ordre) 

    lvoisins = [] 
    for i in range(k): 
        lvoisins.append(ldistances[i]) 
    return lvoisins

def extract_class_id(image_name, brand_only = False):
    """
    Image name example:
    0_1_BMW_X3_156 => 1
    2_4_Volkswagen_Polo_3471 => 24
    8_6_Ford_Focus_11936 => 86
    """
    image_name = image_name.split("/")[-1].split("\\")[-1]
    image_name = image_name.split(".")[0]
    if brand_only:
        return int(image_name.split("_")[0])
    else:
        return int(image_name.split("_")[0] + image_name.split("_")[1])