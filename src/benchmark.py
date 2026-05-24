import os
import time
import torch
import timm
import torchvision.models as models
import pandas as pd
from metrics import *   
from torchvision import transforms

from feature_extractor import *
from utils import *
from retrieval import *

def evaluate_retrieval(feature_extractor, data_dir, output_dir, distanceName="Euclidienne", df_global=pd.DataFrame(columns=["Feature_Extractor", "Distance", "k", "brand_only", "MAP"]), df_requests=pd.DataFrame(columns=["Feature_Extractor", "Request", "k", "brand_only", "Recall@k", "R-precision", "Average precision@k"]), plot_curves=False):
    """ Evaluate performance of a given generic feature extractor on all image requests
    
    Args:
        feature_extractor: FeatureExtractor object
        data_dir: str, path to the data directory
        output_dir: str, path to the output directory containing the extracted features
        distanceName: str, distance name
        brand_only: bool, whether to use only brand as class
        k: int, number of retrieved items
    """
    print("Evaluating feature extractor: ", feature_extractor.name)
    dataset_dir = data_dir + "/dataset"
    files = os.listdir(dataset_dir)
    files = [os.path.join(dataset_dir, file) for file in files]
    feature_extractor.index_database(files, output_dir)
    request_images = os.listdir(data_dir + "/request_images")
    lfeatures = feature_extractor.load_features(output_dir)
    
    retrieved_classes_list_all, relevant_class_list_all = [], []
    retrieved_classes_list_brand_only, relevant_class_list_brand_only = [], []
    for img_idx in range(len(request_images)):
        request_image = os.path.join(data_dir, "request_images", request_images[img_idx])
        print(f"\n🔍 Requête {img_idx+1} : {request_image}")
        print(f"  ➤ Classe modèle : {extract_class_id(request_image)}")
        request_feature = feature_extractor.extract_feature(request_image)
        
        #Initial request class
        request_class = extract_class_id(request_image)
        relevant_items = len([file for file in os.listdir(dataset_dir) if extract_class_id(file, brand_only=False) == request_class])
        
        #Distance computations
        lvoisins = getkVoisins(lfeatures, request_feature, max(100, relevant_items), distanceName)
        print(f"  ➤ Top 5 voisins (brand+model) : {[extract_class_id(v[0]) for v in lvoisins[:5]]}")
        
        #Evaluation metrics brand+model
        request_class = extract_class_id(request_image, brand_only=False)
        retrieved_classes = [extract_class_id(lvoisin[0], brand_only=False) for lvoisin in lvoisins]
        print(f"  ➤ retrieved_classes (brand+model) : {retrieved_classes[:10]}")
        retrieved_classes_list_all.append(retrieved_classes)
        relevant_class_list_all.append([request_class])
        for k in [50, 100, "max"]:
            if k == "max":
                k = relevant_items
            r = recall(k, retrieved_classes, [request_class], relevant_items)
            r_p = r_precision(k, retrieved_classes, [request_class])
            print(f"    - R@50={r:.3f}, P@50={r_p:.3f}, AP@50={ap:.3f}")
            ap = average_precision(k, retrieved_classes, [request_class])
            df_requests = pd.concat([df_requests, pd.DataFrame({"Feature_Extractor": [feature_extractor.name], "Request": [request_images[img_idx]], "k": [k], "brand_only": [False], "Recall@k": [r], "R-precision": [r_p], "Average precision@k": [ap]})])

        if plot_curves:
            fig = plot_precision_recall_curve(relevant_items, retrieved_classes, [request_class], relevant_items)
            fig.savefig(f"./results/pr_curves/{feature_extractor.name}_{request_images[img_idx]}_brand_model.pdf")

        #Evaluation metrics brand only
        request_class = extract_class_id(request_image, brand_only=True)
        retrieved_classes = [extract_class_id(lvoisin[0], brand_only=True) for lvoisin in lvoisins]
        retrieved_classes_list_brand_only.append(retrieved_classes)
        relevant_class_list_brand_only.append([request_class])
        for k in [50, 100, "max"]:
            if k == "max":
                k = relevant_items
            r = recall(k, retrieved_classes, [request_class], relevant_items)
            r_p = r_precision(k, retrieved_classes, [request_class])
            ap = average_precision(k, retrieved_classes, [request_class])
            df_requests = pd.concat([df_requests, pd.DataFrame({"Feature_Extractor": [feature_extractor.name], "Request": [request_images[img_idx]], "k": [k], "brand_only": [True], "Recall@k": [r], "R-precision": [r_p], "Average precision@k": [ap]})])

        if plot_curves:
            fig = plot_precision_recall_curve(relevant_items, retrieved_classes, [request_class], relevant_items)
            fig.savefig(f"./results/pr_curves/{feature_extractor.name}_{request_images[img_idx]}_brand.pdf")

    #Evaluate map for k=50 and k=100, brand+model and brand only
    for k in [50, 100]:
        map_all = mean_average_precision(k, retrieved_classes_list_all, relevant_class_list_all)
        map_brand_only = mean_average_precision(k, retrieved_classes_list_brand_only, relevant_class_list_brand_only)
        df_global = pd.concat([df_global, pd.DataFrame({"Feature_Extractor": [feature_extractor.name], "Distance": [distanceName], "k": [k], "brand_only": [False], "MAP": [map_all]})])
        df_global = pd.concat([df_global, pd.DataFrame({"Feature_Extractor": [feature_extractor.name], "Distance": [distanceName], "k": [k], "brand_only": [True], "MAP": [map_brand_only]})])
    
    return df_global, df_requests


def time_evaluation(feature_extractor, data_dir, output_dir, distanceName, df_global=pd.DataFrame(columns=["Feature_Extractor", "Index_time", "Research_time_mean", "Research_time_std"])):
    dataset_dir = data_dir + "/dataset"
    files = os.listdir(dataset_dir)
    files = [os.path.join(dataset_dir, file) for file in files]
    start = time.time()
    feature_extractor.index_database(files, output_dir, save_features=False)
    end = time.time()
    index_time = end - start
    index_time = index_time 
    request_images = os.listdir(data_dir + "/request_images")
    lfeatures = feature_extractor.load_features(output_dir)
    retrieval_times = []
    for img_idx in range(len(request_images)):
        request_image = os.path.join(data_dir, "request_images", request_images[img_idx])
        request_feature = feature_extractor.extract_feature(request_image)
        start = time.time()
        lvoisins = getkVoisins(lfeatures, request_feature, 100, distanceName)
        end = time.time()
        retrieval_time = end - start
        retrieval_time = retrieval_time 
        retrieval_times.append(retrieval_time)
    
    df_global = pd.concat([df_global, pd.DataFrame({"Feature_Extractor": [feature_extractor.name], "Index_time": [index_time], "Research_time_mean": [np.mean(retrieval_times)], "Research_time_std": [np.std(retrieval_times)]})])

    return df_global

def study_class_imbalance(data_dir):
    dataset_dir = data_dir + "/dataset"
    files = os.listdir(dataset_dir)
    classes = [extract_class_id(file) for file in files]
    classes = pd.Series(classes)
    classes.value_counts().to_csv("results/brand_model_imbalance.csv")  

    #Plot histogram of classes with enough room between bars
    #Augmenter la police
    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(20, 10))
    plt.bar(classes.value_counts().index, classes.value_counts().values, width=0.8)
    plt.xlabel("Classes")
    plt.ylabel("Nombres d'échantillons")
    plt.savefig("results/brand_model_imbalance.pdf")


    #print number of classes
    print("Number of classes: ", classes.nunique())

    classes = [extract_class_id(file, brand_only=True) for file in files]
    classes = pd.Series(classes)
    classes.value_counts().to_csv("results/brand_imbalance.csv")

    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(20, 10))
    plt.bar(classes.value_counts().index, classes.value_counts().values, width=0.8)
    #Ici montrer tous les noms des classes
    plt.xticks(classes.value_counts().index)

    plt.xlabel("Classes")
    plt.ylabel("Nombres d'échantillons")
    plt.savefig("results/brand_imbalance.pdf")

    print("Number of classes (brand only): ", classes.nunique())


if __name__ == "__main__":
    data_dir = "../data"
    output_dir = "../extracted_features"
    
    move_request_images(data_dir)

    """benchmark = { 
        Histogram_Color_Extractor(): "Euclidienne",
        Histogram_HSV_Extractor(): "Euclidienne", 
        SIFT_Extractor(): "Flann",
        ORB_Extractor(): "Flann", 
        LBP_Extractor(): "Euclidienne",
        GLCM_Extractor(): "Euclidienne",
        HOG_Extractor(): "Euclidienne", 
        ResNet34_ImageNet_Extractor(): "Euclidienne", 
        ResNet34_2_steps_data_augmentation_Steps_Extractor(): "Euclidienne", 
        InceptionV4_2_steps_data_augmentation_Extractor(): "Euclidienne"
    }"""

    benchmark = { 
        ResNet34_ImageNet_Extractor(): "Euclidienne"
    }  
    
    #Performance evaluation
    df_global = pd.DataFrame(columns=["Feature_Extractor", "Distance", "k", "brand_only", "MAP"])
    df_requests = pd.DataFrame(columns=["Feature_Extractor", "Request", "k", "brand_only", "Recall@k", "R-precision", "Average precision@k"])
    for feature_extractor, distance in benchmark.items():
        df_global, df_requests = evaluate_retrieval(feature_extractor, data_dir, output_dir, distance, df_global, df_requests)
    #Dump results to csv
    df_global.to_csv("./results/raw_results_global_2.csv", index=False)
    df_requests.to_csv("./results/raw_requests_results_2.csv", index=False)

    """#Time evaluation
    df_time = pd.DataFrame(columns=["Feature_Extractor", "Index_time", "Research_time_mean", "Research_time_std"])
    for feature_extractor, distance in benchmark.items():
        df_time = time_evaluation(feature_extractor, data_dir, output_dir, distance, df_time)

    df_time.to_csv("./results/time_results.csv", index=False)"""



    """ #Pr_curves computations
    benchmark = {
        InceptionV4_2_steps_data_augmentation_Extractor(): "Euclidienne"
    }
    for feature_extractor, distance in benchmark.items():
        evaluate_retrieval(feature_extractor, data_dir, output_dir, distance, plot_curves=True)"""
    

    study_class_imbalance(data_dir)
