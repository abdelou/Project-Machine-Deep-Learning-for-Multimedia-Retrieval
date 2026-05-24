import os
import torch
import timm
import torchvision.models as models
import pandas as pd

from torchvision import transforms

from feature_extractor import *
from utils import *

#This file is used to generate the latex tables from the results

def requests_df_to_latex(df, filename):
    df = df.round(3)
    header = "\\begin{table}[H]\n"
    header += "\\centering\n"
    header += "\\begin{tabular}{l|rrr|rrr|rrr|rr}\n"
    header += "\\toprule\n"
    header += "\\textbf{Requête} & \\multicolumn{3}{c}{\\textbf{Rappel}} & \\multicolumn{3}{c}{\\textbf{Précision}} & \\multicolumn{3}{c}{\\textbf{AP}} & \\multicolumn{2}{c}{\\textbf{mAP}} \\\\\n"
    header += " & \\textbf{[50]} & \\textbf{[100]} & \\textbf{[max]} & \\textbf{[50]} & \\textbf{[100]} & \\textbf{[max]} & \\textbf{[50]} & \\textbf{[100]} & \\textbf{[max]} & \\textbf{[50]} & \\textbf{[100]} \\\\\n"
    header += "\\midrule\n"

    body = ""
    for i in range(len(df)):
        row = df.iloc[i]
        if i == 0:
            body += row["Request"] + " & " + str(row["Recall@50"]) + " & " + str(row["Recall@100"]) + " & " + str(row["Recall@max"]) + " & " + str(row["Precision@50"]) + " & " + str(row["Precision@100"]) + " & " + str(row["Precision@max"]) + " & " + str(row["AP@50"]) + " & " + str(row["AP@100"]) + " & " + str(row["AP@max"]) + " & \\multirow{" + str(len(df)) + "}{*}{" + str(row["mAP@50"]) + "} & \\multirow{" + str(len(df)) + "}{*}{" + str(row["mAP@100"]) + "} \\\\\n"
        else:
            body += row["Request"] + " & " + str(row["Recall@50"]) + " & " + str(row["Recall@100"]) + " & " + str(row["Recall@max"]) + " & " + str(row["Precision@50"]) + " & " + str(row["Precision@100"]) + " & " + str(row["Precision@max"]) + " & " + str(row["AP@50"]) + " & " + str(row["AP@100"]) + " & " + str(row["AP@max"]) + " \\\\\n"

    footer = "\\bottomrule\n"
    footer += "\\end{tabular}\n"
    footer += "\\caption{Mesures de précision du moteur recherche}\n"
    footer += "\\label{tab:results}\n"
    footer += "\\end{table}\n"

    with open(filename, "w") as f:
        f.write(header + body + footer)
    
def global_df_to_latex(df, filename):
    df = df.round(3)
    df = df.sort_values(by="mAP@100_brand_model", ascending=False)
    header = "\\begin{table}[H]\n"
    header += "\\centering\n"
    header += "\\begin{tabular}{l|rr|rr}\n"
    header += "\\toprule\n"

    header += "\\textbf{Descripteur} & \\multicolumn{2}{c}{\\textbf{mAP@50}} & \\multicolumn{2}{c}{\\textbf{mAP@100}} \\\\\n"
    header += " & \\textbf{Marque} & \\textbf{Marque + Modèle} & \\textbf{Marque} & \\textbf{Marque + Modèle} \\\\\n"
    header += "\\midrule\n"

    body = ""
    for i in range(len(df)):
        row = df.iloc[i]
        body += row["Feature_Extractor"].replace("_", "\_") + " & " + str(row["mAP@50_brand"]) + " & " + str(row["mAP@50_brand_model"]) + " & " + str(row["mAP@100_brand"]) + " & " + str(row["mAP@100_brand_model"]) + " \\\\\n"

    footer = "\\bottomrule\n"
    footer += "\\end{tabular}\n"
    footer += "\\caption{Benchmark des descripteurs}\n"
    footer += "\\label{tab:results}\n"
    footer += "\\end{table}\n"

    with open(filename, "w") as f:
        f.write(header + body + footer)


if __name__ == "__main__":
    df_global = pd.read_csv("./results/raw_results_global_2.csv")
    df_requests = pd.read_csv("./results/raw_requests_results_2.csv")

    requests = df_requests.Request.unique()
    requests = sorted(requests)

    #Request dictionary to map request to R1, R2, ...
    request_dict = {}
    for i, request in enumerate(requests):
        request_dict[request] = "R" + str(i+1)
    
    #Write the request dict to latex
    with open("./results/request_dict.tex", "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{ll}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Requête} & \\textbf{Label} \\\\\n")
        f.write("\\midrule\n")
        for request in requests:
            f.write(request_dict[request] + " & " + request.replace("_", "\_") + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Mapping des requêtes}\n")
        f.write("\\label{tab:request_dict}\n")
        f.write("\\end{table}\n")

    descriptors = df_requests.Feature_Extractor.unique()

    for d in descriptors:
        for brand_only in [True, False]:
            df_descriptor = pd.DataFrame(columns=["Request", "Recall@50", "Recall@100", "Recall@max", "Precision@50", "Precision@100", "Precision@max", "AP@50", "AP@100", "AP@max", "mAP@50", "mAP@100"])
            for request in requests:
                df = df_requests[(df_requests.Feature_Extractor == d) & (df_requests.Request == request) & (df_requests.brand_only == brand_only)]
                recall_50 = df[df.k == 50]["Recall@k"].values[0]
                recall_100 = df[df.k == 100]["Recall@k"].values[0]
                precision_50 = df[df.k == 50]["R-precision"].values[0]
                precision_100 = df[df.k == 100]["R-precision"].values[0]
                ap_50 = df[df.k == 50]["Average precision@k"].values[0]
                ap_100 = df[df.k == 100]["Average precision@k"].values[0]
                df = df_global[(df_global.Feature_Extractor == d) & (df_global.brand_only == brand_only)]
                mAP_50 = df[df.k == 50]["MAP"].values[0]
                mAP_100 = df[df.k == 100]["MAP"].values[0]
                
                df = df_requests[(df_requests.Feature_Extractor == d) & (df_requests.Request == request) & (df_requests.brand_only == brand_only)]
                if len(df.k.unique()) > 2:
                    print("More than 2 k values for " + d + " and " + request)
                    k_max = [x for x in df.k.unique() if x != 50 and x != 100][0]
                    recall_max = df[df.k == k_max]["Recall@k"].values[0]
                    precision_max = df[df.k == k_max]["R-precision"].values[0]
                    ap_max = df[df.k == k_max]["Average precision@k"].values[0]
                else:
                    recall_max, precision_max, ap_max = recall_100, precision_100, ap_100
                
                df_descriptor = pd.concat([df_descriptor, pd.DataFrame({"Request": [request_dict[request]], "Recall@50": [recall_50], "Recall@100": [recall_100], "Recall@max": [recall_max], "Precision@50": [precision_50], "Precision@100": [precision_100], "Precision@max": [precision_max], "AP@50": [ap_50], "AP@100": [ap_100], "AP@max": [ap_max], "mAP@50": [mAP_50], "mAP@100": [mAP_100]})])
            
            df_descriptor = df_descriptor.reset_index(drop=True)
            output_dir = "./results/brand/" if brand_only else "./results/brand_model/"
            requests_df_to_latex(df_descriptor, output_dir + d + "_2.tex")


    df = pd.DataFrame(columns=["Feature_Extractor", "mAP@50_brand", "mAP@100_brand", "mAP@50_brand_model", "mAP@100_brand_model"])
    for d in descriptors:
        mAP_50_brand = df_global[(df_global.Feature_Extractor == d) & (df_global.brand_only == True) & (df_global.k == 50)]["MAP"].values[0]
        mAP_100_brand = df_global[(df_global.Feature_Extractor == d) & (df_global.brand_only == True) & (df_global.k == 100)]["MAP"].values[0]
        mAP_50_brand_model = df_global[(df_global.Feature_Extractor == d) & (df_global.brand_only == False) & (df_global.k == 50)]["MAP"].values[0]
        mAP_100_brand_model = df_global[(df_global.Feature_Extractor == d) & (df_global.brand_only == False) & (df_global.k == 100)]["MAP"].values[0]
        df = pd.concat([df, pd.DataFrame({"Feature_Extractor": [d], "mAP@50_brand": [mAP_50_brand], "mAP@100_brand": [mAP_100_brand], "mAP@50_brand_model": [mAP_50_brand_model], "mAP@100_brand_model": [mAP_100_brand_model]})])

    df = df.reset_index(drop=True)
    output_dir = "./results/"
    global_df_to_latex(df, output_dir + "global_benchmark_2.tex")


    #Do the same with time
    df_time = pd.read_csv("./results/time_results.csv")
    df_time = df_time.round(3)

    order = ["ViT", "InceptionV4_2_steps_data_augmentation", "ResNet34_2_steps_data_augmentation", "ResNet34_ImageNet", "ORB", "SIFT", "HOG", "Histogram_Color", "Histogram_HSV", "LBP", "GLCM"]

    
    with open("./results/time_results.tex", "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l|cc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Descripteur} & \\textbf{Temps d'indexation (s)} & \\textbf{Temps de recherche moyen (s)} \\\\\n")
        f.write("\\midrule\n")
        for d in order: 
            if d not in df_time.Feature_Extractor.values:
                continue
            row = df_time[df_time.Feature_Extractor == d]
            f.write(d.replace("_", "\_") + " & " + str(row.Index_time.values[0]) + " & " + str(row.Research_time_mean.values[0]) + "($\pm$ " +  str(row.Research_time_std.values[0]) + ") \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Temps d'exécution des descripteurs}\n")
        f.write("\\label{tab:time_results}\n")
        f.write("\\end{table}\n")
        

       


                


