import os
import pandas as pd
from tqdm import tqdm
from distances import distance_f
from metrics import recall, r_precision, average_precision, mean_average_precision
from feature_extractor import Histogram_HSV_Extractor
from retrieval import extract_class_id

# === CONFIGURATION ===
DATASET_DIR = "src/flat_dataset"
DESCRIPTOR = Histogram_HSV_Extractor()
DISTANCE = "Euclidienne"
DESCRIPTOR_NAME = "BGR"
TOP_K_VALUES = [50, 100]

# Images requêtes associées à R1 à R9
REQUESTS = [
    "0_5_araignees_tarantula_795.jpg",
    "0_4_araignees_gardenspider_631.jpg",
    "0_1_araignees_wolfspider_259.jpg",
    "1_0_chiens_Siberianhusky_849.jpg",
    "1_3_chiens_Chihuahua_1315.jpg",
    "1_1_chiens_Labradorretriever_1054.jpg",
    "2_2_oiseaux_greatgreyowl_2092.jpg",
    "2_4_oiseaux_robin_2359.jpg",
    "2_3_oiseaux_bluejay_2232.jpg"
]

# === Chargement des images de la base ===
images = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.lower().endswith(".jpg")]

# === Indexation ===
print(f"📥 Indexation des descripteurs {DESCRIPTOR_NAME}...")
DESCRIPTOR.index_database(images, "./descriptors", save_files=True)
features = DESCRIPTOR.load_features("./descriptors")

# === Évaluation ===
results = []
all_AP_50 = []
all_AP_100 = []

for idx, req in enumerate(REQUESTS):
    req_path = os.path.join(DATASET_DIR, req)
    req_class = extract_class_id(req)
    feature_query = DESCRIPTOR.extract_feature(req_path)

    # Nombre d’images pertinentes dans la base (TopMax)
    topmax = sum(extract_class_id(img) == req_class for img in os.listdir(DATASET_DIR))

    # Recherche
    dists = [(f, distance_f(feature_query, feat, DISTANCE)) for f, feat in features]
    dists.sort(key=lambda x: x[1])
    retrieved_classes = [extract_class_id(f) for f, _ in dists]

    result_row = {"Requête": f"R{idx+1}", "TopMax": topmax}

    for k in TOP_K_VALUES:
        r = recall(k, retrieved_classes, [req_class], topmax)
        p = r_precision(k, retrieved_classes, [req_class])
        ap = average_precision(k, retrieved_classes, [req_class])

        result_row[f"R@{k}"] = round(r, 3)
        result_row[f"P@{k}"] = round(p, 3)
        result_row[f"AP@{k}"] = round(ap, 3)

        if k == 50:
            all_AP_50.append(ap)
        if k == 100:
            all_AP_100.append(ap)

    results.append(result_row)

# === DataFrame final ===
df = pd.DataFrame(results)

# Ajout du mAP
map50 = round(mean_average_precision(all_AP_50), 3)
map100 = round(mean_average_precision(all_AP_100), 3)
df.loc[len(df.index)] = {"Requête": "mAP", "AP@50": map50, "AP@100": map100}

# === Affichage formaté ===
print("\n" + "="*70)
print(" TABLEAU 2 – Précision du moteur de recherche (BGR + Euclidienne) ")
print("="*70)
print(df.to_string(index=False))
print("="*70)

# Sauvegarde CSV
df.to_csv("tableau_2_resultats.csv", index=False)