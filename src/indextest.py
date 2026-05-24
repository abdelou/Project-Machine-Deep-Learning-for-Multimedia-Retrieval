import os
from feature_extractor import Histogram_HSV_Extractor

# Chemin vers le dossier contenant les images
image_folder = "/Users/tolga/TP3_MIR/ProjetMathieuMohssine/flat_dataset"
output_folder = "features"

# Récupérer tous les fichiers d'images
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Instancier l'extracteur
extractor = Histogram_HSV_Extractor()

# Indexer la base de données et sauvegarder les caractéristiques
extractor.index_database(image_files, output_folder)

print("Extraction terminée. Les caractéristiques sont sauvegardées dans le dossier :", output_folder)