# Project-Machine-Deep-Learning-for-Multimedia-Retrieval
Ce projet est un moteur de recherche d'images avancé combinant des approches de Machine Learning classique et de Deep Learning pour la recherche d'images unimodale et multimodale. Développé dans le cadre du module Machine and Deep Learning for Multimedia Retrieval.

#Fonctionnalités Clés

1. 🔍 Recherche d'Images Unimodale (Image-to-Image)
Recherche basée sur le contenu visuel d'une image requête sur une base de données de 10 000 images ("Cars" dataset).

Descripteurs Classiques : SIFT, SURF, ORB, HOG, LBP, GLCM et Histogrammes de couleur (HSV/RGB).
Deep Learning (CNN & ViT) : Intégration de modèles pré-entraînés et fine-tunés : ResNet34, InceptionV4 et Vision Transformer (ViT).
Fusion de Descripteurs : Possibilité de combiner plusieurs types de caractéristiques pour améliorer la précision.

2. 🌐 Recherche Multimodale (Text-to-Image & Inverse)
Exploitation du dataset Flickr8k pour une recherche croisée texte/image.

Modèle CLIP (OpenAI) : Utilisation d'encodeurs texte et image pour projeter les données dans un espace latent commun.
FAISS (Facebook AI Similarity Search) : Indexation vectorielle ultra-rapide pour une recherche de similarité en temps réel.
Modes : Recherche d'images par texte (Text-to-Image) et recherche d'images similaires (Image-to-Image).

3. 📊 Évaluation & Analyse
Métriques : Calcul de la Précision, Rappel, AP, mAP et R-Precision.
Visualisation : Génération automatique de courbes Précision-Rappel.
Reporting : Export automatique des résultats en tableaux LaTeX pour des rapports académiques.

4. ☁️ Déploiement & Interface
Interface Web (SaaS) : UI moderne développée avec Flask, permettant l'upload d'images et la recherche par texte.
Interface Desktop : GUI interactive en PyQt5 pour l'exploration détaillée des descripteurs.
Docker : Projet entièrement conteneurisé pour un déploiement simplifié en environnement Cloud.

🛠️ Technologies
Langage : Python 3.9+
Deep Learning : PyTorch, CLIP, Timm
Vision par ordinateur : OpenCV, Scikit-Image
Indexation : FAISS
Interface : Flask (Web), PyQt5 (Desktop)

📦 Installation & Utilisation
git clone https://github.com/abdelou/Project-Machine-Deep-Learning-for-Multimedia-Retrieval.git
Installation des dépendances : pip install -r requirements/requirements.txt
Lancement de l'interface Web : python src/app.py 


