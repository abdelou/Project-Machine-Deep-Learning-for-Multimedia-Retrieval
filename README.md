# 🖼️ Multimedia Image Retrieval Engine (MIR)

Ce projet est un moteur de recherche d'images avancé combinant des approches de **Machine Learning classique** et de **Deep Learning** pour la recherche d'images unimodale et multimodale. Développé dans le cadre du module *Machine and Deep Learning for Multimedia Retrieval*.

---

## 🚀 Fonctionnalités Clés

### 1. 🔍 Recherche d'Images Unimodale (Image-to-Image)
Recherche basée sur le contenu visuel d'une image requête effectuée sur une base de données de 10 000 images ("Cars" dataset).
*   **Descripteurs Classiques** : SIFT, SURF, ORB, HOG, LBP, GLCM et Histogrammes de couleur (HSV/RGB).
*   **Deep Learning (CNN & ViT)** : Modèles pré-entraînés et fine-tunés : **ResNet34**, **InceptionV4** et **Vision Transformer (ViT)**.
*   **Fusion de Descripteurs** : Possibilité de fusionner plusieurs descripteurs pour améliorer la pertinence des résultats.

### 2. 🌐 Recherche Multimodale (Text-to-Image & Inverse)
Exploitation du dataset **Flickr8k** pour une recherche croisée texte/image.
*   **Modèle CLIP (OpenAI)** : Utilisation d'encodeurs texte et image pour projeter les données dans un espace latent commun.
*   **FAISS (Facebook AI Similarity Search)** : Indexation vectorielle pour des recherches de similarité ultra-rapides.
*   **Modes de recherche** : Recherche d'images par texte (Text-to-Image) et recherche d'images par similarité visuelle.

### 3. 📊 Évaluation & Analyse
*   **Métriques** : Calcul de la Précision, Rappel, AP, mAP et R-Precision.
*   **Visualisation** : Génération automatique de courbes Précision-Rappel.
*   **Reporting** : Export des résultats au format LaTeX pour l'intégration directe dans des rapports académiques.

### 4. ☁️ Déploiement & Interface
*   **Interface Web (SaaS)** : UI moderne développée avec **Flask**, permettant l'upload d'images et la recherche par texte.
*   **Interface Desktop** : GUI interactive avec **PyQt5** pour l'exploration détaillée des descripteurs.
*   **Docker** : Projet entièrement conteneurisé pour un déploiement simplifié en environnement Cloud.

---

## 🛠️ Technologies Utilisées
*   **Langage** : `Python 3.9+`
*   **Deep Learning** : `PyTorch`, `CLIP`, `Timm`
*   **Vision par Ordinateur** : `OpenCV`, `Scikit-Image`
*   **Indexation** : `FAISS`
*   **Web & UI** : `Flask` (Web), `PyQt5` (Desktop)

---

## 📦 Installation & Utilisation

### 1. Cloner le projet 
```bash
git clone https://github.com/abdelou/Project-Machine-Deep-Learning-for-Multimedia-Retrieval.git
cd Project-Machine-Deep-Learning-for-Multimedia-Retrieval
Installer les dépendances :  pip install -r requirements/requirements.txt
Lancer l'interface Web : python src/app.py
