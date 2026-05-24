# 🖼️ Multimedia Image Retrieval Engine (MIR)

Ce projet est un moteur de recherche d'images avancé combinant des approches de **Machine Learning classique** et de **Deep Learning** pour la recherche d'images unimodale et multimodale. Développé dans le cadre du module *Machine and Deep Learning for Multimedia Retrieval*.

| Interface Unimodale (ViT) | Recherche SIFT + Courbe PR | Recherche BGR + Courbe PR |
| :---: | :---: | :---: |
| ![Search ViT](docs/screenshots/vit_search.png) | ![Search SIFT](docs/screenshots/sift_search.png) | ![Search BGR](docs/screenshots/bgr_search.png) |

---

## 🚀 Fonctionnalités Clés (Alignement TP3)

### 1. 🔍 Recherche d'Images Unimodale (Image-to-Image)
Recherche basée sur le contenu visuel avec support multi-critères :
*   **Sélecteur de Descripteurs** : Choix dynamique parmi 8 descripteurs (**ViT, BGR, HSV, SIFT, ORB, LBP, GLCM, HOG**).
*   **Sélecteur de Distances** : Support de plusieurs métriques (**Euclidienne, Brute Force, Flann, Correlation, Intersection, Bhattacharyya, Chi-square**).
*   **Courbes Rappel/Précision (PR Curves)** : Génération automatique et affichage dynamique d'une courbe PR pour chaque recherche afin d'évaluer la pertinence.
*   **Optimisation de l'Index** : Système d'indexation binaire agrégée (`.npy`) pour gérer les 10 000 images de manière fluide et économe en espace disque.


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
```

### 2. Installer les dépendances
```bash
pip install -r requirements/requirements.txt
```

### 3. Lancement de l'interface Web
```bash
python3 src/app.py
```

```
L'interface sera disponible sur : http://127.0.0.1:5001
```
---

*Projet réalisé dans le cadre du module Machine and Deep Learning for Multimedia Retrieval - 2026*

