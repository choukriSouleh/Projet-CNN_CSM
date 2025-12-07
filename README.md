# Projet-CNN_CSM

# Détection de Deepfakes avec XceptionNet

# Description
Ce projet implémente un système avancé de détection de deepfakes en utilisant une version modifiée du modèle XceptionNet, pré‑entraîné sur ImageNet.
Le modèle est ensuite fine‑tuné sur le dataset FaceForensics++ (C23), un des datasets les plus utilisés pour l'analyse de manipulations faciales.

# Notre objectif est classifier une image en Originale ou Deepfake.

 Donc dans ce projet on va : 
 
- Charger et adapter l’architecture XceptionNet pour la tâche de classification binaire

- Prétraiter et structurer correctement le dataset FaceForensics++ C23

- Entraîner un modèle robuste et stable

- Atteindre une accuracy élevée (>95%)

- Générer un rapport d’évaluation complet (precision, recall, f1‑score)

# Articles de Référence
Rossler et al. (2019)
"FaceForensics++: Learning to Detect Manipulated Facial Images"
 arXiv:1901.08971

# Technologies Utilisées
Langage : Python 3

Framework : PyTorch

Librairies : torchvision, pandas, numpy, scikit‑learn, matplotlib & seaborn

Environnement : Google Colab 

 # Dataset
Nom : FaceForensics++ C23

Source : https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23

Images : ~500k (50% Original, 50% Deepfake)

Split : 70% train, 15% validation, 15% test

Format : Images JPEG 299×299 pixels

# Architecture du Modèle
XceptionNet (Chollet, 2017) pré-entraîné sur ImageNet, modifié pour 2 classes.

Caractéristiques :

Backbone Xception (convolutions séparables en profondeur)

Fine-tuning complet (toutes couches entraînables)

Classifieur final : 2048 → 2 neurones

Fonction de perte : CrossEntropyLoss avec pondération des classes

Optimiseur : Adam (LR=1e-4)

# Résultats
Métriques sur Test Set (1500 échantillons)
Classe	           Precision	Recall	 F1-Score	  Support

Original (0)	      95.03%	  99.33%	  97.13%	     750

Deepfake (1)	      99.30%	  94.80%	  97.00%	     750

Accuracy	          97.07%	  97.07%	  97.07%	     1500

# Accuracy totale : 97.07%
Le modèle atteint une robustesse très élevée pour distinguer deepfakes et images authentiques.

# Conclusion
Ce projet démontre la puissance du transfer learning avec XceptionNet pour la détection de deepfakes.
Avec une accurate de 97%, le modèle offre une excellente fiabilité tout en restant optimisé pour l’inférence sur GPU.

