import os
import cv2
import imgaug.augmenters as iaa
import numpy as np

# Dossier contenant les images
input_folder = r"C:\Users\saad_\OneDrive\Bureau\TruckClassification-v2\TruckClassification-v2\trainnig\dataset\déchargement remorque"
output_folder = r"C:\Users\saad_\OneDrive\Bureau\TruckClassification-v2\TruckClassification-v2\trainnig\dataset\déchargement_remorque"


# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Séquence d'augmentations
augmentation_seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Flip horizontal avec une probabilité de 50%
    iaa.Affine(rotate=(-25, 25)),  # Rotation aléatoire entre -25 et 25 degrés
    iaa.Multiply((0.8, 1.2)),  # Changer la luminosité
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # Ajouter du bruit
])

# Nombre d'augmentations par image
augmentations_per_image = 4

# Appliquer les augmentations et redimensionner chaque image
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):  # Filtrer les types d'image
        # Lire l'image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        # Vérifier si l'image est bien chargée
        if image is None:
            print(f"Erreur de lecture pour {filename}")
            continue
        
        # Redimensionner l'image originale à 224*224
        resized_image = cv2.resize(image, (224, 224))

        # Sauvegarder l'image originale redimensionnée dans le dossier de sortie
        output_original_path = os.path.join(output_folder, filename)  # Nom original
        cv2.imwrite(output_original_path, resized_image)
        print(f"Image originale redimensionnée sauvegardée : {output_original_path}")
        
        # Appliquer les augmentations plusieurs fois
        for i in range(augmentations_per_image):
            # Appliquer les augmentations
            augmented_images = augmentation_seq(images=[resized_image])

            # Sauvegarder chaque image augmentée avec un identifiant unique
            output_augmented_path = os.path.join(output_folder, f"aug_{i+1}_{filename}")
            cv2.imwrite(output_augmented_path, augmented_images[0])

            print(f"Image augmentée sauvegardée : {output_augmented_path}")

