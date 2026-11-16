import streamlit as st
import numpy as np
import joblib
import time

from PIL import Image
from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

# Configurer la page Streamlit : titre, icône et disposition
st.set_page_config(
    page_title="Face Recognition",
    page_icon="👤",
    layout="wide"
)


# ==========================================
# 2. STYLE
# ==========================================

# Personnaliser l'apparence de l'application avec CSS
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #2563EB;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 17px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #E8F1FF;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. TITRE
# ==========================================

# Afficher le titre principal de l'application
st.markdown(
    '<div class="main-title">👤 Face Recognition System</div>',
    unsafe_allow_html=True
)

# Afficher le sous-titre qui présente l'objectif du projet
st.markdown(
    '<div class="subtitle">Comparaison des algorithmes Eigenfaces et Fisherfaces</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()


# ==========================================
# 4. CHARGEMENT DES MODELES
# ==========================================

# Définir le chemin du fichier contenant les modèles entraînés
# Le fichier modeles_orl.joblib contient PCA, LDA et les deux KNN
MODEL_PATH = Path(__file__).parent / "modeles_orl.joblib"


# Charger les modèles depuis le fichier Joblib
# cache_resource évite de recharger les modèles à chaque interaction
@st.cache_resource
def charger_modeles():
    return joblib.load(MODEL_PATH)


# Essayer de charger les modèles
try:
    modeles = charger_modeles()

# En cas d'erreur, afficher un message et arrêter l'application
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()


# ==========================================
# 5. PRESENTATION
# ==========================================

# Présenter la fonctionnalité de reconnaissance faciale
st.subheader("📸 Reconnaissance faciale")

st.write(
    "Chargez une image pour identifier la personne "
    "à l'aide des deux algorithmes."
)

# Permettre à l'utilisateur de sélectionner une image
fichier = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png", "pgm"]
)


# ==========================================
# 6. TRAITEMENT DE L'IMAGE
# ==========================================

# Vérifier si une image a été chargée
if fichier is not None:

    # Ouvrir l'image et la convertir en niveaux de gris
    image = Image.open(fichier).convert("L")

    # Diviser l'interface en deux colonnes
    col1, col2 = st.columns([1, 2])

    with col1:

        # Afficher l'image importée
        st.image(
            image,
            caption="Image chargée",
            width=250
        )

    with col2:

        # Afficher les informations relatives à l'image
        st.write("### Informations")
        st.write("Dimensions originales :", image.size)
        st.write("Format :", fichier.type)

        st.info(
            "L'image sera redimensionnée en 92 × 112 pixels "
            "pour correspondre au dataset ORL."
        )

    st.divider()

    # Lancer le traitement lorsque l'utilisateur clique sur le bouton
    if st.button("🔍 Lancer la reconnaissance", use_container_width=True):

        # ----------------------------------
        # PREPARATION DE L'IMAGE
        # ----------------------------------

        # Redimensionner l'image pour obtenir les mêmes dimensions
        # que les images du dataset ORL : 92 pixels de largeur et 112 de hauteur
        image_preparee = image.resize((92, 112))

        # Convertir l'image en tableau numérique NumPy
        image_array = np.array(image_preparee)

        # Transformer la matrice 112 × 92 en un vecteur de 10304 valeurs
        # Le format obtenu est (1, 10304), compatible avec les modèles entraînés
        image_flat = image_array.reshape(1, -1)


        # ==================================
        # EIGENFACES (PCA) + KNN
        # ==================================

        # Enregistrer le temps de début du traitement PCA
        debut_pca = time.perf_counter()

        # ETAPE 1 : EXTRACTION DES CARACTERISTIQUES AVEC PCA
        # Le modèle PCA déjà entraîné transforme l'image dans un espace réduit
        # Il conserve les principales variations présentes dans les images
        X_pca = modeles["pca"].transform(image_flat)

        # ETAPE 2 : CLASSIFICATION AVEC KNN
        # Le KNN utilise les caractéristiques extraites par PCA
        # Il recherche les 3 voisins les plus proches parmi les images d'entraînement
        # Puis il attribue à l'image la classe majoritaire de ces voisins
        prediction_pca = modeles["knn_pca"].predict(X_pca)

        # Calculer le temps total de transformation et de classification
        temps_pca = time.perf_counter() - debut_pca

        # Récupérer le numéro de la personne prédite
        personne_pca = int(prediction_pca[0])


        # ==================================
        # FISHERFACES (LDA) + KNN
        # ==================================

        # Enregistrer le temps de début du traitement LDA
        debut_lda = time.perf_counter()

        # ETAPE 1 : EXTRACTION DES CARACTERISTIQUES AVEC LDA
        # Le modèle LDA déjà entraîné transforme l'image
        # Il cherche les caractéristiques qui distinguent le mieux les personnes
        X_lda = modeles["lda"].transform(image_flat)

        # ETAPE 2 : CLASSIFICATION AVEC KNN
        # Le KNN utilise les caractéristiques extraites par LDA
        # Il recherche les 3 voisins les plus proches
        # Puis il prédit l'identité de la personne à partir de leurs classes
        prediction_lda = modeles["knn_lda"].predict(X_lda)

        # Calculer le temps total de transformation et de classification
        temps_lda = time.perf_counter() - debut_lda

        # Récupérer le numéro de la personne prédite
        personne_lda = int(prediction_lda[0])


        # ==================================
        # AFFICHAGE DES RESULTATS
        # ==================================

        # Afficher le titre de la section des résultats
        st.subheader("📊 Résultats de reconnaissance")

        # Diviser l'affichage en deux colonnes pour comparer les modèles
        col_pca, col_lda = st.columns(2)

        with col_pca:

            # Afficher les résultats du modèle Eigenfaces
            st.markdown("### 🔵 Eigenfaces (PCA)")

            # Afficher l'identité prédite par PCA + KNN
            st.metric(
                "Personne identifiée",
                f"Personne {personne_pca}"
            )

            # Afficher le temps de traitement en millisecondes
            st.metric(
                "Temps de prédiction",
                f"{temps_pca * 1000:.3f} ms"
            )

        with col_lda:

            # Afficher les résultats du modèle Fisherfaces
            st.markdown("### 🟢 Fisherfaces (LDA)")

            # Afficher l'identité prédite par LDA + KNN
            st.metric(
                "Personne identifiée",
                f"Personne {personne_lda}"
            )

            # Afficher le temps de traitement en millisecondes
            st.metric(
                "Temps de prédiction",
                f"{temps_lda * 1000:.3f} ms"
            )

        st.divider()


        # ==================================
        # COMPARAISON FINALE
        # ==================================

        # Comparer les identités prédites par les deux modèles
        st.subheader("🏆 Comparaison finale")

        # Vérifier si les deux modèles donnent la même prédiction
        if personne_pca == personne_lda:

            # Si les prédictions sont identiques, afficher un message positif
            st.success(
                f"Les deux modèles sont d'accord : "
                f"Personne {personne_lda}"
            )

        else:

            # Si les prédictions sont différentes, afficher un avertissement
            st.warning(
                "Les deux modèles donnent des prédictions différentes."
            )

            # Afficher séparément les deux identités prédites
            st.write(f"Eigenfaces : Personne {personne_pca}")
            st.write(f"Fisherfaces : Personne {personne_lda}")

        # Préciser ce qui est inclus dans le temps mesuré
        st.caption(
            "Les temps affichés correspondent à la transformation "
            "et à la prédiction, sans compter le chargement du modèle."
        )


# ==========================================
# 7. PIED DE PAGE
# ==========================================

# Ajouter une séparation et une mention du projet
st.divider()

st.markdown(
    "<center>Projet académique | Reconnaissance faciale "
    "avec PCA et LDA | Dataset ORL</center>",
    unsafe_allow_html=True
)