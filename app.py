import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Titre de l'application
st.title("Exploration des Données avec Streamlit")

# Charger le fichier CSV (dataset existant)
df = pd.read_csv('data/Test.csv')  # Assurez-vous que 'Test.csv' est dans le même dossier que ce script

# Afficher un message d'accueil
st.write("**Dataset chargé avec succès !**")

# Afficher les premières lignes du dataset
st.header("Aperçu des données")
st.dataframe(df.head())  # Utilisation de st.dataframe pour un affichage interactif
st.sidebar.header('MENU')
# Sélectionner les colonnes à afficher
st.sidebar.header("Sélectionnez des colonnes à afficher")
colonnes = st.sidebar.multiselect("Choisissez les colonnes à afficher", df.columns.tolist(), default=df.columns.tolist())

# Afficher le dataset avec les colonnes sélectionnées
if colonnes:
    st.dataframe(df[colonnes])
else:
    st.write("Veuillez sélectionner au moins une colonne à afficher.")

# Afficher les informations sur le dataset
st.header("Informations sur le dataset")
if st.checkbox("Afficher les informations du dataset (types de colonnes, nombre de valeurs manquantes, etc.)"):
    buffer = df.info(buf=None)
    st.text(buffer)

# Afficher les statistiques descriptives
st.header("Statistiques descriptives")
if st.checkbox("Afficher les statistiques descriptives des colonnes numériques"):
    st.write(df.describe())

# Supprimez les espaces des noms des colonnes
df.columns = df.columns.str.strip()

 # Retirer les espaces dans les noms de colonnes
df.columns = df.columns.str.strip()
# Ajouter des filtres interactifs
st.sidebar.header("Filtres interactifs")
age_min, age_max = st.sidebar.slider("Sélectionner la plage d'âge", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
income_min, income_max = st.sidebar.slider("Sélectionner la plage de Family_Size", int(df['Family_Size '].min()), int(df['Family_Size'].max()), (int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max())))
gender_filter = st.sidebar.selectbox("Sélectionner le genre", options=['All', 'Male', 'Female'])

# Appliquer les filtres aux données
filtered_df = df[(df['Age'] >= age_min) & (df['Age'] <= age_max)]
filtered_df = filtered_df[(filtered_df['Family_Size'] >= income_min) & (filtered_df['Family_Size'] <= income_max)]
if gender_filter != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]

# Afficher les données filtrées
st.header("Données filtrées")
st.dataframe(filtered_df)

# 1. Histogramme de la distribution de l'Age
st.header("Distribution de l'Age")
plt.figure(figsize=(8, 4))
sns.histplot(filtered_df['Age'], kde=True)
st.pyplot()

# 2. Histogramme de la distribution du Spending Score
st.header("Distribution du Spending Score")
plt.figure(figsize=(8, 4))
sns.histplot(filtered_df['Spending_Score (1-100)'], kde=True)
st.pyplot()

# 3. Scatterplot pour la relation entre Annual Income et Spending Score
st.header("Relation entre Family_Sizeet Spending Score")
plt.figure(figsize=(8, 4))
sns.scatterplot(data=filtered_df, x='Family_Size', y='Spending Score (1-100)', hue='Gender')
st.pyplot()
df.columns = df.columns.str.strip()

# Ajouter des filtres interactifs
st.sidebar.header("Filtres interactifs")
age_min, age_max = st.sidebar.slider("Sélectionner la plage d'âge", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
family_size_min, family_size_max = st.sidebar.slider("Sélectionner la plage de Family_Size", int(df['Family_Size'].min()), int(df['Family_Size'].max()), (int(df['Family_Size'].min()), int(df['Family_Size'].max())))

# Appliquer les filtres aux données
filtered_df = df[(df['Age'] >= age_min) & (df['Age'] <= age_max)]
filtered_df = filtered_df[(filtered_df['Family_Size'] >= family_size_min) & (filtered_df['Family_Size'] <= family_size_max)]

# Afficher les données filtrées
st.header("Données filtrées")
st.dataframe(filtered_df)

# Afficher les statistiques descriptives des données filtrées
st.header("Statistiques descriptives des données filtrées")
st.write(filtered_df.describe())

# 1. Histogramme de la distribution de l'Age
st.header("Distribution de l'Age")
plt.figure(figsize=(8, 4))
sns.histplot(filtered_df['Age'], kde=True)
st.pyplot()

# 2. Histogramme de la distribution du Spending Score
st.header("Distribution du Spending Score")
plt.figure(figsize=(8, 4))
sns.histplot(filtered_df['Spending_Score (1-100)'], kde=True)
st.pyplot()

# 3. Scatterplot pour la relation entre Family_Size et Spending Score
st.header("Relation entre Family_Size et Spending Score")
plt.figure(figsize=(8, 4))
sns.scatterplot(data=filtered_df, x='Family_Size', y='Spending_Score (1-100)', hue='Gender')
st.pyplot()
# 5. Résumé des résultats et conclusion
st.header("Résumé des résultats")
st.write("""
    - **Distribution de l'Age** : Les données montrent une distribution spécifique de l'âge, avec une concentration notable dans certaines plages d'âge.
    - **Spending Score** : La distribution du Spending Score est indiquée pour chaque groupe d'âge et de taille de famille.
    - **Relation entre Family_Size et Spending Score** : La relation entre la taille de la famille et le Spending Score peut être explorée par un graphique interactif.
    - **Filtres appliqués** : Les utilisateurs peuvent filtrer les données en fonction de l'âge et de la taille de la famille pour affiner l'analyse.
""")

# 6. Téléchargement des résultats sous forme de CSV
@st.cache
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(filtered_df)
st.download_button(
    label="Télécharger les données filtrées en CSV",
    data=csv,
    file_name='data_filtered.csv',
    mime='text/csv',
)

# 7. Téléchargement du graphique de distribution de l'Age sous forme de PNG
def save_plot_as_png():
    plt.figure(figsize=(8, 4))
    sns.histplot(filtered_df['Age'], kde=True)
    plt.title("Distribution de l'Age")
    plt.savefig('age_distribution.png')

save_plot_as_png()

with open('age_distribution.png', 'rb') as file:
    st.download_button(
        label="Télécharger le graphique de la distribution de l'Age",
        data=file,
        file_name="age_distribution.png",
        mime="image/png"
    )