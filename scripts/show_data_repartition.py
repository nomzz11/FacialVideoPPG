import os
import pandas as pd
import seaborn as sns
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
ENV: str = os.getenv("ENV")

# TODO : use etl lib to get json data
from fetchData import getData
from fetchData import extraire_donnees_json

if ENV == "development":
    path2json = getData(ENV)[0]
    data = getData(ENV)[1]

    metadata = extraire_donnees_json(path2json)

    # Affiche les données extraites
    # for element in metadata:
    # print(element)

    dfMetadata = pd.DataFrame(metadata)
    dfMetadata["location"] = dfMetadata[
        "location"
    ].str.strip()  ##On enlève l'espace après un des lables T contenu dans la colonne "location"

    plt.rcParams.update(
        {
            "axes.titlesize": 22,  # Taille du titre du graphique
            "axes.labelsize": 20,  # Taille des étiquettes des axes
            "xtick.labelsize": 16,  # Taille des labels de l'axe X
            "ytick.labelsize": 16,
        }
    )

    # Pie chart de la répartition du genre des individus
    dfMetadata["gender"] = pd.Categorical(
        dfMetadata["gender"], categories=["M", "F"], ordered=True
    )
    genderCounts = (
        dfMetadata["gender"].value_counts(normalize=True) * 100
    ).sort_index()
    plt.figure(figsize=(8, 8))
    plt.pie(
        genderCounts,
        labels=genderCounts.index,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 16},
    )
    palette = {"M": "#1f77b4", "F": "#ff7f0e"}
    plt.title("Répartition des individus par genre")
    plt.show()

    # Histogramme des âges
    dfAgeSorted = dfMetadata.sort_values(by="age", ascending=True)
    dfAgeSorted["age"] = pd.to_numeric(dfAgeSorted["age"], errors="coerce")
    bins = [i for i in range(0, 101, 10)]
    age_groups = pd.cut(dfAgeSorted["age"], bins=bins)
    age_counts = age_groups.value_counts().sort_index()
    x_labels = [f"{bin_left}-{bin_left+9}" for bin_left in bins[:-1]]
    plt.figure(figsize=(12, 6))
    plt.bar(x_labels, age_counts, color="lightblue", edgecolor="black")
    plt.xlabel("Tranche d'âge (années)")
    plt.ylabel("Nombre d'individus")
    plt.title("Répartition des individus par tranche d'âge")
    plt.tight_layout()
    plt.show()

    # Boxplot des âges par genre
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="gender", y="age", data=dfAgeSorted, palette=palette)
    plt.title("Distribution des âges par genre")
    plt.xlabel("Genre")
    plt.ylabel("Âge (années)")
    plt.show()

    # Pie chart de la répartition de fitzpatrick
    fitzpatrickCounts = (
        dfMetadata["fitzpatrick"].value_counts(normalize=True) * 100
    ).sort_index()
    fitzpatrickCountsSorted = fitzpatrickCounts[::-1]
    plt.figure(figsize=(8, 8))
    plt.pie(
        fitzpatrickCountsSorted,
        labels=fitzpatrickCountsSorted.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Set2", len(fitzpatrickCountsSorted)),
        textprops={"fontsize": 16},
    )
    plt.title("Répartition des individus par Fitzpatrick")
    plt.show()

    # Pie chart de la répartition de la localisation
    locationCounts = (
        dfMetadata["location"].value_counts(normalize=True) * 100
    ).sort_index()
    plt.figure(figsize=(8, 8))
    plt.pie(
        locationCounts,
        labels=locationCounts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Set2", len(locationCounts)),
        textprops={"fontsize": 16},
    )
    plt.title("Répartition des individus par localisation")
    plt.show()

    # Histogramme des bp_sys
    dfBpSysSorted = dfMetadata.sort_values(by="bp_sys", ascending=True)
    bins = [
        i
        for i in range(
            dfBpSysSorted["bp_sys"].min(), dfBpSysSorted["bp_sys"].max() + 5, 5
        )
    ]
    bp_sys_groups = pd.cut(dfBpSysSorted["bp_sys"], bins=bins)
    bp_sys_counts = bp_sys_groups.value_counts().sort_index()
    x_labels = [f"{bin_left}-{bin_left+4}" for bin_left in bins[:-1]]
    plt.figure(figsize=(12, 6))
    plt.bar(x_labels, bp_sys_counts, color="lightblue", edgecolor="black")
    plt.xlabel("Tranche de tension systolique (mmHg)")
    plt.ylabel("Nombre d'individus")
    plt.title("Répartition des individus par tranche de tension systolique")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Histogramme des bp_dia
    dfBpDiaSorted = dfMetadata.sort_values(by="bp_dia", ascending=True)
    bins = [
        i
        for i in range(
            dfBpDiaSorted["bp_dia"].min(), dfBpDiaSorted["bp_dia"].max() + 5, 5
        )
    ]
    bp_dia_groups = pd.cut(dfBpDiaSorted["bp_dia"], bins=bins)
    bp_dia_counts = bp_dia_groups.value_counts().sort_index()
    x_labels = [f"{bin_left}-{bin_left+4}" for bin_left in bins[:-1]]
    plt.figure(figsize=(12, 6))
    plt.bar(x_labels, bp_dia_counts, color="lightblue", edgecolor="black")
    plt.xlabel("Tranche de tension diastolique (mmHg)")
    plt.ylabel("Nombre d'individus")
    plt.title("Répartition des individus par tranche de tension diastolique")
    plt.xticks(rotation=45)
    plt.yticks(range(0, max(bp_dia_counts) + 2, 2))
    plt.tight_layout()
    plt.show()
