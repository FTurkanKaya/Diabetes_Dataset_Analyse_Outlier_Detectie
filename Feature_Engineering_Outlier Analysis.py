import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#!pip install missingno
import missingno as msno
from datetime import date

from skimage.feature import shape_index
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

diabetes = "/Users/onayk/PycharmProjects/Python_FutureEngineering/Datasets/diabetes.csv"

def load(dataset):
    data = pd.read_csv(dataset)
    return data

##########################
#     1. Overzicht


#  Eerst wordt de algemene toestand van de dataset bekeken om een beter begrip te krijgen
#  van de structuur, de variabelen en mogelijke bijzonderheden in de gegevens.
##########################

df = load(diabetes)
df.head()

df.shape
df.describe().T

df.dtypes

##########################
#  2. Bepalen van variabeltypen

#  In deze stap worden de verschillende typen variabelen (bijvoorbeeld numeriek, categorisch)
#  geïdentificeerd om te bepalen welke analysetechnieken en transformaties geschikt zijn.
##########################


def col_names_grab(dataframe, cat_th=10, car_th=20):
    """
    Geeft de namen van de categorische, numerieke en categorisch-ogende maar kardinale variabelen in de dataset.
    Let op: Bij de categorische variabelen worden ook numeriek-ogende categorische variabelen meegenomen.

    Parameters
    ------
        dataframe: dataframe
                De dataframe waarvan de variabelnamen worden opgehaald
        cat_th: int, optioneel
                Drempelwaarde voor het aantal klassen waarbij een numerieke variabele als categorisch wordt beschouwd
        car_th: int, optioneel
                Drempelwaarde voor het aantal klassen waarbij een categorische variabele als kardinaal wordt beschouwd

    Returns
    ------
        cat_cols: list
                Lijst met categorische variabelen
        num_cols: list
                Lijst met numerieke variabelen
        cat_but_car: list
                Lijst met categorisch-ogende kardinale variabelen

    Voorbeelden
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(col_names_grab(df))

    Notities
    ------
        cat_cols + num_cols + cat_but_car = totaal aantal variabelen
        num_but_cat zit in cat_cols.
        De som van de drie geretourneerde lijsten is gelijk aan het totale aantal variabelen: cat_cols + num_cols + cat_but_car = aantal variabelen
"""

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}, {cat_cols}')
    print(f'num_cols: {len(num_cols)}, {num_cols}')
    print(f'cat_but_car: {len(cat_but_car)}, {cat_but_car}')
    print(f'num_but_cat: {len(num_but_cat)}, {num_but_cat}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = col_names_grab(df)

##########################
#  3. Analyse van numerieke & categorische variabelen
#  4. Gemiddelde van de doelvariabele per categorie van de categorische variabelen
#     Gemiddelden van de numerieke variabelen per categorie van de doelvariabele

#  In deze stap worden de distributies van numerieke en categorische variabelen afzonderlijk geanalyseerd
#  om inzicht te krijgen in de verdeling, frequenties en mogelijke bijzonderheden in de gegevens.
##########################

def doelvariabele_analyse(dataframe, target, cat_cols, num_cols):
    """
    Analyse van de doelvariabele:
    1. Gemiddelde van de doelvariabele per categorie van categorische variabelen
    2. Gemiddelden van numerieke variabelen per categorie van de doelvariabele

    Parameters
    ----------
    dataframe : pandas.DataFrame
        De dataset
    target : str
        Naam van de doelvariabele
    cat_cols : list
        Lijst van categorische variabelen
    num_cols : list
        Lijst van numerieke variabelen
    """

    print("\n Gemiddelde van de doelvariabele per categorie van categorische variabelen")
    for col in cat_cols:
        print(f"\n→ {col} :")
        print(dataframe.groupby(col)[target].mean())

    print("\n\n Gemiddelden van numerieke variabelen per categorie van de doelvariabele")
    for col in num_cols:
        print(f"\n→ {col} :")
        print(dataframe.groupby(target)[col].mean())

doelvariabele_analyse(df, "Outcome",cat_cols, num_cols)

##########################
#  5. Analyse van Outliers

# In deze stap worden de gegevens gecontroleerd op mogelijke uitbijters (extreme waarden)
# die de analyses kunnen beïnvloeden. Er wordt onderzocht welke observaties ver buiten
# de verwachte verdeling liggen.
##########################

#*******************

def plot_all_numerical_with_outliers(df, lower_pct=4, upper_pct=96):
    """
    Tüm sayısal sütunlar için hem histogram hem boxplot çizer,
    %1-%99 percentile sınırlarını gösterir ve aykırı değer sayısını yazar.
    """

    for col in num_cols:
        data = df[col].dropna()
        low = np.percentile(data, lower_pct)
        up = np.percentile(data, upper_pct)

        outliers = data[(data < low) | (data > up)]
        n_outliers = len(outliers)

        plt.figure(figsize=(12, 5))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(data, bins=30, color='skyblue', edgecolor='black')
        plt.axvline(low, color='red', linestyle='--', label=f'{lower_pct}e percentiel')
        plt.axvline(up, color='green', linestyle='--', label=f'{upper_pct}e percentiel')
        plt.title(f"Histogram van {col}")
        plt.xlabel(col)
        plt.ylabel("Aantal")
        plt.legend()

        # Boxplot
        plt.subplot(1, 2, 2)
        plt.boxplot(data, vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
        plt.axvline(low, color='red', linestyle='--', label=f'{lower_pct}e percentiel')
        plt.axvline(up, color='green', linestyle='--', label=f'{upper_pct}e percentiel')
        plt.title(f"Boxplot van {col} \n Aantal uitschieters buiten {lower_pct}-{upper_pct} percentiel: {n_outliers}")
        plt.xlabel(col)
        plt.legend()

        plt.tight_layout()
        plt.show()


plot_all_numerical_with_outliers(df)

#  ********************************************************
# Het doel van deze functie is om de verdeling van de numerieke variabelen visueel te analyseren en
# mogelijke uitschieters te identificeren. De functie genereert voor elke numerieke variabele een histogram en
# een boxplot, en geeft daarbij de onder- en bovengrens weer op basis van geselecteerde percentielen
# (bijvoorbeeld het 5e en het 95e percentiel).
#
# Ik heb ervoor gekozen om de percentielmethode te gebruiken in plaats van de traditionele IQR-methode,
# omdat de IQR-drempelwaarden in deze dataset vaak onder de minimale waarden liggen en
# daardoor geen realistische uitschieters detecteren.
# De percentielmethode geeft flexibelere grenzen en sluit beter aan bij de werkelijke verdeling van de data.
#
# Door de grenzen visueel weer te geven samen met de verdeling kan ik beter beoordelen
# welke waarden als uitschieters moeten worden beschouwd en of deze eventueel verwijderd of aangepast moeten worden.
#  *********************************************************

def get_percentile_bounds(series, lower_pct=4, upper_pct=96):
    """
    Berekent de onder- en bovengrens van een variabele op basis van percentielen.
    """
    lower = np.percentile(series.dropna(), lower_pct)
    upper = np.percentile(series.dropna(), upper_pct)
    return lower, upper

low, up = get_percentile_bounds(df)

df[(df["Outcome"] < low) | (df["Outcome"] > up)].head()

df[(df["Outcome"] < low) | (df["Outcome"] > up)].index


def outlier_check(dataframe, col_name):
    low_limit, up_limit = get_percentile_bounds(dataframe)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(f"{col} = {outlier_check(df, col)}")


###################
# De outliers zelf bekijken
###################

def outliers_grab(dataframe, col_name, index=False):
    low, up = get_percentile_bounds(dataframe)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


outliers_grab(df, "Glucose", True)
outliers_grab(df, "Insulin", True)

((df['Insulin'] < low) | (df['Insulin'] > up)).sum()
((df['Glucose'] < low) | (df['Glucose'] > up)).sum()

###################
# Re-assignment with thresholds
###################

def replace_as_thresholds(dataframe, column):
    low_limit, up_limit = get_percentile_bounds(dataframe)
    dataframe.loc[(dataframe[column] < low_limit), column] = low_limit
    dataframe.loc[(dataframe[column] > up_limit), column] = up_limit

outlier_check(df, "Insulin")
outlier_check(df, "Glucose")

replace_as_thresholds(df, "Insulin")
replace_as_thresholds(df, "Glucose")


#############################################
#  Local Outlier Factor
#############################################
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

#  De afstandsscores (LOF scores) bekijken
df_scores = clf.negative_outlier_factor_
df_scores[0:5]

# df_scores = -df_scores
np.sort(df_scores)[0:5]

# Drempelwaarde kiezen
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

# Drempelwaarde toevoegen
th = np.sort(df_scores)[13]

# Kleiner dan drempelwaarde
df[df_scores < th ]

# Aantal ven drempelwaarde
df[df_scores < th].shape

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

# IndexInfo
outlier_indices = df[df_scores < th].index

df.shape


# Verwijder de outliers uit de dataframe
df_clean = df.drop(index=outlier_indices)

print(f"Originele dataset grootte: {df.shape}")
print(f"Aantal outliers: {len(outlier_indices)}")
print(f"Gereinigde dataset grootte: {df_clean.shape}")









