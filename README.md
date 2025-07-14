Diabetes Dataset Analyse & Outlier Detectie
1. Overzicht van de dataset
Eerst wordt de dataset ingeladen en bekeken om inzicht te krijgen in de structuur, variabelen en algemene eigenschappen. Dit helpt om het type data en eventuele bijzonderheden te begrijpen.

df.head() toont de eerste vijf rijen

df.shape geeft het aantal observaties en variabelen

df.describe().T geeft statistische samenvattingen van numerieke kolommen

df.dtypes toont het datatype per kolom

2. Bepalen van variabeltypen
In deze stap onderscheiden we categorische, numerieke en categorisch-uitziende maar kardinale variabelen. Dit is belangrijk om later de juiste analysetechnieken te gebruiken.

Categorisch: Kolommen met datatype object of met weinig unieke waarden

Numeriek: Kolommen met numerieke waarden, exclusief numeriek-uitziende categorische

Kardinaal: Categorische kolommen met veel unieke waarden (zoals ID’s)
De functie col_names_grab() bepaalt deze lijsten en print het overzicht.

3 & 4. Analyse van variabelen en doelvariabele
Gemiddelde waarde van de doelvariabele (Outcome) per categorie van categorische variabelen

Gemiddelden van numerieke variabelen per categorie van de doelvariabele
Hiermee krijg je inzicht in verbanden en verschillen tussen groepen.

5. Analyse van uitbijters (outliers)
Met de IQR-methode (thresholds_outlier) bepalen we grenzen waarboven of waaronder waarden als outlier kunnen gelden.

We controleren welke observaties buiten deze grenzen vallen.

Met de functie plot_all_numerical_with_outliers() worden per numerieke variabele histogrammen en boxplots getekend, inclusief percentielgrenzen (standaard 4e en 96e percentiel). Dit is visueel erg nuttig om uitschieters te identificeren.

Percentielmethode is gekozen in plaats van standaard IQR, omdat deze realistischere grenzen geeft bij deze dataset.

Outlier Detectie & Behandeling
Met outlier_check() wordt gecheckt of een kolom uitbijters bevat volgens percentielen.

outliers_grab() toont de daadwerkelijke outlier rijen, indien aanwezig.

replace_as_thresholds() vervangt extreme waarden door de grenswaarden om uitschieters te beperken zonder ze direct te verwijderen.

Geavanceerde Outlier Detectie: Local Outlier Factor (LOF)
LOF identificeert data punten die zich anders gedragen dan hun buren (lokale afwijkers).

Scores worden gesorteerd en geplot om een drempelwaarde te kiezen.

Observaties met LOF-score lager dan de drempel worden als outliers beschouwd.

Deze outliers worden verwijderd voor een schone dataset (df_clean).

Resultaten
Originele dataset grootte: aantal rijen vóór schoonmaken

Aantal geïdentificeerde outliers

Dataset grootte na verwijderen outliers

Code Annotaties en Doelen (in het kort)
Data inladen en verkennen: Inzicht krijgen in structuur en variabelen

Variabeltypen bepalen: Categorisch, numeriek, kardinaal

Doelvariabele-analyse: Gemiddelden per categorie voor interpretatie

Outlierdetectie: Zowel klassieke als percentielmethode gebruikt om extreme waarden te vinden

Visualisatie: Histogram en boxplot per numerieke variabele om uitschieters visueel te beoordelen

Outlier behandeling: Extreme waarden afkappen op grenswaarden of verwijderen via LOF

Dataset opschonen: Outliers verwijderen om analyses robuuster te maken
