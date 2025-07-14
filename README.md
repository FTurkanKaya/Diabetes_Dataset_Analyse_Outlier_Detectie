Diabetes Dataset Analyse & Outlier Detectie
Dit project voert een exploratieve data-analyse (EDA) en outlierdetectie uit op de bekende diabetes dataset.
Hieronder worden de stappen, methoden en doelen kort uitgelegd.

📋 Overzicht van de dataset
Eerst wordt de dataset ingeladen en bekeken om inzicht te krijgen in de structuur, variabelen en algemene eigenschappen. Dit helpt om het type data en eventuele bijzonderheden te begrijpen.

df.head() toont de eerste vijf rijen.

df.shape geeft het aantal observaties en variabelen.

df.describe().T geeft statistische samenvattingen van numerieke kolommen.

df.dtypes toont het datatype per kolom.

🔢 Bepalen van variabeltypen
In deze stap worden de variabelen onderverdeeld in:

Categorisch: kolommen met datatype object of met weinig unieke waarden.

Numeriek: kolommen met numerieke waarden (exclusief numeriek-uitziende categorische).

Kardinaal: categorische kolommen met veel unieke waarden (zoals ID’s).

De functie col_names_grab() bepaalt deze lijsten en toont een overzicht. Dit is belangrijk om later de juiste analysetechnieken te gebruiken.

📊 Analyse van variabelen en doelvariabele
We onderzoeken verbanden tussen variabelen en de doelvariabele (Outcome):

Gemiddelde waarde van de doelvariabele per categorie van categorische variabelen.

Gemiddelden van numerieke variabelen per categorie van de doelvariabele.

Dit geeft inzicht in verschillen tussen groepen en mogelijke patronen in de data.

🚨 Analyse van uitbijters (outliers)
We controleren op mogelijke uitschieters die de analyse kunnen verstoren.

Methoden:
✅ IQR-methode
De functie thresholds_outlier() berekent grenzen op basis van de interkwartielafstand.
Waarden buiten deze grenzen worden als uitbijters beschouwd.

✅ Percentielmethode
Omdat de IQR-methode soms te strenge grenzen geeft, gebruiken we ook percentielen (standaard 4e en 96e) om realistischere grenzen te bepalen.

De functie plot_all_numerical_with_outliers() maakt voor elke numerieke variabele:

Een histogram

Een boxplot
inclusief de gekozen percentielgrenzen en het aantal uitschieters.

🧽 Outlier Detectie & Behandeling
outlier_check() controleert of een kolom uitbijters bevat.

outliers_grab() toont de daadwerkelijke outlier-rijen.

replace_as_thresholds() vervangt extreme waarden door de grenswaarden om de impact van uitschieters te beperken zonder ze te verwijderen.

🤖 Geavanceerde Outlier Detectie: Local Outlier Factor (LOF)
LOF identificeert punten die zich afwijkend gedragen ten opzichte van hun buren.

LOF-scores worden gesorteerd en geplot om een drempelwaarde te kiezen.

Observaties met een LOF-score lager dan de drempel worden als outliers beschouwd.

Deze outliers worden verwijderd voor een opgeschoonde dataset (df_clean).

📈 Resultaten
Originele dataset grootte: aantal rijen vóór opschonen.

Aantal geïdentificeerde outliers.

Dataset grootte na verwijderen outliers.

🎯 Samenvatting van de doelen:
✔️ Data inladen en verkennen: inzicht krijgen in structuur en variabelen.
✔️ Variabeltypen bepalen: categorisch, numeriek, kardinaal.
✔️ Doelvariabele-analyse: gemiddelden per categorie interpreteren.
✔️ Outlierdetectie: zowel IQR- als percentielmethode en LOF gebruiken.
✔️ Visualisatie: histogram & boxplot per numerieke variabele om uitschieters visueel te beoordelen.
✔️ Outlierbehandeling: extreme waarden afkappen of verwijderen via LOF.
✔️ Dataset opschonen: outliers verwijderen voor robuustere analyses.

📂 Dit project helpt bij het uitvoeren van een grondige en visuele analyse van de dataset en biedt technieken om de kwaliteit van de data te verbeteren voor verdere modellering.
