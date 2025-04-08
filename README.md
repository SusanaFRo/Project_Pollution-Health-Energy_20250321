1. BESCHREIBUNG
   
Das Projekt zielt darauf ab, den Zusammenhang zwischen der Energieerzeugungsmatrix (aufgeschlüsselt nach Quellen wie fossile Brennstoffe, erneuerbare Energien, Kernenergie etc.) und der Luftqualität zu untersuchen – und wie diese beiden Faktoren die Inzidenz von schweren Krankheiten (respiratorische Erkrankungen, Herz-Kreislauf-Erkrankungen sowie Lungenkrebs) beeinflussen. Dabei werden Daten von 30 Ländern aus dem Zeitraum 2010 bis 2019 integriert, um sowohl zeitliche als auch länderspezifische Unterschiede zu analysieren.


3. ZIEL
   
Das Hauptziel besteht darin, zu evaluieren, wie die Zusammensetzung der Energieerzeugung (z. B. hoher Anteil fossiler Brennstoffe gegenüber erneuerbaren oder Kernenergie) die Luftqualität beeinflusst und in der Folge die Inzidenz schwerwiegender Krankheiten beeinflusst.


5. METHODIK
   
a. Datenvorverarbeitung und Bereinigung:

Durchschnittswerte für alle Städte und Jahre: Berechnung der Durchschnittswerte, um Daten nach Land und Jahr zu erhalten.

Umgang mit fehlenden Werten: Fehlende Werte werden durch den Mittelwert der jeweiligen Spalte ersetzt.

b. Explorative Datenanalyse (EDA):

Visualisierung von zeitlichen Trends: Darstellung der Entwicklung der Luftverschmutzung, Erkrankungen und Energieproduktion über die Jahre hinweg.

c. Statistische Modellierung und Machine Learning:

Mit Hilfe des maschinellen Lernens wurden Prognosen für die Daten erstellt, um drei Fragen zu beantworten. Es wurden zwei Modelle bewertet: LinearRegression und RandomForestRegressor. Das Modell mit den besten Kennzahlen (MSE, RMSE, R²) wurde für jede Prognose ausgewählt. Weitere Einzelheiten zur Methodik des maschinellen Lernens finden sich im Abschnitt „Prognosen“.

d. Präsentation:

Die Präsentation der Ergebnisse erfolgt mit Streamlit.






1. DESCRIPTION
   
The project aims to analyse the relationship between the energy production matrix (broken down by sources such as fossil fuels, renewable energy, nuclear energy, etc.) and air quality - and how these two factors influence the incidence of serious diseases (respiratory diseases, cardiovascular diseases and lung cancer). Data from 30 countries from the period 2010 to 2019 will be integrated in order to analyse both temporal and country-specific differences.


3. TARGET

The main objective is to evaluate how the composition of energy production (e.g. high share of fossil fuels versus renewable or nuclear energy) affects air quality and subsequently influences the incidence of serious diseases.


5. METHODOLOGY
   
a. Data pre-processing and adjustment:

Average values for all cities and years: calculation of average values to obtain data by country and year.
Dealing with missing values: Missing values are replaced by the mean value of the respective column.

b. Exploratory data analysis (EDA):

Visualisation of trends over time: presentation of the evolution of air pollution, diseases and energy production over the years.

c. Statistical modelling and machine learning:

Machine learning was used to make predictions for the data to answer three questions. Two models were evaluated: linear regression and random forest regressor. The model with the best metrics (MSE, RMSE, R²) was selected for each prediction. Further details on the machine learning methodology can be found in the ‘Forecasts’ section.

d. Presentation: 

The presentation of the results is done using Streamlit.
