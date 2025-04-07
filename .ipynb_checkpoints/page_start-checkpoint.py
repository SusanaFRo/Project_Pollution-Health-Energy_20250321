import streamlit as st


def app():
    st.subheader("1. Beschreibung")
    st.write("""
        Das Projekt zielt darauf ab, den Zusammenhang zwischen der Energieerzeugungsmatrix (aufgeschlüsselt nach Quellen wie fossile Brennstoffe, erneuerbare Energien, Kernenergie etc.) und der Luftqualität zu untersuchen – und wie diese beiden Faktoren die Inzidenz von schweren Krankheiten (respiratorische Erkrankungen, Herz-Kreislauf-Erkrankungen sowie Lungenkrebs) beeinflussen. Dabei werden Daten von 30 Ländern aus dem Zeitraum 2010 bis 2019 integriert, um sowohl zeitliche als auch länderspezifische Unterschiede zu analysieren.
    """)
    st.image("picture.jpg", caption="Energiewende in Europa")
   
  
    st.markdown("""
        ### 2. Ziel
        Das Hauptziel besteht darin, zu evaluieren, wie die Zusammensetzung der Energieerzeugung (z. B. hoher Anteil fossiler Brennstoffe gegenüber erneuerbaren oder Kernenergie) die Luftqualität beeinflusst und in der Folge die Inzidenz schwerwiegender Krankheiten beeinflusst.
        ### 3. Projektmethodik

        #### a. Datenvorverarbeitung und Bereinigung:
        - **Durchschnittswerte für alle Städte und Jahre**: Berechnung der Durchschnittswerte, um Daten nach Land und Jahr zu erhalten.
        - **Umgang mit fehlenden Werten**: Fehlende Werte werden durch den Mittelwert der jeweiligen Spalte ersetzt.

        #### b. Explorative Datenanalyse (EDA):
        - **Visualisierung von zeitlichen Trends**: Darstellung der Entwicklung der Luftverschmutzung, Erkrankungen und Energieproduktion über die Jahre hinweg.

        #### c. Statistische Modellierung und Machine Learning:

    """)
    st.write("""
    Mit Hilfe des maschinellen Lernens wurden Prognosen für die Daten erstellt, um drei Fragen zu beantworten. 
    Es wurden zwei Modelle bewertet: **LinearRegression** und **RandomForestRegressor**. 
    Zur Beantwortung jeder Frage wurde das Modell mit den besten Metriken (MSE, RMSE, R²) ausgewählt. 
    Die Struktur der in Python geschriebenen Programme für maschinelles Lernen ist für die Beantwortung jeder Frage die gleiche:
    """)

    st.markdown("""
    - **Importieren von Bibliotheken**
    - **Laden und Vorverarbeitung der Daten**
    - **Definition von X (Features) und Y (Target)**
    - **Vorverarbeitung:**
    - Imputation fehlender Werte mit `SimpleImputer`
    - Kodierung kategorischer Variablen mit `OneHotEncoder`
    - Skalierung mit `StandardScaler`
    - **Aufteilung in Trainings- und Testdaten**
    - **Training des Modells**
    - **Simulation von Szenarien** (das aktuellste Jahr (2019) wird als Vergleichsbasis herausgefiltert)
    - **Vorhersage und Visualisierung** mit `Plotly`
    - **Bewertung des Modells** anhand der Metriken MSE, RMSE und R² auf Testdaten
    """)