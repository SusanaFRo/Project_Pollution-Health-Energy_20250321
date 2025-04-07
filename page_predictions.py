# 1. Importieren von Bibliotheken
# =======================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

def app():

# Abschnitt 1
    st.header("1. Ziele: Durchzuführende Prognosen")
    st.write("""
Machine Learning wurde verwendet, um die folgenden Fragen zu beantworten:

1. Prognose der **gesamten Schadstoffemissionen** zur Reduzierung umweltbedingter Todesfälle um 0% bis 95% (Basisjahr 2019)
2. Prognose des **Jahres** zur Reduzierung umweltbedingter Todesfälle um 0% bis 95% (Basisjahr 2019)
3. Prognose der **Erzeugung erneuerbarer Energien** zur Reduzierung gesamten Schadstoffemissionen (PM2,5 + PM10 + NO₂) gemäß der WHO-Empfehlung (PM2,5: 5 µg/m³, PM10: 15 µg/m³, NO₂: 10 µg/m³ (jährlich)).
""")
    
# Abschnitt 2   
    st.header("2. Machine Learning Modelle")
    st.write("""
Es wurden zwei Modelle bewertet: LinearRegression und RandomForestRegressor.

1. **LinearRegression Modell**: Geht von einer linearen Beziehung zwischen den Eingangsvariablen (X) und der Zielvariablen (Y) aus. Wird verwendet, wenn wenige Daten vorhanden sind und Interpretierbarkeit gewünscht wird.
2. **RandomForestRegressor Modell**: Erzeugt mehrere Bäume und mittelt deren Vorhersagen, was die Genauigkeit verbessert und Überanpassung reduziert. Wird verwendet, wenn die Beziehung zwischen den Variablen nicht linear oder komplex ist, Interaktionen zwischen Variablen bestehen und genauere Vorhersagen benötigt werden (reduziert die Varianz).
""")
    
    st.subheader("2.1. Auswahl des Modells")
    st.write("""
Zur Bewertung der Modelle wurden die Metriken (MSE, RMSE, R²) verglichen. Im Allgemeinen hat ein Modell eine bessere Leistung bei niedrigen MSE-, RMSE-Werten und einem R² nahe 1. RMSE gibt den Fehler an, den das Modell im Durchschnitt macht, und R² zeigt, wie gut das Modell die Variabilität der Daten erklärt.
""")

    data = {
    "FRAGE": [1, 1, 2, 2, 3, 3],
    "MODELL": ["LinearRegression", "RandomForestRegressor", "LinearRegression", "RandomForestRegressor", "LinearRegression", "RandomForestRegressor"],
    "MSE": [21.84, 22.48, 1.77, 5.21, 93.31, 49.1],
    "RMSE": [4.67, 4.74, 1.33, 2.2, 9.66, 7.0],
    "R2": [0.87, 0.86, 0.77, 0.31, 0.96, 0.98]
}

    df = pd.DataFrame(data)

# Zu markierende Zeilen (1, 3 und 6 → Index 0, 2, 5)
    highlight_indices = [0, 2, 5]

# Funktion zur Umwandlung von DataFrame in HTML mit schwarzem Text und Farben
    def dataframe_to_html(df, highlight_indices):
        html = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f4f4f4;
            color: black;  /* Color negro para el texto del encabezado */
        }
        .highlight {
            background-color: rgba(255, 255, 150, 0.5);  /* Amarillo claro con transparencia */
            color: black;  /* Texto negro */
        }
    </style>
    <table>
    """
    
    # Überschriften
        html += "<tr>" + "".join([f"<th>{col}</th>" for col in df.columns]) + "</tr>"
    
    # Filas
        for i, row in df.iterrows():
            row_class = "highlight" if i in highlight_indices else ""
            html += f"<tr class='{row_class}'>" + "".join(
            [f"<td>{row[col]:.2f}</td>" if isinstance(row[col], float) else f"<td>{row[col]}</td>" for col in df.columns]
        ) + "</tr>"
    
        html += "</table>"
        return html

# DataFrame in HTML umwandeln und in Streamlit anzeigen
    html_table = dataframe_to_html(df, highlight_indices)
    st.markdown(html_table, unsafe_allow_html=True)




# Abschnitt 3
    st.header("3. Auswahl der Variablen (und Target, X Features)")
    st.markdown("""
- **y (Target)** wurde als die Zielvorhersage ausgewählt, um jede Frage zu beantworten.  
-	**X (Features)**. Die Auswahl der Merkmale ist entscheidend, um die Genauigkeit zu verbessern, Überanpassung zu reduzieren und die Effizienz eines Modells zu optimieren. X wurde unter Berücksichtigung der Korrelationen in der Korrelationsmatrix, der Feature-Importance-Analyse (RandomForest)/der Koeffizientenanalyse (LinearRegression) und der Bewertung der Auswirkungen auf die Metriken ausgewählt.
""")

# Abschnitt 4
    st.header("4. Struktur des Machine-Learning-Programms")
    st.markdown("""
    Ein Python-Programm wird geschrieben, um jede Prognose zu erhalten. Die Struktur des Programms ist für alle Prognosen die gleiche:
    1. **Importieren von Bibliotheken**
    2. **Laden und Vorverarbeitung der Daten**
    3. **Definition von X (Features) und Y (Target)**
    4. **Vorverarbeitung**: Imputation, Kodierung und Skalierung (Kodierung kategorischer Variablen mit OneHotEncoder, Imputation fehlender Werte mit SimpleImputer, Skalierung mit StandardScaler).
    5. **Aufteilung in Trainings- und Testdaten.**
    6. **Training des Modells.**
    7. **Simulation von Szenarien** (das aktuellste Jahr (2019) wird als Vergleichsbasis herausgefiltert).
    8. **Vorhersage und Visualisierung**: Plotly.
    9. **Bewertung des Modells**: MSE, RMSE, R² auf Testdaten.
    """)
    
    
# Abschnitt 5
    st.header("5. Ergebnisse")
    st.subheader("5.1. Korrelationsmatrix numerischer Variablen")

    st.markdown("""
    Die Korrelationsmatrix zeigt die folgenden Korrelationen:
- **Hohe Korrelation zwischen Schadstoffen**: PM2.5 und PM10 (0,78)
- **Hohe Korrelation zwischen Umweltverschmutzung und Gesundheit (Sterblichkeit)**: PM2.5, PM10 und Todesfälle durch Erkrankungen des Kreislaufsystems, Atemwegserkrankungen und Atemwegskrebs
- **Energiequellen und Umweltverschmutzung**:
  - *Fossile Energie (Fossil_energie) und Gesamtschmutzung (0,15)*: Schwache, aber positive Beziehung, die darauf hindeutet, dass mehr fossile Energieerzeugung zur Umweltverschmutzung beitragen könnte.
  - *Erneuerbare Energie (Ern_energie) und Gesamtschmutzung (-0,22)*: Mäßig negative Beziehung, was darauf hindeutet, dass mehr erneuerbare Energie mit weniger Umweltverschmutzung verbunden sein könnte.
- **Erneuerbare Energie und Sterblichkeit**: Höhere Produktion von erneuerbarer Energie führt zu einer niedrigeren Sterblichkeitsrate durch umweltverschmutzungsbedingte Krankheiten. Dies verstärkt die Hypothese, dass der Übergang zu erneuerbarer Energie vorzeitige Todesfälle reduzieren könnte.
- **Zeitliche Entwicklung**:
  - *Jahr vs. PM2.5 (-0,31), PM10 (-0,30), NO₂ (-0,31), Gesamtschmutzung (-0,38)*: Deutet auf einen Rückgang der Verschmutzung im Laufe der Zeit hin.
  - *Jahr vs. Erneuerbare Energie (0,10)*: Leicht steigender Trend in der Produktion erneuerbarer Energie.
  - *Jahr vs. Sterblichkeit durch Krankheiten (-0,24 bis 0,00)*: Unklare Trends, könnte darauf hinweisen, dass neben der Umweltverschmutzung auch andere Faktoren die Sterblichkeit beeinflussen.
""")
    
    
    # Load and preprocess data
    df = pd.read_csv('daten_ML_20250313.csv', sep=';', decimal='.', engine='python')

    # Replace commas with dots and convert to numeric
    df = df.replace({',': '.'}, regex=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

    # Rename columns
    df.rename(columns={
        'PM2,5 (ug/m3)': 'PM2_5',
        'PM10 (ug/m3)': 'PM10',
        'NO2 (ug/m3)': 'NO2',
        'Gesamtschadstoffe (ug/m3)': 'Gesamtschadstoffe',
        'Erkrankungen des Kreislaufsystems (Todesfalle pro 100.000)': 'Kreislaufsystem_Erk_Tod',
        'Erkrankungen der Atemwege (Todesfalle pro 100.000)': 'Atemwege_Erk_Tod',
        'Krebs der Atemwege (Todesfalle pro 100.000)': 'Krebs_Atemwege_Tod',
        'Todesfalle insgesamt (pro 100.000)': 'Tod_insgesamt',
        'Fossiler_Energieerzeugung (TWh)': 'Fossil_energie',
        'nukleare_Energieerzeugung (TWh)': 'Nuklear_energie',
        'Erneuerbare_Energieerzeugung (TWh)': 'Ern_energie'
    }, inplace=True)

    df.drop(['ISO3', 'Erneuerbare + nukleare_Energieerzeugung (TWh)'], axis=1, inplace=True)

    # Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('')

    # Display plot in Streamlit
    st.pyplot(plt)

    st.markdown("---")

    st.subheader("5.2. Prognose der gesamten Schadstoffemissionen zur Reduzierung umweltbedingter Todesfälle um 0% bis 95% (Basisjahr 2019)")

    st.markdown("""
Im folgenden Diagramm ist zu sehen, dass zur Reduzierung der Mortalität durch umweltverschmutzungsbedingte Krankheiten (im Vergleich zum neuesten Jahr der Daten: 2019) die Verschmutzung verringert werden muss. Einige Länder erreichen eine Reduktion von 95 % der Todesfälle bei einer höheren Vorhersage der Verschmutzung als die von der WHO empfohlene Grenze (30 µg/m³, Summe der WHO-Empfehlungen für PM2,5, PM10 und NO₂). Länder mit weniger Todesfällen erreichen das Ziel der 95 % Reduktion bei einer höheren Verschmutzung.
""")
    # =======================
    # 2. Laden und Vorverarbeitung der Daten
    # =======================
    df = pd.read_csv('daten_ML_20250313.csv', sep=';', decimal='.', engine='python')

    df = df.replace({',': '.'}, regex=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

    df.rename(columns={
        'PM2,5 (ug/m3)': 'PM2_5', 
        'PM10 (ug/m3)': 'PM10', 
        'NO2 (ug/m3)': 'NO2',
        'Gesamtschadstoffe (ug/m3)': 'Gesamtschadstoffe',
        'Erkrankungen des Kreislaufsystems (Todesfalle pro 100.000)': 'Kreislaufsystem_Erk',
        'Erkrankungen der Atemwege (Todesfalle pro 100.000)': 'Atemwege_Erk',
        'Krebs der Atemwege (Todesfalle pro 100.000)': 'Krebs_Atemwege_Erk',
        'Todesfalle insgesamt (pro 100.000)': 'Tod_insgesamt',
        'Fossiler_Energieerzeugung (TWh)': 'Fossil_energie',
        'nukleare_Energieerzeugung (TWh)': 'Nukl_energie',
        'Erneuerbare_Energieerzeugung (TWh)': 'Ern_energie',
        'Erneuerbare + nukleare_Energieerzeugung (TWh)': 'Ern_nukl_energie_TWh'
    }, inplace=True)

    df.drop(['ISO3', 'Nukl_energie_TWh', 'Ern_nukl_energie_TWh',
            'Kreislaufsystem_Erk_Tod', 'Atemwege_Erk_Tod', 'Krebs_Atemwege_Tod',
            'PM2_5', 'PM10', 'NO2',
            'Fossil_energie_TWh'],
            axis=1, inplace=True, errors='ignore')

    # =======================
    # 3. Definition von X (Features) und Y (Target)
    # =======================
    X, y = df.drop(columns=['Gesamtschadstoffe']), df['Gesamtschadstoffe']

    # =======================
    # 4. Vorverarbeitung: Imputation, Kodierung und Skalierung
    # =======================
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    if cat_features:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        X_encoded = pd.DataFrame(encoder.fit_transform(X[cat_features]), 
                                columns=encoder.get_feature_names_out(cat_features))
        X = X.drop(columns=cat_features).reset_index(drop=True)
        X = pd.concat([X, X_encoded], axis=1)

    scaler, imputer = StandardScaler(), SimpleImputer(strategy='mean')
    X = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X)), columns=X.columns)

    # =======================
    # 5. Aufteilung in Trainings- und Testdaten
    # =======================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # =======================
    # 6. Training des Modells (Lineare Regression)
    # =======================
    model = LinearRegression()
    model.fit(X_train, y_train)

    # =======================
    # 8. Szenarien und Vorhersagen
    # =======================
    baseline_df = df[df['Jahr'] == 2019].copy()

    reductions = [0, 10, 20, 50, 70, 95]
    sim_list = []

    for _, row in baseline_df.iterrows():
        for red in reductions:
            new_row = row.copy()
            new_row['Tod_insgesamt'] = new_row['Tod_insgesamt'] * (1 - red / 100)
            new_row['Reduction'] = red
            sim_list.append(new_row)

    sim_df = pd.DataFrame(sim_list)

    sim_df_encoded = sim_df.copy()
    if cat_features:
        encoded_countries = pd.DataFrame(encoder.transform(sim_df_encoded[cat_features]), 
                                        columns=encoder.get_feature_names_out(cat_features))
        sim_df_encoded = sim_df_encoded.drop(columns=cat_features).reset_index(drop=True)
        sim_df_encoded = pd.concat([sim_df_encoded, encoded_countries], axis=1)

    X_new = sim_df_encoded[X.columns]

    X_new_transformed = imputer.transform(X_new)
    X_new_scaled = scaler.transform(X_new_transformed)
    X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=X.columns)

    sim_df['Vorhergesagte_Gesamtschadstoffe'] = model.predict(X_new_scaled_df)

    # =======================
    # 8. Visualisierung der Vorhersagegrafik
    # =======================
    order = sim_df[sim_df['Reduction'] == 0].sort_values(by='Vorhergesagte_Gesamtschadstoffe', ascending=False)['Land'].tolist()

    fig = px.bar(sim_df.sort_values(by='Vorhergesagte_Gesamtschadstoffe', ascending=False),
                x='Land', y='Vorhergesagte_Gesamtschadstoffe',
                color='Reduction',
                color_continuous_scale='RdYlGn',
                title=" ",
                labels={'Land': 'Land', 'Vorhergesagte_Gesamtschadstoffe': 'Vorhersage (µg/m³)', 'Reduction': 'Todesfallreduktion (%)'},
                barmode='overlay', opacity=0.7, width=900, height=600,
                category_orders={'Land': order},
                custom_data=['Reduction', 'Tod_insgesamt', 'Gesamtschadstoffe'])

    fig.update_coloraxes(colorbar_tickvals=[0, 10, 20, 50, 70, 95],
                        colorbar_ticktext=[f"{v}%" for v in [0, 10, 20, 50, 70, 95]])
    fig.update_traces(
        hovertemplate='<b>Land:</b> %{x}<br>' +
                    '<b>Vorhersage der Gesamtschadstoffe (µg/m³):</b> %{y:.1f}<br>' +
                    '<b>Todesfälle (%{customdata[0]}% Reduktion):</b> %{customdata[1]:.0f}<br>' +
                    '<b>Ursprüngliche Schadstoffe (µg/m³):</b> %{customdata[2]:.1f}<extra></extra>'
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title='Prognose der gesamten Schadstoffemissionen (µg/m³)',
        xaxis=dict(tickangle=45),
        legend_title='Todesfallreduktion (%)',
        legend=dict(x=0.5, xanchor='center', y=1.05, yanchor='bottom'),
        title=dict(x=0.5, xanchor='center', y=0.9))

    fig.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=30,
        y1=30,
        line=dict(color="red", width=2)
    )

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color='red', width=2),
        name='WHO-Empfehlung: gesamten Schadstoffemissionen = 30 µg/m³',
        hovertemplate='<b>WHO-Empfehlung:</b> gesamten Schadstoffemissionen = 30 µg/m³'
    ))

    fig.update_layout(
        legend=dict(
            x=1, y=1,
            xanchor='right',
            yanchor='top'
        ),
        showlegend=True,
        legend_title_text=None
    )

    # Display Plotly figure in Streamlit
    st.plotly_chart(fig)

    # =======================
    # 9. Vorhersagen und Bewertung
    # =======================
    y_pred = model.predict(X_test)
    st.write(f"***Modellbewertung (LinearRegression):** MSE: {mean_squared_error(y_test, y_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}, R²: {r2_score(y_test, y_pred):.2f}*")




    st.markdown("---")

    st.subheader("5.3. Prognose des Jahres zur Reduzierung umweltbedingter Todesfälle um 0% bis 95% (Basisjahr 2019)")

    st.markdown("""
Im folgenden Diagramm ist zu sehen, dass Länder mit mehr Todesfällen mehr Jahre benötigen, um eine Reduktion der Todesfälle (im Vergleich zu 2019) zu erreichen, was zu erwarten ist. Wenn die Trends der verfügbaren Daten fortgesetzt werden, ist das Land, das am längsten brauchen würde, um eine Reduktion von 95 % zu erreichen, Bulgarien (Jahr 2059), während Spanien und Frankreich die Länder sind, die am wenigsten Zeit benötigen, um dieses Ziel zu erreichen (Jahr 2025).
""")

    # =======================
    # 2. Laden und Vorverarbeitung der Daten
    # =======================
    df = pd.read_csv('daten_ML_20250313.csv', sep=';', decimal='.', engine='python')

    df = df.replace({',': '.'}, regex=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

    df.rename(columns={
        'PM2,5 (ug/m3)': 'PM2_5', 
        'PM10 (ug/m3)': 'PM10', 
        'NO2 (ug/m3)': 'NO2',
        'Gesamtschadstoffe (ug/m3)': 'Gesamtschadstoffe',
        'Erkrankungen des Kreislaufsystems (Todesfalle pro 100.000)': 'Kreislaufsystem_Erk_Tod',
        'Erkrankungen der Atemwege (Todesfalle pro 100.000)': 'Atemwege_Erk_Tod',
        'Krebs der Atemwege (Todesfalle pro 100.000)': 'Krebs_Atemwege_Tod',
        'Todesfalle insgesamt (pro 100.000)': 'Tod_insgesamt',
        'Fossiler_Energieerzeugung (TWh)': 'Fossil_energie_TWh',
        'nukleare_Energieerzeugung (TWh)': 'Nukl_energie_TWh',
        'Erneuerbare_Energieerzeugung (TWh)': 'Ern_energie_TWh',
        'Erneuerbare + nukleare_Energieerzeugung (TWh)': 'Ern_nukl_energie_TWh'
    }, inplace=True)

    df.drop(['ISO3', 'Nukl_energie_TWh', 'Ern_nukl_energie_TWh',
            'Kreislaufsystem_Erk_Tod', 'Atemwege_Erk_Tod', 'Krebs_Atemwege_Tod',
            'PM2_5', 'PM10', 'NO2',
            'Fossil_energie_TWh'],
            axis=1, inplace=True, errors='ignore')

    # =======================
    # 3. Definition von X (Features) und Y (Target)
    # =======================
    X, y = df.drop(columns=['Jahr']), df['Jahr']

    # =======================
    # 4. Vorverarbeitung: Imputation, Kodierung und Skalierung
    # =======================
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    if cat_features:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        X_encoded = pd.DataFrame(encoder.fit_transform(X[cat_features]), 
                                columns=encoder.get_feature_names_out(cat_features))
        X = X.drop(columns=cat_features).reset_index(drop=True)
        X = pd.concat([X, X_encoded], axis=1)

    scaler, imputer = StandardScaler(), SimpleImputer(strategy='mean')
    X = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X)), columns=X.columns)

    # =======================
    # 5. Aufteilung in Trainings- und Testdaten
    # =======================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # =======================
    # 6. Training des Modells (Lineare Regression)
    # =======================
    model = LinearRegression()
    model.fit(X_train, y_train)

    # =======================
    # 8. Szenarien und Vorhersagen
    # =======================
    baseline_df = df[df['Jahr'] == 2019].copy()

    reductions = [0, 10, 20, 50, 70, 95]
    sim_list = []

    for _, row in baseline_df.iterrows():
        for red in reductions:
            new_row = row.copy()
            new_row['Tod_insgesamt'] = new_row['Tod_insgesamt'] * (1 - red / 100)
            new_row['Reduction'] = red
            sim_list.append(new_row)

    sim_df = pd.DataFrame(sim_list)

    sim_df_encoded = sim_df.copy()
    if cat_features:
        encoded_countries = pd.DataFrame(encoder.transform(sim_df_encoded[cat_features]), 
                                        columns=encoder.get_feature_names_out(cat_features))
        sim_df_encoded = sim_df_encoded.drop(columns=cat_features).reset_index(drop=True)
        sim_df_encoded = pd.concat([sim_df_encoded, encoded_countries], axis=1)

    X_new = sim_df_encoded[X.columns]

    X_new_transformed = imputer.transform(X_new)
    X_new_scaled = scaler.transform(X_new_transformed)
    X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=X.columns)

    sim_df['Predicted_Year'] = model.predict(X_new_scaled_df).round().astype(int)

    # =======================
    # 8. Visualisierung der Vorhersagegrafik
    # =======================
    fig = px.bar(sim_df.sort_values(by='Predicted_Year', ascending=False),
                x='Land', y='Predicted_Year',
                color='Reduction',
                title=" ",
                labels={'Land': 'Land', 'Predicted_Year': 'Prognose des Jahres', 'Reduction': 'Todesfallreduktion (%)'},
                barmode='overlay', opacity=0.7, color_continuous_scale='RdYlGn',
                width=900, height=600,
                category_orders={'Land': sim_df[sim_df['Reduction'] == 95]
                                .sort_values(by='Predicted_Year', ascending=False)['Land']
                                .tolist(),
                                'Reduction': [95, 70, 50, 20, 10, 0]},
                custom_data=['Reduction', 'Tod_insgesamt'])

    fig.update_coloraxes(colorbar_tickvals=[0, 10, 20, 50, 70, 95],
                        colorbar_ticktext=[f"{v}%" for v in [0, 10, 20, 50, 70, 95]])
    fig.update_traces(
        hovertemplate='<b>Land:</b> %{x}<br>' +
                    '<b>Prognose des Jahres:</b> %{y}<br>' +
                    '<b>Todesfälle (</b>%{customdata[0]}% reduktion):</b> %{customdata[1]:.0f}<extra></extra>')
    fig.update_layout(yaxis=dict(range=[2000, sim_df['Predicted_Year'].max()+5]),
                    xaxis=dict(tickangle=45),
                    xaxis_title=None,
                    legend_title='Todesfallreduktion (%)',
                    title=dict(x=0.5, xanchor='center', y=0.9),
                    legend=dict(x=0.5, xanchor='center', y=1.05, yanchor='bottom'))

    # Display Plotly figure in Streamlit
    st.plotly_chart(fig)

    # =======================
    # 9. Vorhersagen und Bewertung
    # =======================
    y_pred = model.predict(X_test)
    st.write(f"***Modellbewertung (LinearRegression):** MSE: {mean_squared_error(y_test, y_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}, R²: {r2_score(y_test, y_pred):.2f}*")



    st.markdown("---")

    st.subheader("5.4. Prognose der Erzeugung erneuerbarer Energien zur Reduzierung gesamten Schadstoffemissionen (PM2,5 + PM10 + NO₂) gemäß der WHO-Empfehlung")
    st.markdown("""
    Im folgenden Diagramm wird die Prognose für die Produktion von erneuerbaren Energien dargestellt, die erforderlich ist, um das Ziel der WHO zur Reduzierung der Verschmutzung auf 30 µg/m³ zu erreichen (Summe der WHO-Empfehlungen für PM2,5, PM10 und NO₂). Das Land, das am meisten erneuerbare Energie benötigt, um das Verschmutzungsziel der WHO zu erreichen, ist Deutschland, gefolgt von Norwegen und Italien. Die Länder, die am wenigsten erneuerbare Energie benötigen, um dieses Ziel zu erreichen, sind Ungarn und Lettland. Es ist zu beobachten, dass die Länder, die am meisten erneuerbare Energie produzieren müssen, um die Verschmutzung zu reduzieren, die Länder sind, die derzeit (im Jahr 2019) am meisten erneuerbare Energie produzieren. Dies könnte darauf zurückzuführen sein, dass diese Länder bereits von einer erheblichen Reduktion der Verschmutzung durch den Einsatz erneuerbarer Energien profitieren.
    """)
    
    # =======================
    # 2. Laden und Vorverarbeitung der Daten
    # =======================
    df = pd.read_csv('daten_ML_20250313.csv', sep=';', decimal='.', engine='python')

    df = df.replace({',': '.'}, regex=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

    df.rename(columns={
        'PM2,5 (ug/m3)': 'PM2_5',
        'PM10 (ug/m3)': 'PM10',
        'NO2 (ug/m3)': 'NO2',
        'Gesamtschadstoffe (ug/m3)': 'Gesamtschadstoffe',
        'Erkrankungen des Kreislaufsystems (Todesfalle pro 100.000)': 'Kreislaufsystem_Erk_Tod',
        'Erkrankungen der Atemwege (Todesfalle pro 100.000)': 'Atemwege_Erk_Tod',
        'Krebs der Atemwege (Todesfalle pro 100.000)': 'Krebs_Atemwege_Tod',
        'Todesfalle insgesamt (pro 100.000)': 'Tod_insgesamt',
        'Fossiler_Energieerzeugung (TWh)': 'Fossil_energie_TWh',
        'nukleare_Energieerzeugung (TWh)': 'Nukl_energie_TWh',
        'Erneuerbare_Energieerzeugung (TWh)': 'Ern_energie_TWh',
        'Erneuerbare + nukleare_Energieerzeugung (TWh)': 'Ern_nukl_energie_TWh'
    }, inplace=True)

    df.drop(['ISO3', 'Nukl_energie_TWh', 'Ern_nukl_energie_TWh',
            'Kreislaufsystem_Erk_Tod','Atemwege_Erk_Tod', 'Krebs_Atemwege_Tod',
            'PM2_5', 'PM10', 'NO2'],
            axis=1, inplace=True, errors='ignore')

    # =======================
    # 3. Definition von X (Features) und Y (Target)
    # =======================
    target_col = 'Ern_energie_TWh'
    if target_col not in df.columns:
        raise ValueError(f"Die Zielspalte '{target_col}' ist im DataFrame nicht vorhanden.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # =======================
    # 4. Vorverarbeitung: Imputation, Kodierung und Skalierung
    # =======================
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    if cat_cols:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        X_encoded = pd.DataFrame(encoder.fit_transform(X[cat_cols]),
                                columns=encoder.get_feature_names_out(cat_cols),
                                index=X.index)
        X = X.drop(columns=cat_cols).reset_index(drop=True)
        X = pd.concat([X, X_encoded], axis=1)
    else:
        encoder = None

    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns, index=X.index)
    X = X_scaled.copy()

    # =======================
    # 5. Aufteilung in Trainings- und Testdaten
    # =======================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # =======================
    # 6. Training des Modells (RandomForestRegressor)
    # =======================
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # =======================
    # 7. Szenarien und Vorhersagen
    # =======================
    baseline_df = df[df['Jahr'] == 2019].copy()
    who_contamination = 30
    sim_list = []

    for country in baseline_df['Land'].unique():
        base_country = baseline_df[baseline_df['Land'] == country].iloc[0]
        original_renewable = base_country['Ern_energie_TWh']
        
        new_row = base_country.copy()
        new_row['Gesamtschadstoffe'] = who_contamination
        new_row['Reduction'] = 0
        
        sim_df_temp = pd.DataFrame([new_row])
        if cat_cols and encoder is not None:
            sim_cat = pd.DataFrame(encoder.transform(sim_df_temp[cat_cols]),
                                columns=encoder.get_feature_names_out(cat_cols),
                                index=sim_df_temp.index)
            sim_df_temp = sim_df_temp.drop(columns=cat_cols)
            sim_df_temp = pd.concat([sim_df_temp, sim_cat], axis=1)
        if target_col in sim_df_temp.columns:
            sim_df_temp = sim_df_temp.drop(columns=[target_col])
        sim_df_temp = sim_df_temp.reindex(columns=X.columns, fill_value=0)
        sim_df_temp = pd.DataFrame(imputer.transform(sim_df_temp), columns=X.columns, index=sim_df_temp.index)
        sim_df_temp = pd.DataFrame(scaler.transform(sim_df_temp), columns=X.columns, index=sim_df_temp.index)

        predicted_renew = model.predict(sim_df_temp)[0]
        
        sim_list.append({
            'Land': country,
            'Gesamtschadstoffe (µg/m³)': who_contamination,
            'Predicted_Renew_TWh': predicted_renew,
        })

    sim_df = pd.merge(pd.DataFrame(sim_list), baseline_df[['Land', 'Gesamtschadstoffe', 'Ern_energie_TWh']], on='Land', how='left')

    # =======================
    # 8. Visualisierung der Vorhersagegrafik
    # =======================
    total_pred = sim_df.groupby('Land')['Predicted_Renew_TWh'].sum().sort_values(ascending=False)
    ordered_countries = total_pred.index.tolist()

    fig = px.bar(sim_df,
                x='Land', y='Predicted_Renew_TWh',
                color_discrete_sequence=['green'],
                title=" ",
                labels={'Land': 'Land', 
                        'Predicted_Renew_TWh': 'Prognose der erneuerbaren Energieerzeugung (TWh)', 
                        'Gesamtschadstoffe (µg/m³)': 'Gesamtschadstoffe (µg/m³)'},
                barmode='group', opacity=0.7, color_continuous_scale='RdYlGn_r',
                width=900, height=600,
                category_orders={'Land': ordered_countries},
                custom_data=['Gesamtschadstoffe (µg/m³)', 'Gesamtschadstoffe', 'Ern_energie_TWh'])

    fig.update_traces(
        hovertemplate='<b>Land:</b> %{x}<br>' +
                    '<b>Prognose der erneuerbaren Energieerzeugung:</b> %{y:.1f} TWh<br>' +
                    '<b>Gesamtschadstoffe (WHO-Empfehlung):</b> %{customdata[0]:.1f} µg/m³<br>' +
                    '<b>Erneuerbare Energieerzeugung (2019):</b> %{customdata[2]:.1f} TWh<br>'+
                    '<b>Ursprüngliche Gesamtschadstoffe (2019):</b> %{customdata[1]:.1f} µg/m³<br><extra></extra>',
        hoverlabel=dict(
            bgcolor="yellow",
            font=dict(color="black")))

    fig.update_layout(xaxis=dict(tickangle=45),
                    xaxis_title=None,
                    legend_title='Gesamtschadstoffe (µg/m³)',
                    title=dict(x=0.5, xanchor='center', y=0.9),
                    legend=dict(x=0.5, xanchor='center', y=1.05, yanchor='bottom'))

    # Anzeigen der Grafik in Streamlit
    st.plotly_chart(fig)

    # =======================
    # 9. Vorhersagen und Bewertung
    # =======================
    y_pred = model.predict(X_test)
    st.write(f"***Modellbewertung (LinearRegression):** MSE: {mean_squared_error(y_test, y_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}, R²: {r2_score(y_test, y_pred):.2f}*")


# Abschnitt 6
    st.header("6. Einschränkungen der Ergebnisse")

    st.markdown("""
1. **Vorhersagen mit Extrapolation**: Machine-Learning-Modelle wie LinearRegression und RandomForestRegressor funktionieren gut innerhalb des Bereichs der Daten, auf denen sie trainiert wurden (Interpolation), können jedoch Probleme bei der Vorhersage außerhalb dieses Bereichs haben (Extrapolation) (zum Beispiel, bei Vorhersagen für zukünftige Jahre werden Muster aus den letzten Jahren wiederholt, anstatt zukünftige Trends zu projizieren).
   - **Zeitreihenmodelle (ARIMA, SARIMA, Prophet)**: können kleine Schwankungen erfassen und bessere Vorhersagen machen.
   - **Recurrent Neural Networks (RNN, LSTM)**: funktioniert nicht gut, wenn nur wenige Daten vorliegen.

2. **Große Variabilität in den Trends der Beziehungen einiger Daten für verschiedene Länder und Jahre**. Wenn ein einzelnes Modell auf den gesamten Datensatz angewendet wird (zum Beispiel, um die Variation der Verschmutzung mit der Produktion erneuerbarer Energien in 30 Ländern mit sehr unterschiedlichen Trends zu analysieren), können Präzisionsprobleme auftreten. Alternativen, um genauere Vorhersagen zu erhalten, wären, separate Modelle für jedes Land zu trainieren oder fortschrittliche Modelle wie XGBoost zu verwenden.

3. **Wenige Daten** (10 Jahre für jedes Land).

4. **Einfluss anderer Faktoren auf die Daten, die im Modell nicht berücksichtigt wurden** (zum Beispiel Unterschiede im Lebensstil und deren Einfluss auf die Gesundheit).
5. **Optimierung von Modellen**: Wäre es möglich, eine Modelloptimierung durchzuführen, um die Vorhersagegenauigkeit zu verbessern, Overfitting zu reduzieren und Underfitting zu vermeiden. Optimierung der Hyperparameter (zum Beispiel mit Grid Search oder Random Search).
""")