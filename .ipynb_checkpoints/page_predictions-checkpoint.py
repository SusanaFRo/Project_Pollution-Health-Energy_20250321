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

    st.subheader("1. Korrelationsmatrix numerischer Variablen")

    st.write("LinearRegressionsmodell")
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
        'Erkrankungen des Kreislaufsystems (Todesfalle pro 100.000)': 'Kreislaufsystem_Erk_Tod',
        'Erkrankungen der Atemwege (Todesfalle pro 100.000)': 'Atemwege_Erk_Tod',
        'Krebs der Atemwege (Todesfalle pro 100.000)': 'Krebs_Atemwege_Tod',
        'Todesfalle insgesamt (pro 100.000)': 'Tod_insgesamt',
        'Fossiler_Energieerzeugung (TWh)': 'Fossil_energie_TWh',
        'nukleare_Energieerzeugung (TWh)': 'nuklear_energie_TWh',
        'Erneuerbare_Energieerzeugung (TWh)': 'Ern_energie_TWh'
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

    st.subheader("2. Prognose der gesamten Schadstoffemissionen zur Reduzierung umweltbedingter Todesfälle um 0% bis 95% (Basisjahr 2019)")


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
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write("\n===== Modellbewertung: LinearRegression =====")
    st.write(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")



    st.markdown("---")

    st.subheader("3. Prognose des Jahres zur Reduzierung umweltbedingter Todesfälle um 0% bis 95% (Basisjahr 2019)")


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
    st.write("\n===== Modellbewertung: LinearRegression =====")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}, R²: {r2_score(y_test, y_pred):.2f}")



    st.markdown("---")

    st.subheader("4. Prognose der Erzeugung erneuerbarer Energien zur Reduzierung gesamten Schadstoffemissionen (PM2,5 + PM10 + NO2) gemäß der WHO-Empfehlung")

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
                        'Predicted_Renew_TWh': 'Prognose der erneuerbaren Energieerzeugung(TWh)', 
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
    st.write("\n===== Modellbewertung: RandomForestRegressor =====")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}, R²: {r2_score(y_test, y_pred):.2f}")