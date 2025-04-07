import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go


# Load  CSV file
df = pd.read_csv("daten_AK.csv",sep=";", dtype={'Jahr': str})
df['PM2.5'] = df['PM2.5'].astype(float)
df['PM10'] = df['PM10'].astype(float)
df['NO2'] = df['NO2'].astype(float)


def app():

    st.header("1. Lineplot: Globale Entwicklung der Energieproduktion (Fossile Energie / Erneuerbare Energie)")
    trend_data = df.groupby("Jahr")[["FEn", "EEn"]].mean().reset_index()
    trend_data["Jahr"] = trend_data["Jahr"].astype(str)

    plt.figure(figsize=(10, 6))
    plt.plot(trend_data["Jahr"], trend_data["FEn"], label="Fossile Energie")
    plt.plot(trend_data["Jahr"], trend_data["EEn"], label="Erneuerbare Energie")
    plt.xlabel("Jahr")
    plt.ylabel("Energieproduktion")
    plt.title("")
    plt.legend()
    st.pyplot(plt)

    st.markdown("---")

    st.header("2. Totale Energieproduktion")
    total_energy = df.groupby("Jahr")[["FEn", "KEn", "EEn"]].sum().reset_index()
    melted_total_energy = total_energy.melt(id_vars="Jahr", var_name="Energy Type", value_name="Production")

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Jahr", y="Production", hue="Energy Type", data=melted_total_energy)
    plt.xlabel("Jahr")
    plt.ylabel("Totale Energieproduktion")
    plt.title("")
    st.pyplot(plt)

    st.markdown("---")

    st.header("3. Heatmap: Fossile Energieproduktion bei Land und Jahr")
    heatmap_data = df.pivot_table(index="ISO3", columns="Jahr", values="FEn", aggfunc="sum")  # Example for FEn
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.5)
    plt.xlabel("Jahr")
    plt.ylabel("Land")
    plt.title("")
    st.pyplot(plt)

    st.markdown("---")

    st.header("4. Anteil der Energieproduktion (Fossile Energie / Erneuerbare Energie)")
    proportion_data = df.groupby("Jahr")[["FEn", "EEn"]].sum().reset_index()
    proportion_data[["FEn", "EEn"]] = proportion_data[["FEn", "EEn"]].div(proportion_data[["FEn", "EEn"]].sum(axis=1), axis=0)

    # Convert "Jahr" to string
    proportion_data["Jahr"] = proportion_data["Jahr"].apply(str)

    # Debugging (optional)
    print(proportion_data.dtypes)
    print(proportion_data.head())

    plt.figure(figsize=(10, 6))
    plt.stackplot(proportion_data["Jahr"], proportion_data["FEn"], proportion_data["EEn"],
                labels=["FEn", "EEn"], colors=["orange", "green"])
    plt.xlabel("Jahr")
    plt.ylabel("Anteil")
    plt.title("")
    plt.legend(loc="upper left")
    st.pyplot(plt)


    # 5. Faceted Line Plot: Energy Trends by Country (FEn + EEn)
    st.header("5. Entwicklung der Energieproduktion bei Land (Fossile Energie vs. Kern- und Erneuerbare Energie)")
    g = sns.FacetGrid(df, col="ISO3", col_wrap=3, height=3, sharey=False)
    g.map(sns.lineplot, "Jahr", "FEn", color="orange", label="FEn")
    g.map(sns.lineplot, "Jahr", "EKEn", color="green", label="EKEn")
    g.set_axis_labels("Jahr", "Energie Produktion")
    g.set_titles("{col_name}")
    g.add_legend()
    st.pyplot(g)