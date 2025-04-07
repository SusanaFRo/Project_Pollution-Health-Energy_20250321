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

    
    st.header("1. Lineplot: Übersicht Todesfälle an Kreislaufsystem, Respiratorische und Lungenkrebs Erkrankungen")
    trend_data = df.groupby("Jahr")[["KE", "RE", "LE"]].mean().reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(trend_data["Jahr"], trend_data["KE"], color="blue", label="Kreislaufsystem Erkrankungen (KE)")
    ax1.set_xlabel("Jahr")
    ax1.set_ylabel("KE Fälle", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(trend_data["Jahr"], trend_data["RE"], color="green", label="Respiratorische Erkrankungen (RE)")
    ax2.plot(trend_data["Jahr"], trend_data["LE"], color="red", label="Lungenkrebs (LE)")
    ax2.set_ylabel("RE & LE Fälle", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    plt.title("Global Disease Trends (2010-2019)")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    st.pyplot(fig)

    st.markdown("---")
    st.header("2. Bar plot: Respiratorische Erkrankungen und Lungenkrebs über die Jahre")
   
    yearly_avg_cases = df.groupby("Jahr")[["RE", "LE"]].mean().reset_index()

    melted_yearly_avg_cases = yearly_avg_cases.melt(id_vars="Jahr", value_vars=["RE", "LE"],
                                                var_name="Disease", value_name="Cases")

   
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Jahr", y="Cases", hue="Disease", data=melted_yearly_avg_cases,
                order=sorted(yearly_avg_cases["Jahr"]))  # Sort by year
    plt.xlabel("Jahr")
    plt.ylabel("Durchschnittswerte")
    plt.title("")
    st.pyplot(plt)

    st.markdown("---")
    
   
    st.header("3. Korrelation zwischen den Erkrankungen")
    corr_data = df[["KE", "RE", "LE"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_data, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("")
    st.pyplot(plt)

    st.markdown("---")
    
    
    st.header("3. Heatmap: Kreislaufsystem Erkrankungen bei Land und Jahr")
    heatmap_data = df.pivot_table(index="ISO3", columns="Jahr", values="KE", aggfunc="sum")
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.5)
    plt.xlabel("Jahr")
    plt.ylabel("Land")
    plt.title("")
    st.pyplot(plt)
