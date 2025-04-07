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
df_sum = pd.read_csv('daten_sum_20250310.csv', sep=';')
df_sum['sum_polutants'] = df_sum['sum_polutants'].str.replace(',', '.').astype(float)


def app():
    st.header("Luftqualität in Europa")

    st.markdown("---")

    st.subheader("Lineplot: Entwicklung der PM2.5, PM10 und NO2 Werte über die Jahre")

    yearly_avg = df.groupby('Jahr')[['PM2.5', 'PM10', 'NO2']].mean().reset_index()

    # Plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=yearly_avg, x='Jahr', y='PM2.5', label='PM2.5')
    sns.lineplot(data=yearly_avg, x='Jahr', y='PM10', label='PM10')
    sns.lineplot(data=yearly_avg, x='Jahr', y='NO2', label='NO2')

    # Add WHO recommended values
    plt.axhline(y=5, color='blue', linestyle='--', label='WHO PM2.5 (5 µg/m³)')
    plt.axhline(y=15, color='orange', linestyle='--', label='WHO PM10 (15 µg/m³)')
    plt.axhline(y=10, color='green', linestyle='--', label='WHO NO2 (10 µg/m³)')

    plt.title("Durchschnittliche Werte (2010-2019)")
    plt.xlabel("Jahr")
    plt.ylabel("Konzentration (µg/m³)")
    plt.legend()
    st.pyplot(plt)

    st.markdown("---")

    st.subheader("Heatmap: Summe der Schadstoffe nach Land und Jahr")
    # Create the pivot table for the heatmap
    heatmap_data = df_sum.pivot(index='ISO3', columns='Jahr', values='sum_polutants')
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        heatmap_data, 
        cmap='YlOrRd', 
        annot=True, 
        fmt=".1f", 
        linewidths=0.5, 
        annot_kws={"size": 10}, 
        cbar_kws={'label': 'Summe der Schadstoffe (µg/m³)'}
    )
    
    # Replace dots with commas
    for text in ax.texts:
        text.set_text(text.get_text().replace('.', ','))
    
    # Customize the plot
    plt.title('Summe der Schadstoffe nach Land und Jahr', fontsize=14)
    plt.xlabel('Jahr', fontsize=12)
    plt.ylabel('Land', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Display the plot in Streamlit
    st.pyplot(plt.gcf())  

    st.markdown("---")
    st.subheader("Interaktives Heatmap: Summe der Schadstoffe nach Land und Jahr")

    # Create the interactive heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Jahr", y="Land", color="sum_polutants (µg/m³)"),
        color_continuous_scale="YlOrRd",
        aspect="auto"
    )
    
    # Customize the layout
    fig.update_layout(
        title="Summe der Schadstoffe-Wärmekarte",
        title_x=0.5,  # Center the title
        xaxis_title="Jahr",
        yaxis_title="Land",
        coloraxis_colorbar=dict(title="Summe der Schadstoffe (µg/m³)"),
        xaxis=dict(
            tickmode="array",
            tickvals=list(heatmap_data.columns), 
            ticktext=[str(year) for year in heatmap_data.columns]
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(heatmap_data.index))),  
            ticktext=heatmap_data.index,  
            tickangle=0  
        ),
        height=600,  
        width=800    
    )
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)  

    st.markdown("---")

    st.subheader("Bar Plot: Luftqualitätsvergleich nach Ländern")

    # Select a specific year
    selected_year = st.selectbox("Jahr auswählen:", df['Jahr'].unique())

    # Filter data for the selected year
    df_year = df[df['Jahr'] == selected_year]

    # Plot top 10 countries with highest PM2.5 levels
    top_pm25 = df_year.nlargest(10, 'PM2.5')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_pm25, x='ISO3', y='PM2.5', palette='viridis')

    # Add WHO recommended value for PM2.5
    plt.axhline(y=5, color='red', linestyle='--', label='WHO PM2.5 (5 µg/m³)')

    plt.title(f"Die 10 Länder mit den höchsten PM2.5-Werten ({selected_year})")
    plt.xlabel("Land")
    plt.ylabel("PM2.5 Werte (µg/m³)")
    plt.legend()
    st.pyplot(plt)

    st.markdown("---")

    st.subheader("Scatter Plot: PM2.5 vs NO2")

    # Interactive scatter plot
    fig = px.scatter(df, x='PM2.5', y='NO2', color='ISO3',
                    title="PM2.5 vs NO2",
                    labels={"PM2.5": "PM2.5 (µg/m³)", "NO2": "NO2 (µg/m³)"})

    # Add WHO recommended values
    fig.add_hline(y=10, line_dash="dash", line_color="green", annotation_text="WHO NO2 (10 µg/m³)")
    fig.add_vline(x=5, line_dash="dash", line_color="red", annotation_text="WHO PM2.5 (5 µg/m³)")

    st.plotly_chart(fig)

    st.markdown("---")

    st.subheader("Histogram: Verteilung von PM2.5 Werten")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['PM2.5'], bins=20, kde=True, color='blue')

    # Add WHO recommended value for PM2.5
    plt.axvline(x=5, color='red', linestyle='--', label='WHO PM2.5 (5 µg/m³)')

    #plt.title("Distribution of PM2.5 Levels")
    plt.xlabel("PM2.5 Werte (µg/m³)")
    plt.ylabel("Verteilung")
    plt.legend()
    st.pyplot(plt)

    st.markdown("---")

    st.subheader("Box Plot")

    # Plot box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['PM2.5', 'PM10', 'NO2']], palette='Set2')

    # Add WHO recommended values
    plt.axhline(y=5, color='green', linestyle='--', label='WHO PM2.5 (5 µg/m³)')
    plt.axhline(y=15, color='red', linestyle='--', label='WHO PM10 (15 µg/m³)')
    plt.axhline(y=10, color='blue', linestyle='--', label='WHO NO2 (10 µg/m³)')

    #plt.title("Box Plot of PM2.5, PM10, and NO2\nWHO Guidelines: PM2.5=5, PM10=15, NO2=10")
    plt.ylabel("Werte (µg/m³)")
    plt.legend()
    st.pyplot(plt)

    st.markdown("---")

    st.subheader("Durchschnittliche Verschmutzung pro Land")

    df_avg = pd.read_csv('daten_polutans_avg_jahr_20250310.csv', sep=';')
    df_avg['PM2,5_avg_jahr'] = df_avg['PM2,5_avg_jahr'].str.replace(',', '.').astype(float)
    df_avg['PM10_avg_jahr'] = df_avg['PM10_avg_jahr'].str.replace(',', '.').astype(float)
    df_avg['NO2_avg_jahr'] = df_avg['NO2_avg_jahr'].str.replace(',', '.').astype(float)

   
    df_avg = df_avg.sort_values(by='PM2,5_avg_jahr', ascending=False)
    
    
    fig = go.Figure()
    
    
    fig.add_trace(go.Scatter(
        x=df_avg['ISO3'], y=df_avg['PM2,5_avg_jahr'], mode='lines+markers', name='PM2.5',
        line=dict(color='blue', width=2), marker=dict(symbol='circle', size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df_avg['ISO3'], y=df_avg['PM10_avg_jahr'], mode='lines+markers', name='PM10',
        line=dict(color='orange', width=2), marker=dict(symbol='square', size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df_avg['ISO3'], y=df_avg['NO2_avg_jahr'], mode='lines+markers', name='NO2',
        line=dict(color='red', width=2), marker=dict(symbol='triangle-up', size=8)
    ))
    
    # Customize the layout
    fig.update_layout(
        title='',
        title_x=0.5,  # Center the title
        xaxis_title='Land',
        yaxis_title='Verschmutzungsgrad (µg/m³)',
        legend_title="Pollutants",
        template='ggplot2',
        hovermode='x unified',  
        xaxis=dict(tickangle=-45), 
        height=600,  
        width=1000,  
    )
    
    # Display the plot in Streamlit
    st.plotly_chart(fig) 