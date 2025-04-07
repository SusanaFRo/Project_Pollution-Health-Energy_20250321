import streamlit as st

import page_start
import page_data
import page_airquality
import page_health
import page_energy
import page_correlation
import page_predictions
import page_abschlussbericht

# Titel der Präsentation
st.title("Auswirkungen der Energiewende auf Luftqualität und Gesundheit in Europa von 2010 bis 2019")


# Definiere die Seiten der App, die verschiedene Aspekte von Streamlit vorstellen
pages = {
    "Startseite"            : page_start,
    "Datenübersicht"        : page_data,
    "Luftqualität"          : page_airquality,
    "Gesundheit"            : page_health,
    "Energieproduktion"     : page_energy,
    "Zusammenhänge"         : page_correlation,
    "Prognosen"             : page_predictions,
    "Abschlussbericht"      : page_abschlussbericht
}



select = st.sidebar.radio("",list(pages.keys()))

# Starte die ausgewählte Seite
pages[select].app()


# Footer
st.sidebar.markdown("---")
st.sidebar.write("Erstellt mit ❤️ von Susana und Yani")