# %%
import streamlit as st
import geopandas as gpd
from plot import *

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv', sep=';')
    
    data = data_preprocessing(data)
    return data

def data_preprocessing(data):
    # Convert event date to datetime format
    data['EVENT_DATE'] = pd.to_datetime(data['EVENT_DATE'])
    # Extract month and year from 'EVENT_DATE'
    data['MONTH'] = data['EVENT_DATE'].dt.month
    data['YEAR'] = data['EVENT_DATE'].dt.year
    # Convert lat, lon to correct delimiter 
    data['LATITUDE'] = pd.to_numeric(data['LATITUDE'].str.replace(',', '.'), errors='coerce')
    data['LONGITUDE'] = pd.to_numeric(data['LONGITUDE'].str.replace(',', '.'), errors='coerce')
    
    return data

# This function can now be used in a Streamlit app


############################
# DATA PREPARATION 
############################
data = load_data()
gdf = gpd.read_file('region.geojson')


# Create a DataFrame for interactions with sum of fatalities
edges = data.groupby(['ACTOR1', 'ACTOR2'])['FATALITIES'].sum().reset_index()

# Filter edges to only include those with fatalities above a certain threshold and select top 20
threshold_fatalities = edges['FATALITIES'].quantile(0.95)  # Adjust threshold as needed
filtered_edges = edges[edges['FATALITIES'] > threshold_fatalities].nlargest(20, 'FATALITIES')

# Create the graph with filtered data
G = nx.from_pandas_edgelist(filtered_edges, 'ACTOR1', 'ACTOR2', ['FATALITIES'])

# Add attributes to each node.
for n in G.nodes():
    G.nodes[n]['name'] = n
    G.nodes[n]['Fatalities caused'] = data[data['ACTOR1'] == n]['FATALITIES'].sum()



st.title('Civilians at the Crossroads: The Human Cost of Conflict in the Black Sea Region')

# st.markdown("## Geopolitical Significance of the Black Sea Region")

st.markdown("The Black Sea region, an important area due to its strategic geopolitical location and rich history, has become an arena for modern conflict and power struggles, particularly impacting Ukraine. In recent years, this area has witnessed an escalation in political violence, deeply affecting the lives of countless civilians.")
st.markdown(
    "> *“We’ve seen more people hiding again in the nearby shelter because of the air raids. The fear is still here because the war continues, and we, as civilians, are still a target”* – A civilian in Ukraine interviewed by CIVIC [2].")

st.markdown(
    "This narrative aims to shed light on the human cost of these conflicts. By looking into data [1] on various types of events such as battles, violent demonstrations, and other forms of political violence. We will tell the story not just about the frequency and types of these events, but their profound impact on civilian populations, from fatalities and injuries to displacement. [2]")

st.markdown('### The real price of conflict: Fatalities')

event_type_and_fatalities = event_type_and_fatalities(data)
st.plotly_chart(event_type_and_fatalities)
st.markdown(
    "The persistent volatility of region conflicts continues to exert a profound impact on civilian populations. The graph highlights the severe toll on civilians, with fatalities prominently associated with each type of event. The battles and explosions/remote violence have been the most prevalent and lethal, with over 36,567 and 32,104 recorded events respectively, resulting in substantial fatalities. ")

# Display the fatalities map
fatalities_map = fatailities_map(data, gdf)
st.plotly_chart(fatalities_map)


st.markdown('### Who is responsible?')

# Call the function to draw the graph
graph_viz = draw_graph(G)

# Use Streamlit components to display the visualization
st.altair_chart(graph_viz, use_container_width=True)

st.markdown('### Everyday people, extraordinary circumstances')

civilians_fig = plot_violence_against_civilians(data)

st.plotly_chart(civilians_fig)


st.markdown("While the broad numbers of fatalities offer a stark picture of the conflict's severity, they do not fully capture the day-to-day reality faced by civilians. To understand the true human cost, we turn our attention to incidents specifically categorized as 'Violence against civilians'. By examining the different sub-events under this category, we can see more clearly how these conflicts permeate the lives of ordinary people.")
# Display the figures in the Streamlit app

st.markdown('### Nowhere is safe')

animated_geo_fig = animated_map(data, gdf)
st.plotly_chart(animated_geo_fig)

# display the distribution of POPULATION_1KM
st.markdown('### Population Distribution')

population_distribution = plot_population_distribution(data)

st.altair_chart(population_distribution)

st.markdown('### Average Proximity of Sub-Event Types to Populations')
average_population = plot_average_population(data)
st.plotly_chart(average_population)

# Protests over time

st.markdown('### Protests Over Time')
protests_over_time = plot_protests_over_time(data)

st.plotly_chart(protests_over_time)

# Generate the figure using your custom function
fig = create_strip_plot(data)

# Use Streamlit's function to display the plot
st.pyplot(fig)

html = """
<div style='width: 100%; display: flex; justify-content: space-between; align-items: center;'>
    <div style='display: flex; align-items: center;'>
        <span style='height: 25px; width: 25px; background-color: #fff; border-radius: 50%; display: inline-block;'></span>
        <p style='margin-left: 10px;'>10 people</p>
    </div>
    <div style='display: flex; align-items: center;'>
        <span style='height: 25px; width: 25px; background-color: #e8a033; border-radius: 50%; display: inline-block;'></span>
        <p style='margin-left: 10px;'>100 people</p>
    </div>
</div>
"""
st.markdown(html, unsafe_allow_html=True)
st.image('./assets/spirals.png', caption='On the left side, we have the number of deaths, and on the right side, we have the number of refugees. One white dot represents 10 deaths, one yellow dot represents 1000 refugees.')


st.markdown("## References")
st.markdown("1. [ACLED](https://acleddata.com/)")
st.markdown("2. [War in Ukraine: Two Years On, Attacks Against Civilians on the Rise Again](https://reliefweb.int/report/ukraine/war-ukraine-two-years-attacks-against-civilians-rise-again)")


