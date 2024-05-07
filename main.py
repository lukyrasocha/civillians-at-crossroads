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
threshold_fatalities = edges['FATALITIES'].quantile(0.95)
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


st.markdown('### The real price of conflict')
event_type_and_fatalities = event_type_and_fatalities(data)
st.plotly_chart(event_type_and_fatalities)
st.markdown("War invariably claims civilian lives. Its lethality is an ever-present reality, with civilians bearing the brunt of conflict. To gain a deeper insight into how these conflicts impact fatalities, we utilized a double y-axis plot to compare the number of different types of events and their associated fatalities. One axis represents the frequency of events with a bar graph, while the other depicts the number of fatalities with a dot plot. The graph highlights the severe toll on civilians, with fatalities prominently associated with each type of event.")
st.markdown("Explosions and remote violence emerge as the most frequent occurrences during this period, while riots are the least common conflict experienced in the region. Of particular note is the prevalence and lethality of battles. Despite occurring less frequently than explosions, battles resulted in significant fatalities. In fact, battles accounted for 36,567 recorded fatalities, surpassing even explosions in their impact on civilian lives.")

# Display the fatalities map
fatalities_map = fatailities_map(data, gdf)
st.plotly_chart(fatalities_map)
st.markdown("Before 2022, fatalities were scarce across most regions, with denser concentrations primarily observed in the eastern parts of Ukraine, though the numbers remained relatively low. However, on 24th February 2022, Russia's invasion of Ukraine marked a significant escalation in the ongoing Russo-Ukrainian War from 2014. This pivotal event is reflected in the data, as the number of fatalities sharply increased and became more concentrated, particularly in the central and western regions of Ukraine.")


st.markdown('### Who is responsible?')
# Call the function to draw the graph
graph_viz = draw_graph(G)
# Use Streamlit components to display the visualization
st.altair_chart(graph_viz, use_container_width=True)
st.markdown("After gaining a deeper understanding of fatalities, the question of accountability becomes intriguing. Through analysis of the network graph, we can identify the actors responsible for initiating conflicts that result in fatalities and who they frequently engage with. Among these actors, the Military Force of Russia (2000-) stands out for its significant contribution to fatalities, particularly in its engagements with the Military Force of Ukraine (2019-). Surprisingly, the second most common interaction for Russia is with Ukrainian civilians. This is particularly tragic because civilians lack the means to defend themselves, and there are no fatalities caused by civilians themselves.")


st.markdown('### Everyday people, extraordinary circumstances')
civilians_fig = plot_violence_against_civilians(data)
st.plotly_chart(civilians_fig)
st.markdown("While the broad numbers of fatalities offer a stark picture of the conflict's severity, they do not fully capture the day-to-day reality faced by civilians. To understand the true human cost, we turn our attention to incidents specifically categorized as 'Violence against civilians'. By examining the different sub-events under this category, we can see more clearly how these conflicts permeate the lives of ordinary people.")
st.markdown("Based on the plot, we observe a sharp increase in most sub-events after March 2022, coinciding with the escalation of the invasion. 'Shelling/artillery/missile attacks' consistently emerge as the most common form of violence against civilians after 2022. The second most common form of violence varies across different time periods. Following March 2022, the occurrence of 'Air/drone strikes' dwindled and remained low throughout 2022. During this time, incidents involving 'Remote explosive/landmine/IED' surged, temporarily assuming the position of the second most common form of violence. However, their occurrence decreased after September 2023. 'Air/drone strikes', on the other hand, increased after 2023 and subsequently became the second most common form of violence.")



st.markdown('### Nowhere is safe')
animated_geo_fig = animated_map(data, gdf)
st.plotly_chart(animated_geo_fig)
st.markdown("The plight of defenseless civilians facing various types of violence daily prompts an important question: Are there any regions where people can flee to seek refuge? Yet the grim truth is that there seems to be nowhere safe to hide. Upon analyzing the map plotting all incidents targeting civilians, it becomes evident that no region in Ukraine has been spared from such violence. What's even more disheartening is that some areas, like Donetsk and Kharkiv, are engulfed in a multitude of conflicts and are almost entirely covered with violence, rendering them particularly unsafe for civilians.")



st.markdown('### Population Distribution')
population_distribution = plot_population_distribution(data)
st.altair_chart(population_distribution)
st.markdown("In Rule 23 from Customary International Humanitarian Law (CIHL), it states that 'Each party to the conflict must, to the extent feasible, avoid locating military objectives within or near densely populated areas'[3]. We can observe the implications of this principle from the population distribution plot, which shows the population density within 1 square kilometer and the number of incidents targeting civilians in these densely populated regions. The plot reveals that most attacks occurred in regions with a population density of less than 10,000 people per square kilometer, reflecting a commitment to humanitarian principles. However, it's concerning that some incidents still target highly dense regions, resulting in a significant toll on civilian lives.")

st.markdown('### Average Proximity of Sub-Event Types to Populations')
average_population = plot_average_population(data)
st.plotly_chart(average_population)

# Protests over time

st.markdown('### Protests Over Time')
protests_over_time = plot_protests_over_time(data)

st.plotly_chart(protests_over_time)
st.markdown("From 2018 to 2021, there was a steady increase in protests, indicating a growing trend of people coming together to voice their concerns and push for change. However, everything changed after 2022 when the conflict escalated. The number of protests dropped significantly during this time. Between May 2022 and October 2023, there were very few protests happening. This decrease coincided with the worsening situation of the war, which made people less likely to gather and protest. The danger and uncertainty caused by the conflict overshadowed the desire for social and political activism.")
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
        <p style='margin-left: 10px;'>1000 people</p>
    </div>
</div>
"""
st.markdown(html, unsafe_allow_html=True)
st.image('./assets/spirals.png', caption='On the left side, we have the number of deaths, and on the right side, we have the number of refugees. One white dot represents 10 deaths, one yellow dot represents 1000 refugees.')


st.markdown("## References")
st.markdown("1. [ACLED](https://acleddata.com/)")
st.markdown("2. [War in Ukraine: Two Years On, Attacks Against Civilians on the Rise Again](https://reliefweb.int/report/ukraine/war-ukraine-two-years-attacks-against-civilians-rise-again)")
st.markdown("3. [Customary International Humanitarian Law (CIHL)](https://ihl-databases.icrc.org/en/customary-ihl/v1/rule23)")

# %%
