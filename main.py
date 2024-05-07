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

st.markdown(
    "LINK TO NOTEBOOK [HERE](https://github.com/lukyrasocha/02806-socviz-final-project/blob/main/final_project.ipynb). Please, if some visualisations are not rendering, try to reload the page")

# st.markdown("## Geopolitical Significance of the Black Sea Region")

st.markdown("The Black Sea region, an important area due to its strategic geopolitical location and rich history, has become an arena for modern conflict and power struggles, particularly impacting Ukraine. In recent years, this area has witnessed an escalation in political violence, deeply affecting the lives of countless civilians.")
st.markdown(
    "> *“We’ve seen more people hiding again in the nearby shelter because of the air raids. The fear is still here because the war continues, and we, as civilians, are still a target”* – A civilian in Ukraine interviewed by CIVIC [2].")

st.markdown(
    "This narrative aims to shed light on the human cost of these conflicts. By looking into data [1] on various types of events such as battles, violent demonstrations, and other forms of political violence. We will tell the story not just about the frequency and types of these events, but their profound impact on civilian populations, from fatalities and injuries to displacement. [2]")


st.markdown('### The real price of conflict')
st.markdown("War invariably claims human lives. Its lethality is an ever-present reality, with people bearing the brunt of conflict. To gain a deeper insight into how these conflicts impact fatalities, we utilized a layered plot to compare the number of different types of events and their associated fatalities. The first one represents the frequency of events with a bar graph, while the other depicts the number of fatalities with a scatter plot. The graph highlights the severe toll on civilians, with fatalities prominently associated with each type of event.")
event_type_and_fatalities = event_type_and_fatalities(data)
st.plotly_chart(event_type_and_fatalities)

st.markdown("Explosions and remote violence emerge as the most frequent occurrences during this period, while riots are the least common conflict experienced in the region. This is naturally not a surprise due the inherent violent natures of such event types together with the frequency with which they have been employed throughout the conflicts. While battles primarily impact military actors,  explosions and remote violence (including grenades, shelling, missile and drone attacks etc.) widely affects military actors and civilians alike not only in terms of fatalities but also with regard to the insecurities and fear felt by people presiding in such perilous environments. Of particular note is the prevalence and lethality of battles. Despite occurring less frequently than explosions, battles resulted in significant fatalities. In fact, battles accounted for 36,567 recorded fatalities, surpassing even explosions in their impact on human lives.")

# Display the fatalities map
st.markdown(
    "Before 2022, fatalities were scarce across most regions, with denser concentrations primarily observed in the eastern parts of Ukraine, though the numbers remained relatively low. However, on 24th February 2022, Russia's invasion of Ukraine marked a significant escalation in the ongoing Russo-Ukrainian War from 2014. [4] This pivotal event is reflected in the data, as the number of fatalities sharply increased and became more concentrated, particularly in the central and western regions of Ukraine.")
fatalities_map = fatailities_map(data, gdf)
st.plotly_chart(fatalities_map)


st.markdown('### Who is responsible?')
st.markdown("After gaining a deeper understanding of all the fatalities, the question of accountability becomes intriguing. Through analysis of the network graph, we can identify the actors responsible for initiating conflicts that result in fatalities and who they frequently engage with. The link width between the actors corresponds to the cost of human lives and the colour of each actor is the total amount for which the actor is responsible for. Among these actors, the Military Force of Russia stands out for its significant contribution to fatalities, particularly in its engagements with the Military Force of Ukraine. Unfortunately, the second most common interaction for Russia is with Ukrainian civilians. This is a stark reminder of the human cost of conflict, with civilians often caught in the crossfire.")
# Call the function to draw the graph
graph_viz = draw_graph(G)
# Use Streamlit components to display the visualization
st.altair_chart(graph_viz, use_container_width=True)


st.markdown('### Everyday people, extraordinary circumstances')
st.markdown("While the broad numbers of fatalities offer a glance of the conflicts' severity, they do not fully capture the day-to-day reality faced by civilians. To understand the other side of human cost, we turn our attention to incidents specifically categorized as 'Violence against civilians'. By examining the different sub-events under this category, we can see more clearly how these conflicts permeate the lives of ordinary people.")
st.markdown(
    ">*“Our children got used to studying, playing, eating, and doing their homework in bomb shelters because of the Russian attacks. It’s now become their childhoods”* — A teacher in Ukraine interviewed by CIVIC [2].")
st.markdown(
    "Based on the plot, we observe a sharp increase in most sub-events after March 2022, coinciding with the escalation of the invasion. 'Shelling/artillery/missile attacks' consistently emerge as the most common form of violence against civilians after 2022. These types of attacks are often employed to demolish critical infrastructure sites, resulting in degradation of internet, water or heat supplies. But more importantly, they pose a direct threat to civilian lives, causing significant casualties. [5] The second most common form of violence varies across different time periods, either being drone strikes or remote explosives. Even though events such as sexual violence does not belong to the most frequent it is still important to highlight the severity of such crimes. Only a few days after the invasion, more than 40 cases of sexual violence were reported. According to the UN, these types of war crimes are often used as a military strategy and a deliberate tactic to dehuminize and terrorize the population. [6]")

civilians_fig = plot_violence_against_civilians(data)
st.plotly_chart(civilians_fig)


st.markdown('### Nowhere is safe')
st.markdown("The plight of defenseless civilians facing various types of violence daily prompts an important question: Are there any regions where people can flee to seek refuge? Yet the grim truth is that there seems to be nowhere safe to hide. Upon analyzing the map plotting all incidents targeting civilians, it becomes evident that no region in Ukraine has been spared from such violence. What's even more disheartening is that some areas, like Donetsk and Kharkiv, are engulfed in a multitude of conflicts and are almost entirely covered with violence, rendering them particularly unsafe for civilians.")

st.markdown(
    "Despite easier pathways to asylum for people living in these areas, in Ukraine there is mandatory conscription for men who are between 18-60 years old. This law has caused many Ukrainian families to not flee their country, as they might be unwilling to leave brothers, fathers and sons behind. As of February 2024, this resulted in 3.7 million internally displaced people in Ukraine [8], who are now subjected to the violence and instability produced by all these different sub-event types. [7]")
animated_geo_fig = animated_map(data, gdf)
st.plotly_chart(animated_geo_fig)

st.markdown('### The Challenge of Protecting Civilians in War')
st.markdown(
    "In Rule 23 from Customary International Humanitarian Law (CIHL), it states that *'Each party to the conflict must, to the extent feasible, avoid locating military objectives within or near densely populated areas'*. [3] We can observe the implications of this principle from the population distribution plot, which shows the population density within 1 square kilometer and the number of incidents targeting civilians in these densely populated regions. The plot reveals that most attacks occurred in regions with a population density of less than 10,000 people per square kilometer, reflecting a commitment to humanitarian principles. However, it's concerning that some incidents still target highly dense regions, resulting in a significant toll on civilian lives.")
population_distribution = plot_population_distribution(data)
st.altair_chart(population_distribution)

st.markdown("If we examine the average population around the different types of violent events, it can be observed that most of the lethal and cruel attacks, such as 'Air/drone strikes' or 'Remote explosive/landmine/IED', tend to target less densely populated regions. However, it's concerning to note that grenades have the highest average population among all categories, indicating that they are often thrown in highly populated regions. Looking closer at the data, there were 78 of such incidents, resulting in maximum of 3 fatalities but presumably many more injuries.")

average_population = plot_average_population(data)
st.plotly_chart(average_population)

# Protests over time

st.markdown('### Protests Over Time')
protests_over_time = plot_protests_over_time(data)

st.plotly_chart(protests_over_time)
st.markdown("The conflicts are also reflected in the number of protests happening in the region. The plot shows the number of protests over time, with a clear after 2022. This decrease in protests coincides with the escalation of the conflict, indicating that the ongoing war has had a significant impact on the ability of people to gather and protest. Protests have been used throughout history as a way for people to voice their concerns and push for change. This shows that conflicts not only have a direct impact on the lives of civilians but also on their ability to engage in social and political activism.")

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
st.markdown("The impact of the war in Black Sea Region is deeply distressing, with a rising number of casualties and individuals forced to flee their homes. According to the UNHCR, as of February 2024, approximately 6.5 million refugees from Ukraine have been documented worldwide. This stark reality underscores the severity of the situation and emphasizes the urgent need for global solidarity and support. ")

st.markdown("## References")
st.markdown("1. [ACLED](https://acleddata.com/)")
st.markdown("2. [War in Ukraine: Two Years On, Attacks Against Civilians on the Rise Again](https://reliefweb.int/report/ukraine/war-ukraine-two-years-attacks-against-civilians-rise-again)")
st.markdown("3. [Customary International Humanitarian Law (CIHL)](https://ihl-databases.icrc.org/en/customary-ihl/v1/rule23)")
st.markdown("4. [Ukraine in maps](https://www.bbc.com/news/world-europe-60506682)")
st.markdown("5. [Russia intensified missile attacks against civilian infrastructure](https://osce.usmission.gov/on-russias-intensified-missile-attacks-against-civilian-infrastructure-of-ukraine-amidst-russias-ongoing-aggression/)")
st.markdown("6. [The Devastating Use of Sexual Violence as a Weapon of War](https://www.thinkglobalhealth.org/article/devastating-use-sexual-violence-weapon-war)")
st.markdown("7. [Conscription in Ukraine](https://apnews.com/article/ukraine-parliament-recruit-army-russia-war-5b7d9f58bb398b4ad1296311b8130b92)")
st.markdown("8. [Internally displaced people in Ukraine](https://www.unrefugees.org/emergencies/ukraine/)")

# %%
