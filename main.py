import json
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import geopandas as gpd
import seaborn as sns
import networkx as nx
import nx_altair as nxa


from math import radians, cos, sin, asin, sqrt, log

# Load data


@st.cache_data
def load_data():
    data = pd.read_csv('data.csv', sep=';')
    data['EVENT_DATE'] = pd.to_datetime(data['EVENT_DATE'])
    data['LATITUDE'] = data['LATITUDE'].str.replace(',', '.')
    data['LONGITUDE'] = data['LONGITUDE'].str.replace(',', '.')
    return data


def filter(data):
    top_5_subevent_types = data['SUB_EVENT_TYPE'].value_counts().index[:5]
    filtered_data = data[data['SUB_EVENT_TYPE'].isin(top_5_subevent_types)]
    filtered_data = filtered_data.sort_values(by='YEAR')
    return filtered_data

# Altair Visualization: Event Type Distribution


def plot_event_type_distribution(data):
    chart = alt.Chart(data).mark_bar().encode(
        x='count()',
        y=alt.Y('EVENT_TYPE', sort='-x'),
        color='EVENT_TYPE',
        tooltip=['EVENT_TYPE', 'count()']
    ).properties(title='Event Type Distribution')
    return chart

# Yearly animation plot


def animate_yearly_event(data):
    # Ensure your data is sorted by 'YEAR' if the animation sequence matters
    data.sort_values('YEAR', inplace=True)

    event_type_counts = data.groupby(['YEAR', 'EVENT_TYPE']).size().reset_index(name='count')

    # Animated distribution of each individual event type over years
    fig = px.bar(event_type_counts, x='EVENT_TYPE', y='count', color='EVENT_TYPE',
                 animation_frame='YEAR', animation_group='EVENT_TYPE',
                 range_y=[0, event_type_counts['count'].max()],
                 labels={'count': 'Number of Events'})
    fig.update_layout(title="Distribution of Event Types over Years")
    return fig

# Animation map with region geo info


def fatailities_map(data, gdf):
    data = data[data['FATALITIES'] > 0]
    # order years
    data.sort_values('YEAR', inplace=True)

    min_year = data['YEAR'].min()
    all_sub_event_types = data['SUB_EVENT_TYPE'].unique()
    dummy_rows = pd.DataFrame({
        'YEAR': [min_year] * len(all_sub_event_types),
        'SUB_EVENT_TYPE': all_sub_event_types,
        'LATITUDE': [0] * len(all_sub_event_types),
        'LONGITUDE': [0] * len(all_sub_event_types),
        'FATALITIES': [0] * len(all_sub_event_types),  # Ensure dummy rows for fatalities
        'EVENT_ID_CNTY': ['dummy'] * len(all_sub_event_types)
    })
    data = pd.concat([data, dummy_rows], ignore_index=True)

    scatter_geo = px.scatter_geo(
        data, lat='LATITUDE', lon='LONGITUDE', color='FATALITIES',
        size='FATALITIES',  # Use fatalities as size for emphasis
        animation_frame='YEAR',  # animation_group='SUB_EVENT_TYPE',
        title='Evolution of Fatalities in Conflict Events',
        # hover_name='SUB_EVENT_TYPE',  # Add more hover details as needed
        # hover_data={'FATALITIES': True, 'YEAR': True},
        size_max=15,  # Adjust max size to fit the visualization
        color_continuous_scale=px.colors.sequential.OrRd  # Use a red-orange color scale
    )
    scatter_geo.update_layout(
        autosize=True,
        height=700,
        geo=dict(
            center=dict(lat=47.2, lon=31.1),  # Adjust center if needed
            scope='europe',  # Ensure the scope is correct for your data
            projection_scale=6,  # Adjust scale for visibility
            bgcolor='rgba(0, 0, 0, 0)',  # Optional: Transparent background
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Optional: Transparent plot background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Optional: Transparent paper background
        font=dict(color='white'),  # Adjust font color for visibility
    )

    return scatter_geo


def animated_map(data, gdf):
    min_year = data['YEAR'].min()
    # Get all unique sub-event types
    all_sub_event_types = data['SUB_EVENT_TYPE'].unique()
    # Create a DataFrame with dummy rows for each sub-event type for the initial year
    dummy_rows = pd.DataFrame({'YEAR': [min_year] * len(all_sub_event_types),
                               'SUB_EVENT_TYPE': all_sub_event_types,
                               'LATITUDE': [0] * len(all_sub_event_types),
                               'LONGITUDE': [0] * len(all_sub_event_types),
                               'EVENT_ID_CNTY': ['dummy'] * len(all_sub_event_types)})
    # Concatenate the dummy rows with the original filtered_data DataFrame
    data = pd.concat([data, dummy_rows], ignore_index=True)

    # Create a choropleth map
    choropleth = go.Figure(go.Choropleth(
        geojson=json.loads(gdf.to_json()),
        locations=gdf.index,
        z=[0] * len(gdf),  # Replace with your data
        hovertext=gdf['name'],  # Add hover text
        hoverinfo='text',  # Show only hover text
        marker_line=dict(width=1, color='red'),  # Set marker line width to 1 to display only the outline
        showscale=False
    ))

    # Create a scatter_geo plot using Plotly Express
    scatter_geo = px.scatter_geo(
        data, lat='LATITUDE', lon='LONGITUDE', color='SUB_EVENT_TYPE', animation_frame='YEAR',
        animation_group='SUB_EVENT_TYPE', title='Map of Events by Year and Event Type',
        hover_data={'LATITUDE': False, 'LONGITUDE': False, 'YEAR': False, 'SUB_EVENT_TYPE': False})

    # Add the choropleth traces to the scatter_geo plot
    for trace in choropleth.data:
        scatter_geo.add_trace(trace)

    # Update layout for the combined plot
    scatter_geo.update_layout(
        autosize=True,
        height=800,
        geo=dict(
            center=dict(lat=47.2, lon=31.1),  # You can adjust these values to focus on your region of interest
            scope='europe',  # Change this as necessary
            projection_scale=5.5,  # Adjust the scale for a better view
            bgcolor='rgba(0, 0, 0, 0)',  # Set background color to transparent
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background color to transparent
        font=dict(color='white'),  # Set font color to white
    )

    return scatter_geo


############################
# DATA PREPARATION AND FILTERING
############################
data = load_data()
filtered = filter(data)
gdf = gpd.read_file('region.geojson')

# Extract month and year from 'EVENT_DATE'
data['MONTH'] = data['EVENT_DATE'].dt.month
data['YEAR'] = data['EVENT_DATE'].dt.year

# Create a DataFrame for interactions with sum of fatalities
edges = data.groupby(['ACTOR1', 'ACTOR2'])['FATALITIES'].sum().reset_index()

# Filter edges to only include those with fatalities above a certain threshold and select top 20
threshold_fatalities = edges['FATALITIES'].quantile(0.95)  # Adjust threshold as needed
filtered_edges = edges[edges['FATALITIES'] > threshold_fatalities].nlargest(20, 'FATALITIES')


def event_type_and_fatalities(data):
    event_fatalities = data.groupby('EVENT_TYPE')['FATALITIES'].sum().reset_index()

    # Filter out the Violence Against Civilians event type
    event_fatalities = event_fatalities[event_fatalities['EVENT_TYPE'] != 'Violence against civilians']

    # Also, count the number of occurrences for each event type
    event_counts = data['EVENT_TYPE'].value_counts().reset_index()
    event_counts.columns = ['EVENT_TYPE', 'COUNT']

    # Merge both dataframes to have counts and fatalities side by side
    event_summary = pd.merge(event_fatalities, event_counts, on='EVENT_TYPE')

    # Create a bar chart for the number of events
    fig = go.Figure(go.Bar(
        x=event_summary['EVENT_TYPE'],
        y=event_summary['COUNT'],
        name='Number of Events',
        marker=dict(color='lightslategray'),
        # hoverinfo='skip'  # Hides hover info for the bars
        hovertemplate="<b>%{x}</b><br>Number of Events: %{y}<extra></extra>"  # Custom hover text for bars

    ))

    # Add a scatter plot on top of the bar chart for fatalities
    fig.add_trace(go.Scatter(
        x=event_summary['EVENT_TYPE'],
        y=event_summary['FATALITIES'],  # Use actual fatalities for positioning
        mode='markers+text',  # Display markers with text
        marker=dict(
            # size=20,  # Fixed size or scale appropriately
            # scale appropriatelyt
            size=np.log(event_summary['FATALITIES']) * 3,
            color='red'
        ),
        text=event_summary['FATALITIES'],  # Fatalities count as text
        textposition='top center',  # Position text above the markers
        name='Fatalities',
        hovertemplate="<b>%{x}</b><br>Total Fatalities: %{y}<extra></extra>"  # Custom hover text for points

    ))

    # Update the layout to add titles and axis labels
    fig.update_layout(
        title='Impact of Different Conflict Events',
        xaxis=dict(title='Event Type'),
        yaxis=dict(title='Number of Events/Fatalities',
                   range=[0, max(event_summary['COUNT'].max(), event_summary['FATALITIES'].max()) + 50]),
        legend_title='Data Type',
        barmode='overlay'  # Ensures bars and scatter points share the same x-axis
    )

    return fig


# Create the graph with filtered data
G = nx.from_pandas_edgelist(filtered_edges, 'ACTOR1', 'ACTOR2', ['FATALITIES'])

# Add attributes to each node.
for n in G.nodes():
    G.nodes[n]['name'] = n
    G.nodes[n]['fatalities_caused'] = data[data['ACTOR1'] == n]['FATALITIES'].sum()


def draw_graph(G):
    # Position nodes using the spring layout algorithm
    pos = nx.spring_layout(G, seed=22)

    # Draw nodes and edges and show weights
    viz = nxa.draw_networkx(G, pos=pos, edge_color='white',
                            node_color="fatalities_caused",
                            cmap='reds',
                            width='FATALITIES:Q',
                            node_tooltip=['name', 'fatalities_caused'])

    return viz

# fatalities map


def animated_fatalities_map(data):
    # Filter out data points where fatalities are 0
    data_filtered = data[data['FATALITIES'] > 0]

    fig = px.scatter_geo(data_filtered,
                         lat='LATITUDE',
                         lon='LONGITUDE',
                         color='FATALITIES',  # Color points by fatalities number
                         animation_frame='YEAR',
                         projection="natural earth",
                         title='Animated Map of Fatalities by Year',
                         color_continuous_scale='Reds',  # Adjust color scale
                         )
    fig.update_layout(
        autosize=True,
        height=600,
        geo=dict(
            center=dict(lat=47.2, lon=31.1),
            scope='europe',
            projection_scale=3,
            bgcolor='rgba(0, 0, 0, 0)',  # Set background color to transparent
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background color to transparent
        font=dict(color='white'),  # Set font color to white
    )
    return fig


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

# Display the fatalities map
fatalities_map = fatailities_map(data, gdf)
st.plotly_chart(fatalities_map)


st.markdown('### Who is responsible?')

# Call the function to draw the graph
graph_viz = draw_graph(G)

# Use Streamlit components to display the visualization
st.altair_chart(graph_viz, use_container_width=True)

# Display the figures in the Streamlit app

st.markdown('### Nowhere is safe')

animated_geo_fig = animated_map(filtered, gdf)
st.plotly_chart(animated_geo_fig)

# Provide context about the Russia vs Ukraine conflict
st.write("The conflict between Russia and Ukraine witnessed a steady employment of shelling, artillery, and missiles from 2018 to 2021. However, a notable surge in these activities occurred in 2022 and 2023, marked by a significant increase in the frequency and intensity of such attacks.")
st.write("Moreover, the years 2022 and 2023 saw a pronounced rise in the utilization of modern air and drone strikes, suggesting a shift towards more technologically advanced warfare strategies.")
st.write("This trend underscores the evolving nature of the conflict and highlights the increasing reliance on advanced weaponry and aerial capabilities by both parties involved.")


st.markdown("## References")
st.markdown("1. [ACLED](https://acleddata.com/)")
st.markdown("2. [War in Ukraine: Two Years On, Attacks Against Civilians on the Rise Again](https: // reliefweb.int/report/ukraine/war-ukraine-two-years-attacks-against-civilians-rise-again)")
