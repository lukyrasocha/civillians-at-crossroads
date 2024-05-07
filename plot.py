import json
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import networkx as nx
import nx_altair as nxa
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# Altair Visualization: Event Type Distribution


def plot_event_type_distribution(data):
    # plot the distribution of the population 1 km, and start the x-axis from 0

    chart = alt.Chart(data, width=400).mark_bar().encode(
        y='count()',
        x=alt.X('EVENT_TYPE', sort='-y'),
        color='EVENT_TYPE',
        tooltip=['EVENT_TYPE', 'count()']
    ).properties(title='Event Type Distribution')

    return chart

# Plot distribution of POPULATION_1KM to see in total the number of events happen in the area


def plot_population_distribution(data):
    chart = alt.Chart(data, width=680).mark_bar().encode(
        y='count()',
        x=alt.Y('POPULATION_1KM', sort='-x'),
        tooltip=['POPULATION_1KM', 'count()']
    ).properties(title='Population Distribution')

    # change colour of bars to slategrey
    chart = chart.configure_mark(color='white')

    return chart

# Use the timestamp to plot the events occurences per each hour of the day


def plot_event_times(data):
    # USE THE TIMESTAMP COLUMN TO PLOT THE EVENTS OCCURENCES PER EACH HOUR OF THE DAY
    # Convert TIMESTAMP to datetime if it's in Unix time (assuming it is)
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='s')

    # Extract the hour from each timestamp
    data['HOUR'] = data['TIMESTAMP'].dt.hour
    hourly_counts = data.groupby('HOUR').size().reset_index(name='COUNT')

    # Create a bar chart
    fig = px.bar(hourly_counts, x='HOUR', y='COUNT',
                 title='Event Occurrences by Hour of the Day',
                 labels={'HOUR': 'Hour of the Day', 'COUNT': 'Number of Events'},
                 color='COUNT',
                 color_continuous_scale=px.colors.sequential.OrRd)  # px.colors.sequential.Viridis)  # Use a color scale for visual appeal

    return fig


# Plot event times as a calendar heatmap where on x-axis we have the day of the week and on y-axis the hour of the day

def plot_event_times_heatmap(data):
    # Extract the day of the week and hour of the day from the timestamp
    data['DAY'] = data['TIMESTAMP'].dt.day_name()
    data['HOUR'] = data['TIMESTAMP'].dt.hour

    # Group by day and hour to count occurrences
    day_hour_counts = data.groupby(['DAY', 'HOUR']).size().reset_index(name='COUNT')

    # Create a calendar heatmap
    fig = px.imshow(day_hour_counts.pivot('DAY', 'HOUR', 'COUNT'),
                    labels=dict(color='Number of Events'),
                    color_continuous_scale=px.colors.sequential.OrRd,
                    title='Event Occurrences by Day and Hour of the Day')

    return fig

# Evolution of protests over time


def plot_protests_over_time(data):
    # Filter the data for protests
    protests = data[data['EVENT_TYPE'] == 'Protests']

    # Group by year and month to count occurrences
    date_counts = protests.groupby([protests['EVENT_DATE'].dt.to_period('M')]).size().reset_index(name='COUNT')
    date_counts['EVENT_DATE'] = date_counts['EVENT_DATE'].dt.to_timestamp()

    # Create a line plot with larger width
    fig = px.line(date_counts, x='EVENT_DATE', y='COUNT',
                  title='Trends in Protests Over Time',
                  labels={'EVENT_DATE': 'Date', 'COUNT': 'Number of Protests'})

    fig.update_traces(line=dict(width=1))

    # update colour
    fig.update_traces(line=dict(color='white'))

    return fig


def plot_average_population(data):
    # Calculate the average population best for each sub-event type
    # Where event type is not Protest

    data = data[data['EVENT_TYPE'] != 'Protest']
    # where sub event type does not contain 'Protest' or 'demonstration
    data = data[~data['SUB_EVENT_TYPE'].str.contains('Protest|demonstration|protest|Mob|Agreement')]

    average_population_distance = data.groupby('SUB_EVENT_TYPE')['POPULATION_1KM'].mean().reset_index()
    average_population_distance.sort_values('POPULATION_1KM', ascending=True, inplace=True)

    # Round the average population distance to 0 decimal places
    average_population_distance['POPULATION_1KM'] = average_population_distance['POPULATION_1KM'].round(0)

    # Include only non-zero values
    average_population_distance = average_population_distance[average_population_distance['POPULATION_1KM'] > 0]

    # Create a bar chart
    fig = px.bar(average_population_distance, x='SUB_EVENT_TYPE', y='POPULATION_1KM',
                 title='Average Proximity of Sub-Event Types to Populations',
                 labels={'POPULATION_BEST': 'Average Population Distance (Best Estimate)',
                         'SUB_EVENT_TYPE': 'Sub-Event Type'})

    # change colour of bars to slategrey
    fig.update_traces(marker_color='slategrey')

    return fig


def plot_violence_against_civilians(data):
    # Filter the data for where EVENT_TYPE is 'Violence against civilians' or CIVILIAN_TARGETING is equal to Civilian targeting
    violence_civilians = data[(data['EVENT_TYPE'] == 'Violence against civilians') |
                              (data['CIVILIAN_TARGETING'] == 'Civilian targeting')]

    # Convert EVENT_DATE to datetime if it hasn't been done already
    violence_civilians['EVENT_DATE'] = pd.to_datetime(violence_civilians['EVENT_DATE'])

    # Filter where year is 2021 >
    violence_civilians = violence_civilians[violence_civilians['YEAR'] >= 2021]

    # Group by date and sub-event type to count occurrences
    date_sub_event_counts = violence_civilians.groupby(
        [violence_civilians['EVENT_DATE'].dt.to_period('M'),
         'SUB_EVENT_TYPE']).size().reset_index(
        name='COUNT')
    # Convert back to timestamp for plotting
    date_sub_event_counts['EVENT_DATE'] = date_sub_event_counts['EVENT_DATE'].dt.to_timestamp()

    fig = px.line(
        date_sub_event_counts,
        x='EVENT_DATE',
        y='COUNT',
        color='SUB_EVENT_TYPE',  # Different lines for different sub-event types
        title='Trends in Violence Against Civilians by Sub-Event Type',
        labels={'EVENT_DATE': 'Date', 'COUNT': 'Number of Occurrences', 'SUB_EVENT_TYPE': 'Category'}
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Occurrences",
        legend_title="Category"
    )

    return fig


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

# fatalities map


def fatailities_map(data, gdf):
    data = data[data['FATALITIES'] > 0]
    # order years
    data.sort_values('YEAR', inplace=True)

    scatter_geo = px.scatter_geo(
        data, lat='LATITUDE', lon='LONGITUDE', color='FATALITIES',
        size='FATALITIES',
        animation_frame='YEAR',
        title='Evolution of all Fatalities',
        size_max=15,
        color_continuous_scale=px.colors.sequential.OrRd
    )
    scatter_geo.update_layout(
        autosize=True,
        height=700,
        geo=dict(
            center=dict(lat=47.2, lon=31.1),
            scope='europe',
            projection_scale=6,
            bgcolor='rgba(0, 0, 0, 0)',
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
    )

    return scatter_geo

# Animation map with region geo info


def animated_map(data, gdf):

    data = data[(data['EVENT_TYPE'] == 'Violence against civilians') |
                (data['CIVILIAN_TARGETING'] == 'Civilian targeting')]

    min_year = data['YEAR'].min()
    max_year = data['YEAR'].max()

    all_event_types = data['SUB_EVENT_TYPE'].unique()

    dummy_rows = []

    # Loop through each event type
    for event_type in all_event_types:
        for year in range(min_year, max_year + 1):
            # Check if there are any rows for the current event type and year
            if len(data[(data['SUB_EVENT_TYPE'] == event_type) & (data['YEAR'] == year)]) == 0:
                # If no rows exist, create a dummy row
                dummy_rows.append({
                    'YEAR': year,
                    'SUB_EVENT_TYPE': event_type,
                    'LATITUDE': 0,
                    'LONGITUDE': 0,
                    'EVENT_ID_CNTY': 'dummy'
                })
    # Concatenate the dummy rows with the original filtered_data DataFrame
    dummy_df = pd.DataFrame(dummy_rows)
    data = pd.concat([data, dummy_df], ignore_index=True)
    # order years
    data.sort_values('YEAR', inplace=True)

    # Create a choropleth map
    choropleth = go.Figure(go.Choropleth(
        geojson=json.loads(gdf.to_json()),
        locations=gdf.index,
        z=[0] * len(gdf),
        hovertext=gdf['name'],  # Add hover text
        hoverinfo='text',  # Show only hover text
        marker_line=dict(width=1, color='grey'),  # Set marker line width to 1 to display only the outline
        showscale=False
    ))

    # Create a scatter_geo plot using Plotly Express
    scatter_geo = px.scatter_geo(
        data, lat='LATITUDE', lon='LONGITUDE', color='SUB_EVENT_TYPE',
        title='Political Violence Against Civilian in the Black Sea Region',
        hover_data={'LATITUDE': False, 'LONGITUDE': False, 'YEAR': False, 'SUB_EVENT_TYPE': False})

    # Add the choropleth traces to the scatter_geo plot
    for trace in choropleth.data:
        scatter_geo.add_trace(trace)

    # Update layout for the combined plot
    scatter_geo.update_layout(
        autosize=True,
        height=800,
        geo=dict(
            center=dict(lat=47.2, lon=31.1),
            scope='europe',
            projection_scale=5.5,
            bgcolor='rgba(0, 0, 0, 0)',  # Set background color to transparent
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background color to transparent
        font=dict(color='white'),  # Set font color to white
    )

    return scatter_geo


def create_strip_plot(data):
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='s')

    # Calculate the minute of the day
    data['MINUTE_OF_DAY'] = data['TIMESTAMP'].dt.hour * 60 + data['TIMESTAMP'].dt.minute
    # Create a figure with a specified figure size and facecolor
    plt.figure(figsize=(10, 6))
    plt.style.use({'figure.facecolor': '#0f1116'})
    plt.rcParams["font.family"] = "serif"

    # Adjust axes appearances
    ax = plt.axes()
    ax.set_facecolor("#0f1116")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Create a strip plot with adjusted point styles
    sns.stripplot(x='MINUTE_OF_DAY', data=data, jitter=0.35, color="white", linewidth=0.1, marker='o', size=1.5)

    # Set labels and titles with custom colors (optional)
    # plt.title('Event Occurrences Throughout the Day', color='white')
    plt.xlabel('Time of Day', color='white')
    # plt.ylabel('Event Count', color='white')

    # Set x-axis ticks to display hours, with custom colors
    hour_ticks = range(0, 1441, 60)  # From minute 0 to 1440 (24 hours), step by 60 minutes
    hour_labels = [f'{i}:00' for i in range(25)]  # Generate labels from '0:00' to '24:00'
    plt.xticks(hour_ticks, hour_labels, rotation=45, color='white')

    # Adjust tick colors for both axes
    plt.tick_params(axis='both', colors='white')

    # Ensure layout is tight so everything fits
    plt.tight_layout()

    # Instead of plt.show(), return the figure to be used by Streamlit
    return plt.gcf()


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
        title='Impact of Different Conflict Events on Fatalities from 2018 until the present',
        xaxis=dict(title='Event Type'),
        yaxis=dict(title='Number of Events/Fatalities',
                   range=[0, max(event_summary['COUNT'].max(), event_summary['FATALITIES'].max()) + 50]),
        legend_title='Data Type',
        barmode='overlay'  # Ensures bars and scatter points share the same x-axis
    )

    return fig


# correlation plot between population and fatalities
def plot_correlation(data):
    # Check for missing values and fill or remove them
    data['FATALITIES'].fillna(0, inplace=True)
    data['POPULATION_1KM'].fillna(data['POPULATION_1KM'].mean(), inplace=True)
    data['POPULATION_2KM'].fillna(data['POPULATION_2KM'].mean(), inplace=True)
    data['POPULATION_5KM'].fillna(data['POPULATION_5KM'].mean(), inplace=True)

    # Create scatter plots
    fig = px.scatter(data, x='POPULATION_1KM', y='FATALITIES', trendline='ols',
                     title='Fatalities vs. Population within 1km')

    return fig


def draw_graph(G):
    # Position nodes using the spring layout algorithm
    pos = nx.spring_layout(G, seed=22)

    # Draw nodes and edges and show weights
    viz = nxa.draw_networkx(G, pos=pos, edge_color='white',
                            node_color="Fatalities caused",
                            cmap='oranges',
                            width='FATALITIES:Q',
                            node_tooltip=['name', 'Fatalities caused'])

    return viz
