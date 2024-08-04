import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Welvision Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Welvision Dashboard")

# Load data
df_reshaped = pd.read_csv('data.csv')

# Add Quarter column
df_reshaped['Date'] = pd.to_datetime(df_reshaped['Date']).dt.date
df_reshaped['Quarter'] = pd.PeriodIndex(pd.to_datetime(df_reshaped['Date']), freq='Q')

# Sidebar
with st.sidebar:
    st.title('Dashboard Slicer')
    
    # Analysis type toggle
    analysis_type = st.radio('Select Analysis Type', ('Monthly', 'Quarterly'))
    
    if analysis_type == 'Monthly':
        # Date slicer
        Date_list = list(df_reshaped['Date'].unique())[::-1]
        selected_date = st.selectbox('Select a month', Date_list)
        df_selected = df_reshaped[df_reshaped['Date'] == selected_date]
    else:
        # Quarter slicer
        Quarter_list = list(df_reshaped['Quarter'].unique())
        selected_quarter = st.selectbox('Select a quarter', Quarter_list)
        df_selected = df_reshaped[df_reshaped['Quarter'] == selected_quarter]

    df_selected_sorted = df_selected.sort_values(by="Quantity", ascending=False)

# Calculate metrics
ok_rollers = df_selected[df_selected['Class Name'] == 'Rollers']['Quantity'].sum()
highest_defect_rollers = df_selected[df_selected['Class Name'] != 'Rollers'].sort_values(by="Quantity", ascending=False).iloc[0]
highest_defect_rollers_quantity = highest_defect_rollers['Quantity']

# Display metrics
tile_style = """
    <style>
    .metric-tile {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 130px;
        height: 100px;
    }
    .metric-title {
        font-size: 10px;
        color: white;
    }
    .metric-value {
        font-size: 40px;
        font-weight: bold;
        color: #29b5e8;
    }
    .metric-heading {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
"""
st.markdown(tile_style, unsafe_allow_html=True)

def display_metric(title, value):
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True
    )

# Prepare data for plotting
if analysis_type == 'Monthly':
    df_plot = df_reshaped.copy()
else:
    df_plot = df_reshaped.copy()
    df_plot['Date'] = df_plot['Quarter'].astype(str)

# Aggregate OK Rollers and Highest Defect Quantities
df_monthly = df_plot.groupby('Date').agg({
    'Quantity': lambda x: x[df_plot['Class Name'] == 'Rollers'].sum() if 'Rollers' in df_plot['Class Name'].values else 0
}).reset_index()

# Find highest defect for each period
highest_defect_per_period = df_plot[df_plot['Class Name'] != 'Rollers'].groupby('Date').agg({
    'Quantity': 'max'
}).reset_index()

# Rename columns for clarity
df_monthly.columns = ['Date', 'Quantity_Rollers']
highest_defect_per_period.columns = ['Date', 'Quantity_Defect']

# Merge dataframes for plotting
df_plot = pd.merge(df_monthly, highest_defect_per_period, on='Date')

# Create line chart
fig_line = go.Figure()

# Add OK Rollers line
fig_line.add_trace(go.Scatter(
    x=df_plot['Date'],
    y=df_plot['Quantity_Rollers'],
    mode='lines+markers',
    name='OK Rollers',
    line=dict(color='blue')
))

# Add Highest Defect line
fig_line.add_trace(go.Scatter(
    x=df_plot['Date'],
    y=df_plot['Quantity_Defect'],
    mode='lines+markers',
    name='Highest Defect',
    line=dict(color='red')
))

# Update layout
fig_line.update_layout(
    title=f'{analysis_type} OK Rollers vs. Highest Defect',
    xaxis_title='Period',
    yaxis_title='Quantity',
    template='plotly_dark'
)

# Create bar chart
fig_bar = px.bar(
    df_selected,
    x='Class Name',
    y='Quantity',
    title=f'{analysis_type} Quantity of Each Class',
    template='plotly_dark'
)

# Create pie chart for OK Rollers vs. Highest Defect Rollers
fig_pie = px.pie(
    names=['OK Rollers', 'Defect rollers'],
    values=[ok_rollers, highest_defect_rollers_quantity],
    title='Proportion of OK Rollers',
    template='plotly_dark',
    color_discrete_map={'OK Rollers': 'green', 'Defect rollers': 'red'}
)

fig_pie.update_traces(textposition='inside', textinfo='percent+label')
fig_pie.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5
    )
)

# Display plot in columns
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("<div class='metric-heading'>Metrics</div>", unsafe_allow_html=True)
    display_metric("OK Rollers", ok_rollers)
    display_metric(f"Highest Defect: {highest_defect_rollers['Class Name']}", highest_defect_rollers_quantity)
    
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.plotly_chart(fig_line, use_container_width=True)
    st.plotly_chart(fig_bar, use_container_width=True)
