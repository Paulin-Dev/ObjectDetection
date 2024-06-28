#!/usr/bin/env python

# Built-in imports
from datetime import datetime
import json
from os import listdir
from os.path import dirname, isfile, join, realpath
import time

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Constants
ROOT_DIR = dirname(realpath(__file__))
IMAGES_DIR = '/images'


RESULTS_DIR = join(ROOT_DIR, 'results')


st.set_page_config(
    layout='wide',
    page_title='Dashboard - Grand Lyon',
    page_icon=':bar_chart:',
    initial_sidebar_state='expanded',
)


def load_data() -> pd.DataFrame:
    if isfile(join(RESULTS_DIR, 'data.json')):
        try:
            with open(join(RESULTS_DIR, 'data.json'), 'r') as f:
                data = json.loads(f.read())
        except Exception as e:
            print(f'Error: {e}')
            data = {}
    else:
        data = {}
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.fillna(0)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S')
    return df


# TO-DO: Refactor this...
def main():
    st.title('Object Detection Dashboard - Grand Lyon')       
    st.markdown('###')
    
    df = load_data()
    if df.empty:
        st.error('No data available (yet)')
        time.sleep(10)
        st.rerun()

    st.sidebar.title('Filters')
    
    # Select date
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            label='Start date',
            value=df['timestamp'].min(),
            min_value=df['timestamp'].min(),
            max_value=df['timestamp'].max(),
        )
    with col2:
        end_date = st.date_input(
            label='End date',
            value=df['timestamp'].max(),
            min_value=df['timestamp'].min(),
            max_value=df['timestamp'].max(),
        )
    
    # Select time
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_time = st.time_input(label='Start time', value=datetime.min.time())
    with col2:
        end_time = st.time_input(label='End time', value=datetime.max.time())
        
    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)
    
    # Filter data by date and time
    filtered_df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]

    # Select labels
    columns = df.columns.tolist()
    columns.remove('timestamp')
    selected_columns = st.sidebar.multiselect('Select label(s)', columns, default=columns)
    filtered_labels_df = filtered_df[selected_columns]

    # Display image
    st.sidebar.markdown('#')
    st.sidebar.markdown('#')
    st.sidebar.markdown('#')
    st.sidebar.markdown('---')
    st.sidebar.image(join(IMAGES_DIR, listdir(IMAGES_DIR)[-1]), use_column_width='always')
    
    # Display metrics
    st.metric(label='Objects detected', value=filtered_labels_df.sum().sum())
    st.markdown('---')
    
    
    col1, col2 = st.columns([8, 2])
    with col1:
        st.write('Number of objects detected over time')
        st.line_chart(filtered_df.set_index('timestamp'))
    with col2:
        st.write(filtered_labels_df.sum())

    
    st.markdown('#')
    
    # Proportion of each label
    values = filtered_labels_df.sum()
    labels = filtered_labels_df.columns.tolist()
    
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    # ax1.axis('equal')
    
    # set background color to transparent
    fig1.patch.set_alpha(0.0)
    # set text color to white
    for text in ax1.texts:
        text.set_color('white')
    
    col, _ = st.columns([1, 3])
    with col:
        st.pyplot(fig1, clear_figure=True, use_container_width=True)


    # Refresh every 10 seconds
    time.sleep(10)
    st.rerun()
    


if __name__ == '__main__':
    main()
