import streamlit as st
import json
import requests
import csv
import pandas as pd
import tqdm

import cohere
import os


from topically import Topically
from bertopic import BERTopic
from sklearn.cluster import KMeans
import numpy as np

venue = 'ICLR.cc/2023/Conference'
venue_short = 'iclr2023'

def get_conference_notes(venue, blind_submission=False):
    """
    Get all notes of a conference (data) from OpenReview API.
    If results are not final, you should set blind_submission=True.
    """

    blind_param = '-/Blind_Submission' if blind_submission else ''
    offset = 0
    notes = []
    while True:
        print('Offset:', offset, 'Data:', len(notes))
        url = f'https://api.openreview.net/notes?invitation={venue}/{blind_param}&offset={offset}'
        response = requests.get(url)
        data = response.json()
        if len(data['notes']) == 0:
            break
        offset += 1000
        notes.extend(data['notes'])
    return notes

raw_notes = get_conference_notes(venue, blind_submission=True)


st.header("ICLR2023 Papers Visualization: Using Topically")
st.write("Number of submissions at ICLR 2023:", len(raw_notes))

df_raw = pd.json_normalize(raw_notes)
# set index as first column
# df_raw.set_index(df_raw.columns[0], inplace=True)
accepted_venues = ['ICLR 2023 poster', 'ICLR 2023 notable top 5%', 'ICLR 2023 notable top 25%']
df = df_raw[df_raw["content.venue"].isin(accepted_venues)]
st.write("Number of submissions accepted at ICLR 2023:", len(df))

df_filtered = df[['content.title', 'content.keywords', 'content.abstract', 'content.venue']]
df = df_filtered

CO_API_KEY = st.secrets["CO_API_KEY"]

co = cohere.Client(CO_API_KEY)

def to_html(df: pd.DataFrame, table_header: str) -> str:
        table_data = ''.join(df.html_table_content)
        html = f'''
        <table>
            {table_header}
            {table_data}
        </table>'''
        return html


def get_visualizations():
    table_header = '''
            <tr>
                <td width="25%">Title</td>
                <td width="15%">Keywords</td>
                <td width="10%">Venue</td>
                <td width="50%">Abstract</td>
            </tr>'''
    list_of_titles = list(df["content.title"].values)
    embeds = co.embed(texts=list_of_titles,                  				
  					model="small").embeddings
    
    embeds_npy = np.array(embeds)
    
    # Load and initialize BERTopic to use KMeans clustering with 8 clusters only.
    cluster_model = KMeans(n_clusters=8)
    topic_model = BERTopic(hdbscan_model=cluster_model)
    
    # df is a dataframe. df['title'] is the column of text we're modeling
    df['topic'], probabilities = topic_model.fit_transform(df['content.title'], embeds_npy)
    
    app = Topically(CO_API_KEY)
    
    df['topic_name'], topic_names = app.name_topics((df['content.title'], df['topic']), num_generations=5)
    
    #st.write("Topics extracted are:", topic_names)
    
    topic_model.set_topic_labels(topic_names)
    fig1 = topic_model.visualize_documents(df['content.title'].values, 
                                    embeddings=embeds_npy,
                                    topics = list(range(8)),
                                    custom_labels=True)
    topic_model.set_topic_labels(topic_names)
    fig2 = topic_model.visualize_barchart(custom_labels=True)
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)


st.button("Run Visualization", on_click=get_visualizations)