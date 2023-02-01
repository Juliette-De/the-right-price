import streamlit as st

import pandas as pd
import numpy as np
import re

import geopandas as gpd
from streamlit_folium import folium_static

import sys

from helpers import plotting_functions

import warnings

warnings.simplefilter('ignore', FutureWarning)

st.set_page_config(page_title="Home", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

st.sidebar.title("Contact Us!")
st.sidebar.info(
    """
    cesar.bareau@hec.edu
    augustin.de-la-brosse@hec.edu
    clement.giraud@hec.edu
    juliette.demoly@hec.edu
    ran.ding@hec.edu
    mojun.guo@hec.edu
    """
)

# sys.path.append('../')
IDF_sample_100000 = pd.read_csv('server_configuration/data_right_price/data_localisee/sample_100000_mutations_IDF_train_localized.csv', index_col='Unnamed: 0')
IDF_sample = pd.read_csv('server_configuration/data_right_price/data_localisee/sample_mutations_IDF_train_localized.csv')
# the-right-price/server_configuration/data_right_price copie/data_localisee/sample_mutations_IDF_train_localized.csv

geo_IDF_sample = gpd.GeoDataFrame(
    IDF_sample, geometry=gpd.points_from_xy(IDF_sample.longitude, IDF_sample.latitude))




st.title('The Right Price')


st.markdown(
    """
    # 💻 Welcome to your newest tool to predict the right price of properties

    Navigate through the **sidebar tabs** to explore how the real estate market is going and how you can leverage our AI algorithm to make the wisest choices and buy the right property at the right price.
    
    ## Want to know more about our features?
    - [Explore the real estate market](Explore_the_real_estate_market) to visualize the real estate market in Île-de-France and analyze the different market trends according to the parameters you choose.
    - [Modelling](Modelling) to predict the price of a real estate.
    - [Leverage our model](Leveraging_our_model) to save time by seeing directly the real estate with the post potential.
    
    
    
    """
)