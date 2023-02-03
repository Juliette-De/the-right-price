"""Page 3"""


import os

import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
import numpy as np

import geopandas as gpd
import re

from helpers import plotting_functions

import warnings

warnings.simplefilter('ignore', FutureWarning)

st.set_page_config(layout="wide", page_title="Leveraging our model", page_icon=":compass:")

st.title("Leverage our model")

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

X_test = pd.read_csv('server_configuration/data_right_price/data_localisee/sample_predictions.csv')

geo_X_test = gpd.GeoDataFrame(
    X_test, geometry=gpd.points_from_xy(X_test.longitude_x, 
                                        X_test.latitude_x))

# Parameters
expander_1 = st.expander("Parameters", expanded=True)
with expander_1:
    col1, col2, col3 = st.columns(3)
    with col1:
        dept_slider = st.selectbox(
            'Select a department',
            (None, 75, 77, 78, 91, 92, 93, 94, 95))

    with col2:
        insee_list = X_test[X_test['Dept'] == dept_slider].l_codinsee.apply(lambda x: int(re.findall(r'[0-9]+', x)[0])).unique().tolist() if dept_slider is not None else X_test.l_codinsee.apply(lambda x: int(re.findall(r'[0-9]+', x)[0])).unique().tolist()
        insee_list.sort()
        insee_slider = st.selectbox(
            'Select a municipality',
            [None] + insee_list)

    with col3:
        section_list = X_test[X_test['Dept'] == dept_slider].l_section.apply(lambda x: re.findall(r'[0-9A-Z]+', x)[0]).unique().tolist() if dept_slider is not None else X_test.l_section.apply(lambda x: re.findall(r'[0-9A-Z]+', x)[0]).unique().tolist()
        section_list.sort()
        section_slider = st.selectbox(
            'Select a section',
            [None] + section_list)

    property_list = X_test[X_test['Dept'] == dept_slider].libtypbien.unique().tolist() if dept_slider is not None else X_test.libtypbien.unique().tolist()
    property_slider = st.selectbox(
        'Select a type of property',
        [None] + property_list)


    k = X_test['sbati'].min()
    c = 10**(len(str(int(k)))-1)
    min_surface = c*np.floor(k/c) if k > 1 else 0.0
    l = X_test['sbati'].max()
    d = 10**(len(str(int(l)))-1)
    max_surface = d*np.ceil(l/d) if l > 1 else 0.0

    luxury_bool = st.checkbox('Check this box to see the most expansive properties')
    
    if luxury_bool:
        min_surface_slider, max_surface_slider = st.select_slider(
            'Select a surface', 
            np.arange(min_surface, max_surface+1, 10), 
            value=(min_surface, max_surface))
    else:
        min_surface_slider, max_surface_slider = st.select_slider(
            'Select a surface (m²)', 
            np.arange(min_surface, 1501, 10), 
            value=(min_surface, 1500))

plotting_functions_ = plotting_functions(dept=dept_slider, 
                                         insee=insee_slider, 
                                         section=section_slider, 
                                         type_of_property=property_slider, 
                                         min_surface=min_surface_slider, max_surface=max_surface_slider
                                        )

# Map
folium_map = plotting_functions_.plot_prediction_map(geo_X_test,
                                         luxury=luxury_bool)

st.header('Map of the best opportunities')
if folium_map is None:
    st.subheader('There are no mutations that meet your criterias in the selected sample.')
else:
    folium_static(folium_map, width=1050, height=750)
