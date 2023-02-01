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

st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

# sys.path.append('../')
IDF_sample_100000 = pd.read_csv('server_configuration/data_right_price/data_localisee/sample_100000_mutations_IDF_train_localized.csv', index_col='Unnamed: 0')
IDF_sample = pd.read_csv('server_configuration/data_right_price/data_localisee/sample_mutations_IDF_train_localized.csv')
# the-right-price/server_configuration/data_right_price copie/data_localisee/sample_mutations_IDF_train_localized.csv

geo_IDF_sample = gpd.GeoDataFrame(
    IDF_sample, geometry=gpd.points_from_xy(IDF_sample.longitude, IDF_sample.latitude))




st.title('The Right Price')

# Parameters
my_expander = st.expander("Parameters", expanded=True)
with my_expander:
    col1, col2, col3 = st.columns(3)
    with col1:
        dept_slider = st.selectbox(
            'Select a department',
            (None, 75, 77, 78, 91, 92, 93, 94, 95))

    with col2:
        insee_list = IDF_sample[IDF_sample['Dept'] == dept_slider].l_codinsee.apply(lambda x: int(re.findall(r'[0-9]+', x)[0])).unique().tolist() if dept_slider is not None else IDF_sample.l_codinsee.apply(lambda x: int(re.findall(r'[0-9]+', x)[0])).unique().tolist()
        insee_list.sort()
        insee_slider = st.selectbox(
            'Select a municipality',
            [None] + insee_list)
    
    with col3:
        section_list = IDF_sample[IDF_sample['Dept'] == dept_slider].l_section.apply(lambda x: re.findall(r'[0-9A-Z]+', x)[0]).unique().tolist() if dept_slider is not None else IDF_sample.l_section.apply(lambda x: re.findall(r'[0-9A-Z]+', x)[0]).unique().tolist()
        section_list.sort()
        section_slider = st.selectbox(
            'Select a section',
            [None] + section_list)

    property_list = IDF_sample[IDF_sample['Dept'] == dept_slider].libtypbien.unique().tolist() if dept_slider is not None else IDF_sample.libtypbien.unique().tolist()
    property_slider = st.selectbox(
        'Select a type of property',
        [None] + property_list)


    i = IDF_sample['valeurfonc'].min()
    a = 10**(len(str(int(i)))-1)
    min_value = a*np.floor(i/a) if i > 1 else 0.0
    j = IDF_sample['valeurfonc'].max()
    b = 10**(len(str(int(j)))-1)
    max_value = b*np.ceil(j/b) if j > 1 else 0.0

    luxury = st.checkbox('Check this box to see the most expansive properties')

    if luxury:
        min_value_slider, max_value_slider = st.select_slider(
            'Select a property value', 
            np.arange(1500000, max_value+1, 1000), 
            value=(1500000, max_value))
    else:
        min_value_slider, max_value_slider = st.select_slider(
            'Select a property value', 
            np.arange(min_value, 1500001, 1000), 
            value=(min_value, 1500000))


    k = IDF_sample['sbati'].min()
    c = 10**(len(str(int(k)))-1)
    min_surface = c*np.floor(k/c) if k > 1 else 0.0
    l = IDF_sample['sbati'].max()
    d = 10**(len(str(int(l)))-1)
    max_surface = d*np.ceil(l/d) if l > 1 else 0.0

    if luxury:
        min_surface_slider, max_surface_slider = st.select_slider(
            'Select a surface', 
            np.arange(min_surface, max_surface+1, 10), 
            value=(min_surface, max_surface))
    else:
        min_surface_slider, max_surface_slider = st.select_slider(
            'Select a surface', 
            np.arange(min_surface, 1501, 10), 
            value=(min_surface, 1500))
    
    min_year_slider, max_year_slider = st.select_slider(
            'Select a year', 
            range(2014, 2021), 
            value=(2014, 2020))

plotting_functions = plotting_functions(dept=dept_slider, 
                                            insee=insee_slider, 
                                            section=section_slider, 
                                            type_of_property=property_slider, 
                                            min_value=min_value_slider, max_value=max_value_slider,
                                            min_surface=min_surface_slider, max_surface=max_surface_slider,
                                            min_year=min_year_slider, max_year=max_year_slider
                                           )

# Map
folium_map = plotting_functions.plot_map(geo_IDF_sample 
                                        )

st.header('Map of all mutations')
if folium_map is None:
    st.subheader('There are no mutations that meet your criterias in the selected sample.')
else:
    folium_static(folium_map, width=1050, height=750)

    
# Graph 
st.header('A few temporal figures')

fig = plotting_functions.plot_yearly_figures(IDF_sample_100000)
st.plotly_chart(fig, use_container_width=True)
