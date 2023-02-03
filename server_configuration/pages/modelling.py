import os

import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image

from helpers import modelling

import warnings

warnings.simplefilter('ignore', FutureWarning)

st.set_page_config(layout="wide", page_title="Modelling", page_icon=":compass:")

st.title("Modelling")

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

lat_long = pd.DataFrame([[1, 48.8637, 2.332],
                        [2, 48.8676, 2.344],
                        [3, 48.8635, 2.3591],
                        [4, 48.8534, 2.3583],
                        [5, 48.8435, 2.3518],
                        [6, 48.8505, 2.3322],
                        [7, 48.8543, 2.3134], 
                        [8, 48.8742, 2.3111],
                        [9, 48.8783, 2.337],
                        [10, 48.8755, 2.3579],
                        [11, 48.858, 2.3812],
                        [12, 48.8408, 2.3882],
                        [13, 48.8322, 2.3556],
                        [14, 48.833, 2.3269],
                        [15, 48.8413, 2.3003],
                        [16, 48.8531, 2.2488],
                        [17, 48.8822, 2.3078],
                        [18, 48.8922, 2.3444],
                        [19, 48.8824, 2.3818],
                        [20, 48.8651, 2.399],
                        [75, 48.8786, 2.3642],
                        [77, 48.5421, 2.6554],
                        [78, 48.8014, 2.1301],
                        [91, 48.6298, 2.4418],
                        [92, 48.8924, 2.2153],
                        [93, 48.91, 2.4222],
                        [94, 48.7904, 2.4556],
                        [95, 49.0332, 2.0547]], 
                        columns=['location', 'latitude', 'longitude']).set_index('location')


modelling_ = modelling()
reg = modelling_.load_model("./server_configuration/saved_model_bis.pkl")
scaler = modelling_.load_model("./server_configuration/saved_scaler.pkl")

col1, col2 = st.columns(2)
with col1:
    year = st.number_input('Insert a year', value=2023)
with col2:
    month = st.number_input('Insert a month', value=1)
    
col3, col4 = st.columns(2)
with col3:
    dept = st.number_input('Insert a department', value=75)
with col4:
    arro = st.number_input('Insert an INSEE code (make sure it is located in the department you selected)', value=75118)
    
col5, col6 = st.columns(2)
with col5:
    surface = st.number_input('Insert a surface (m²)', value=50)
with col6:
    nb_rooms = st.number_input('Insert the number of rooms', value=3)
    
vefa = st.checkbox('Sale in the future state of acquisition', value=False)
    
columns = ['anneemut', 'moismut', 'coddep', 'vefa', 'sterr', 'nblocdep',
       'latitude', 'longitude', 'nb_accomodations', 'surf', 'arro',
       'nb_rooms']

df_to_predict = pd.DataFrame([[year, month, dept, vefa, 0, 0., lat_long.at[int(str(arro)[-2:]) if int(str(arro)[-2:]) <= 20 else dept, 'latitude'], lat_long.at[int(str(arro)[-2:]) if int(str(arro)[-2:]) <= 20 else dept, 'longitude'], # 48.96152015, 1.79976425,
          1., surface, int(str(arro)[-2:]), nb_rooms]], columns=columns)

col5, col6, col7 = st.columns((1, 3, 1))
with col6:
    txt = "The estimated price for this real estate is {valfonc:.2f}€"
    st.subheader(txt.format(valfonc = modelling_.min_max_rescale(modelling_.predict(reg, df_to_predict), scaler)[0][0]))
    
col8, col9, col10 = st.columns((1, 5, 1))
with col9:
    image = Image.open('./server_configuration/51b9032a-c1c9-48ad-b33a-0bf4cee7a62c.png')
    st.image(image)
    st.markdown("<p style='text-align: center; color: black;'>Interpretability of the model <br> eg. The surface and the location (longitude, latitude, arrondissement) have the biggest impact on the value of the real estate. <br> eg. When the latitude increases, the price increases while when the longitude increases, the price decreases.</p>", unsafe_allow_html=True)




