import os

import streamlit as st
import pandas as pd
import numpy as np

from helpers import Analyzer

st.set_page_config(layout="wide", page_title="Data Exploration", page_icon=":compass:")

st.markdown("<h1 style='color:#674ea7'>Data Exploration</h1>", unsafe_allow_html=True)
