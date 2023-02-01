import pandas as pd
import numpy as np

import folium
import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st


class plotting_functions():
    
    def __init__(self, dept=None, insee=None, section=None, type_of_property=None, max_value=None, min_value=None, min_surface=None, max_surface=None, min_year=None, max_year=None):
        self.dept = dept
        self.insee = insee
        self.section = section
        self.type_of_property = type_of_property
        self.max_value = max_value
        self.min_value = min_value
        self.max_surface = max_surface
        self.min_surface = min_surface
        self.min_year = min_year
        self.max_year = max_year
        
    
    def plot_map(self, geo_df):

        """
        Plots a map showing the locations of properties based on the given parameters

        Args:
        geo_df (geopandas.geodataframe.GeoDataFrame): A dataframe containing property information.
        dept (int, optional): Department number. Defaults to None.
        insee (int, optional): Municipality code. Defaults to None.
        section (str, optional): Property section. Defaults to None.
        type_of_property (str, optional): Type of property. Defaults to None.
        year_of_mutation (int, optional): Year of mutation. Defaults to None.
        max_value (int, optional): Maximum value of property to filter data by, default is None
        min_value (int, optional): Minimum value of property to filter data by, default is None
        min_surface (int, optional): Minimum surface of property to filter data by, default is None
        max_surface (int, optional): Maximum surface of property to filter data by, default is None

        Returns:
        None: If no data meets criteria or if municipality is not in selected department
        folium.Map: Interactive map of filtered properties with markers showing property details
        """


        if self.dept is not None and self.insee is not None:
            if np.floor(self.insee/1000) != self.dept:
                print("The municipality you chose is not in the department you chose. Please change either the department or the municipality.")
                return None

        if self.dept is not None: 
            geo_zone = geo_df[geo_df['Dept'] == float(self.dept)]
        else:
            geo_zone = geo_df

        if self.insee is not None: 
            geo_zone = geo_zone[geo_zone['l_codinsee'].apply(lambda x: str(self.insee) in re.findall(r'[0-9]+', x))]

        if self.section is not None:
            geo_zone = geo_zone[geo_zone['l_section'].apply(lambda x: str(self.section) in re.findall(r'[0-9-A-Z]+', x))]

        if self.type_of_property is not None:
            geo_zone = geo_zone[geo_zone['libtypbien'] == self.type_of_property]

        if self.max_year is not None and self.min_year is not None:
            if self.min_year <= self.max_year:
                geo_zone = geo_zone[(geo_zone['year'] >= self.min_year) & (geo_zone['year'] <= self.max_year)]
            else:
                print("The maximum value cannot be lower than the minimum value. Please change these creteria.")
                return None
        elif self.max_year is None and self.min_year is not None:
            geo_zone = geo_zone[geo_zone['year'] >= self.min_year]
        elif self.max_year is not None and self.min_year is None:
            geo_zone = geo_zone[geo_zone['year'] <= self.max_year]

        if self.max_value is not None and self.min_value is not None:
            if self.min_value <= self.max_value:
                geo_zone = geo_zone[(geo_zone['valeurfonc'] >= self.min_value) & (geo_zone['valeurfonc'] <= self.max_value)]
            else:
                print("The maximum value cannot be lower than the minimum value. Please change these creteria.")
                return None
        elif self.max_value is None and self.min_value is not None:
            geo_zone = geo_zone[geo_zone['valeurfonc'] >= self.min_value]
        elif self.max_value is not None and self.min_value is None:
            geo_zone = geo_zone[geo_zone['valeurfonc'] <= self.max_value]

        if self.min_surface is not None and self.max_surface is not None:
            if self.min_surface <= self.max_surface:
                geo_zone = geo_zone[(geo_zone['sbati'] >= self.min_surface) & (geo_zone['sbati'] <= self.max_surface)]
            else:
                print("The maximum surface cannot be lower than the minimum surface. Please change these creteria.")
                return None
        elif self.max_surface is None and self.min_surface is not None:
            geo_zone = geo_zone[geo_zone['sbati'] >= self.min_surface]
        elif self.max_surface is not None and self.min_surface is None:
            geo_zone = geo_zone[geo_zone['sbati'] <= self.max_surface]


        geo_df_list = [(i[0], i[1]) for i in geo_zone[['latitude', 'longitude']].values]
        if len(geo_df_list) == 0:
            print("There are no mutation that meet the criteria you have selected. Please change these creteria.")
            return None    


        map = folium.Map(location=[geo_zone.latitude.median(), geo_zone.longitude.median()], tiles="OpenStreetMap")
        for i, coordinates in enumerate(geo_df_list):
            map.add_child(
                folium.Marker(
                    location=coordinates,
                    popup=
                        "Nature de la mutation: " + str(geo_zone.libnatmut.values[i]) + "<br>" + "<br>"
                        "Valeur foncière: " + str(geo_zone.valeurfonc.values[i]) + "<br>" + "<br>"
                        f"Date de la mutation: {geo_zone.day.values[i]}/{geo_zone.month.values[i]}/{geo_zone.year.values[i]}" + "<br>" + "<br>"
                        "Type de bien: " + str(geo_zone.libtypbien.values[i]) + "<br>" + "<br>"
                        f"Surface du bien: {str(geo_zone.sbati.values[i])} m^2" + "<br>" + "<br>"
                        "CODE INSEE: " + ', '.join(re.findall(r'[0-9]+', geo_zone.l_codinsee.values[i])) + "<br>" + "<br>"
                        "Section: " + ', '.join(re.findall(r'[0-9A-Z]+', geo_zone.l_section.values[i])) + "<br>" # + "<br>"
                        #+ "Coordinates: " + str(geo_df_list[i]),
                )
            )

        sw = geo_zone[['latitude', 'longitude']].min().values.tolist()
        ne = geo_zone[['latitude', 'longitude']].max().values.tolist()

        map.fit_bounds([sw, ne])

        return map


    def plot_yearly_figures(self, df):

        # IDF_filtered = df[ \
        #                   (df['valeurfonc'] >= (self.min_value if self.min_value is not None else 0)) \
        #                   & (df['valeurfonc'] <= (self.max_value if self.max_value is not None else 10**10)) \
        #                   & (df['year'] >= (self.min_year if self.min_year is not None else 2014)) \
        #                   & (df['year'] <= (self.max_year if self.max_year is not None else 2022)) \
        #                  ]
        
        if self.dept is not None and self.insee is not None:
            if np.floor(self.insee/1000) != self.dept:
                print("The municipality you chose is not in the department you chose. Please change either the department or the municipality.")
                return None

        if self.dept is not None: 
            geo_zone = df[df['Dept'] == float(self.dept)]
        else:
            geo_zone = df

        if self.insee is not None: 
            geo_zone = geo_zone[geo_zone['l_codinsee'].apply(lambda x: str(self.insee) in re.findall(r'[0-9]+', x))]

        if self.section is not None:
            geo_zone = geo_zone[geo_zone['l_section'].apply(lambda x: str(self.section) in re.findall(r'[0-9-A-Z]+', x))]

        if self.type_of_property is not None:
            geo_zone = geo_zone[geo_zone['libtypbien'] == self.type_of_property]

        if self.max_year is not None and self.min_year is not None:
            if self.min_year <= self.max_year:
                geo_zone = geo_zone[(geo_zone['year'] >= self.min_year) & (geo_zone['year'] <= self.max_year)]
            else:
                print("The maximum value cannot be lower than the minimum value. Please change these creteria.")
                return None
        elif self.max_year is None and self.min_year is not None:
            geo_zone = geo_zone[geo_zone['year'] >= self.min_year]
        elif self.max_year is not None and self.min_year is None:
            geo_zone = geo_zone[geo_zone['year'] <= self.max_year]

        if self.max_value is not None and self.min_value is not None:
            if self.min_value <= self.max_value:
                geo_zone = geo_zone[(geo_zone['valeurfonc'] >= self.min_value) & (geo_zone['valeurfonc'] <= self.max_value)]
            else:
                print("The maximum value cannot be lower than the minimum value. Please change these creteria.")
                return None
        elif self.max_value is None and self.min_value is not None:
            geo_zone = geo_zone[geo_zone['valeurfonc'] >= self.min_value]
        elif self.max_value is not None and self.min_value is None:
            geo_zone = geo_zone[geo_zone['valeurfonc'] <= self.max_value]

        if self.min_surface is not None and self.max_surface is not None:
            if self.min_surface <= self.max_surface:
                geo_zone = geo_zone[(geo_zone['sbati'] >= self.min_surface) & (geo_zone['sbati'] <= self.max_surface)]
            else:
                print("The maximum surface cannot be lower than the minimum surface. Please change these creteria.")
                return None
        elif self.max_surface is None and self.min_surface is not None:
            geo_zone = geo_zone[geo_zone['sbati'] >= self.min_surface]
        elif self.max_surface is not None and self.min_surface is None:
            geo_zone = geo_zone[geo_zone['sbati'] <= self.max_surface]
        
        IDF_grouped = geo_zone.groupby('year')

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(x=IDF_grouped.count().index, y=round(IDF_grouped.count()['valeurfonc']*(1271568/df.shape[0])), name="Sum of the property values"),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=IDF_grouped.sum().index, y=round(IDF_grouped.sum()['valeurfonc']*(1271568/df.shape[0])), name="Number of mutations", mode="lines+markers"),
            secondary_y=True
        )

        fig.update_xaxes(title_text="Year")

        # Set y-axes titles
        fig.update_yaxes(title_text="Sum of the property values", secondary_y=True)
        fig.update_yaxes(title_text="Number of mutations", secondary_y=False)

        return fig


    
