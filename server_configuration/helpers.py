import pandas as pd
import numpy as np

import folium
import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class plotting_functions():
    def __init__(self, dept=None, insee=None, section=None, type_of_property=None, max_value=None, min_value=None, min_surface=None, max_surface=None):
        self.dept = dept
        self.insee = insee
        self.section = section
        self.type_of_property = type_of_property
        self.max_value = max_value
        self.min_value = min_value
        self.max_surface = max_surface
        self.min_surface = min_surface
        
    def plot_map(self, geo_df, year_of_mutation=None):

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

        if year_of_mutation is not None:
            geo_zone = geo_zone[geo_zone['year'] == year_of_mutation]

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



    def plot_yearly_figures(self, IDF, year_min=None, year_max=None):

        IDF_grouped = IDF[ \
                          (IDF['valeurfonc'] >= (self.min_value if self.min_value is not None else 0)) \
                          & (IDF['valeurfonc'] <= (self.max_value if self.max_value is not None else 10**10)) \
                          & (IDF['year'] >= (year_min if year_min is not None else 2014)) \
                          & (IDF['year'] <= (year_max if year_max is not None else 2022)) \
                         ].groupby('year')

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(x=IDF_grouped.count().index, y=IDF_grouped.count()['valeurfonc'], name="Sum of the property values"),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=IDF_grouped.mean().index, y=IDF_grouped.mean()['valeurfonc'], name="Number of mutations", mode="lines+markers"),
            secondary_y=True
        )

        fig.update_xaxes(title_text="Year")

        # Set y-axes titles
        fig.update_yaxes(title_text="Sum of the property values", secondary_y=True)
        fig.update_yaxes(title_text="Number of mutations", secondary_y=False)

        return fig


    
def plot_map(geo_df, dept=None, insee=None, section=None, type_of_property=None, year_of_mutation=None, max_value=None, min_value=None, min_surface=None, max_surface=None):
    
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

    
    if dept is not None and insee is not None:
        if np.floor(insee/1000) != dept:
            print("The municipality you chose is not in the department you chose. Please change either the department or the municipality.")
            return None
                
    if dept is not None: 
        geo_zone = geo_df[geo_df['Dept'] == float(dept)]
    else:
        geo_zone = geo_df

    if insee is not None: 
        geo_zone = geo_zone[geo_zone['l_codinsee'].apply(lambda x: str(insee) in re.findall(r'[0-9]+', x))]

    if section is not None:
        geo_zone = geo_zone[geo_zone['l_section'].apply(lambda x: str(section) in re.findall(r'[0-9-A-Z]+', x))]
    
    if type_of_property is not None:
        geo_zone = geo_zone[geo_zone['libtypbien'] == type_of_property]
    
    if year_of_mutation is not None:
        geo_zone = geo_zone[geo_zone['year'] == year_of_mutation]
        
    if max_value is not None and min_value is not None:
        if min_value <= max_value:
            geo_zone = geo_zone[(geo_zone['valeurfonc'] >= min_value) & (geo_zone['valeurfonc'] <= max_value)]
        else:
            print("The maximum value cannot be lower than the minimum value. Please change these creteria.")
            return None
    elif max_value is None and min_value is not None:
        geo_zone = geo_zone[geo_zone['valeurfonc'] >= min_value]
    elif max_value is not None and min_value is None:
        geo_zone = geo_zone[geo_zone['valeurfonc'] <= max_value]
        
    if min_surface is not None and max_surface is not None:
        if min_surface <= max_surface:
            geo_zone = geo_zone[(geo_zone['sbati'] >= min_surface) & (geo_zone['sbati'] <= max_surface)]
        else:
            print("The maximum surface cannot be lower than the minimum surface. Please change these creteria.")
            return None
    elif max_surface is None and min_surface is not None:
        geo_zone = geo_zone[geo_zone['sbati'] >= min_surface]
    elif max_surface is not None and min_surface is None:
        geo_zone = geo_zone[geo_zone['sbati'] <= max_surface]

        
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



def plot_yearly_figures(min_value=None, max_value=None, year_min=None, year_max=None):
    
    IDF_grouped = truc = IDF[ \
                             (IDF['valeurfonc'] >= (min_value if min_value is not None else 0)) \
                             & (IDF['valeurfonc'] <= (max_value if max_value is not None else 10**10)) \
                             & (IDF['year'] >= (year_min if year_min is not None else 2014)) \
                             & (IDF['year'] <= (year_max if year_max is not None else 2022)) \
                            ].groupby('year')

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=IDF_grouped.count().index, y=IDF_grouped.count()['valeurfonc'], name="Sum of the property values"),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=IDF_grouped.mean().index, y=IDF_grouped.mean()['valeurfonc'], name="Number of mutations", mode="lines+markers"),
        secondary_y=True
    )

    fig.update_xaxes(title_text="Year")

    # Set y-axes titles
    fig.update_yaxes(title_text="Sum of the property values", secondary_y=True)
    fig.update_yaxes(title_text="Number of mutations", secondary_y=False)

    fig.show()