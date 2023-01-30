import pandas as pd
import numpy as np

import folium
import re


def plot_map(geo_df, dept=None, insee=None, section=None, type_of_property=None, year_of_mutation=None):
    
    """
    Plots a map showing the locations of properties based on the given parameters

    Args:
    geo_df (geopandas.geodataframe.GeoDataFrame): A dataframe containing property information.
    dept (int, optional): Department number. Defaults to None.
    insee (int, optional): Municipality code. Defaults to None.
    section (str, optional): Property section. Defaults to None.
    type_of_property (str, optional): Type of property. Defaults to None.
    year_of_mutation (int, optional): Year of mutation. Defaults to None.

    Returns:
    folium.Map: A map object showing the locations of the properties. Returns None if the criteria selected result in no data.
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
                    f"Date de la mutation: {geo_zone.day.values[i]}/{geo_zone.month.values[i]}/{geo_zone.year.values[i]}" + "<br>" + "<br>"
                    "Type de bien: " + str(geo_zone.libtypbien.values[i]) + "<br>" + "<br>"
                    "CODE INSEE: " + ', '.join(re.findall(r'[0-9]+', geo_zone.l_codinsee.values[i])) + "<br>" + "<br>"
                    "Section: " + ', '.join(re.findall(r'[0-9A-Z]+', geo_zone.l_section.values[i])) + "<br>" # + "<br>"
                    #+ "Coordinates: " + str(geo_df_list[i]),
            )
        )

    sw = geo_zone[['latitude', 'longitude']].min().values.tolist()
    ne = geo_zone[['latitude', 'longitude']].max().values.tolist()

    map.fit_bounds([sw, ne])
    
    return map