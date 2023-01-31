import pandas as pd
import numpy as np

import folium
import re


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