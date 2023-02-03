"""Helper functions"""

import pandas as pd
import numpy as np

import folium
import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

import os

from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib


class plotting_functions:
    
    def __init__(self, dept=None, insee=None, section=None, type_of_property=None, max_value=None, min_value=None, min_surface=None, max_surface=None, min_year=None, max_year=None):
        """Init"""
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
                        f"Surface du bien: {str(geo_zone.sbati.values[i])} m²" + "<br>" + "<br>"
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
        """
        This function plots the yearly figures of property values and number of mutations based on the given dataframe and user-defined criteria. 
        The criteria include department, municipality, section, type of property, years, value, and surface area. 
        The function groups the data by year and plots the sum of property values and the number of mutations.

        Parameters:
            df (DataFrame) : The input dataframe with property data.

        Returns:
            None
        """

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
    
    def plot_prediction_map(self, geo_df, luxury=False):

        """
        Plots a map showing the locations of the cheapest and most expansives properties based on the given parameters

        Args:
            geo_df (geopandas.geodataframe.GeoDataFrame): A dataframe containing property information.
            dept (int, optional): Department number. Defaults to None.
            insee (int, optional): Municipality code. Defaults to None.
            section (str, optional): Property section. Defaults to None.
            type_of_property (str, optional): Type of property. Defaults to None.
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

        if luxury:
            geo_zone = geo_zone[geo_zone['pred_valeurfonc'] > 0].sort_values('pred_valeurfonc')[-3:]
        else:
            geo_zone = geo_zone[geo_zone['pred_valeurfonc'] > 0].sort_values('pred_valeurfonc')[:3]



        geo_df_list = [(i[0], i[1]) for i in geo_zone[['latitude_x', 'longitude_x']].values]
        if len(geo_df_list) == 0:
            print("There are no mutation that meet the criteria you have selected. Please change these creteria.")
            return None    


        map = folium.Map(location=[geo_zone.latitude_x.median(), geo_zone.longitude_x.median()], tiles="OpenStreetMap")
        if luxury:
            for i, coordinates in enumerate(geo_df_list):
                map.add_child(
                    folium.Marker(
                        icon=folium.Icon(color="red", icon="info-sign"),
                        location=coordinates,
                        popup=
                            "Nature de la mutation: " + str(geo_zone.libnatmut.values[i]) + "<br>" + "<br>"
                            f"Valeur foncière estimée: {str(round(geo_zone.pred_valeurfonc.values[i], 2))}€" + "<br>" + "<br>"
                            "Type de bien: " + str(geo_zone.libtypbien.values[i]) + "<br>" + "<br>"
                            f"Surface du bien: {str(geo_zone.sbati.values[i])} m^2" + "<br>" + "<br>"
                            "CODE INSEE: " + ', '.join(re.findall(r'[0-9]+', geo_zone.l_codinsee.values[i])) + "<br>" + "<br>"
                            "Section: " + ', '.join(re.findall(r'[0-9A-Z]+', geo_zone.l_section.values[i])) + "<br>" # + "<br>"
                            #+ "Coordinates: " + str(geo_df_list[i]),
                    )
                )
        else:
            for i, coordinates in enumerate(geo_df_list):
                map.add_child(
                    folium.Marker(
                        icon=folium.Icon(color="blue", icon="info-sign"),
                        location=coordinates,
                        popup=
                            "Nature de la mutation: " + str(geo_zone.libnatmut.values[i]) + "<br>" + "<br>"
                            f"Valeur foncière estimée: {str(round(geo_zone.pred_valeurfonc.values[i], 2))}€" + "<br>" + "<br>"
                            "Type de bien: " + str(geo_zone.libtypbien.values[i]) + "<br>" + "<br>"
                            f"Surface du bien: {str(geo_zone.sbati.values[i])} m^2" + "<br>" + "<br>"
                            "CODE INSEE: " + ', '.join(re.findall(r'[0-9]+', geo_zone.l_codinsee.values[i])) + "<br>" + "<br>"
                            "Section: " + ', '.join(re.findall(r'[0-9A-Z]+', geo_zone.l_section.values[i])) + "<br>" # + "<br>"
                            #+ "Coordinates: " + str(geo_df_list[i]),
                    )
                )

        sw = geo_zone[['latitude_x', 'longitude_x']].min().values.tolist()
        ne = geo_zone[['latitude_x', 'longitude_x']].max().values.tolist()

        map.fit_bounds([sw, ne])

        return map


class modelling:
    """Modelling"""
    def __init__(self, folder_path="data_localisee/"):
        """Init"""
        self.folder_path = folder_path
        
    def sort_csv_files(self):
        """
        Sort all the .csv files in the folder path specified in the class.

        Returns:
            A single pandas DataFrame with data from all the .csv files in the folder path.
        """
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        dfs = []
        for file in files:
            df = pd.read_csv(os.path.join(self.folder_path, file), low_memory=False)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def exclude_lines(df, max_fonc=1500000):
        """
        Filter dataframe `df` based on certain conditions.

        This function filters the input dataframe `df` based on the following conditions:
        - `valeurfonc` column should be less than or equal to `max_fonc`
        - `valeurfonc` column should be greater than or equal to 100
        - `libtypbien` column should contain specific values as defined in the function

        Args:
            df (pandas.DataFrame): The dataframe to filter.
            max_fonc (int): The maximum value for the `valeurfonc` column. The default is 1500000.

        Returns:
            pandas.DataFrame: The filtered dataframe that meets the conditions defined in the function.
        """
        
        res = df[df["valeurfonc"] <= max_fonc]
        res = res[res["valeurfonc"] >= 100]
        res = res[((res["libtypbien"] == 'UN APPARTEMENT')
                   | (res["libtypbien"] == 'PPARTEMENT INDETERMINE')
                   | (res["libtypbien"] == 'UNE MAISON')
                   | (res["libtypbien"] == 'MAISON - INDETERMINEE')
                   | (res["libtypbien"] == 'DES MAISONS')
                   | (res["libtypbien"] == 'DEUX APPARTEMENTS')
                  )]
        return res

    @staticmethod
    def add_arrondissement(df):
        """
        Add arrondissement column to the dataframe

        This function takes a dataframe as input and adds a new column "arro" to it, which represents the arrondissement of a property. 
        The arrondissement is derived from the "l_codinsee" column of the dataframe. 
        If the first two digits of the "l_codinsee" value is equal to "75", the last two digits are considered as the arrondissement. 
        Else, the arrondissement value is set to 0. The resulting "arro" column is of integer type.

        Args:
            df (pandas.DataFrame): The input dataframe

        Returns:
            None
        """
        n_col = df["l_codinsee"].apply(lambda s: s[-4:-2] if s[2:4]=="75" else 0)
        df["arro"] = n_col.astype("int")

    @staticmethod
    def feature_selection(df):
        """
        This function selects the relevant features from the input dataframe and returns a new dataframe
        with the selected features.

        Parameters:
            df (pandas.DataFrame): The input dataframe to perform feature selection on

        Returns:
            pandas.DataFrame: The new dataframe with the selected features

        """
        base_columns = ["anneemut", "moismut", "coddep", "vefa", "valeurfonc", "sterr",
                        "nblocdep", "latitude", "longitude", "nb_accomodations", "surf",
                        "arro", "nb_rooms"
                       ]
        df['vefa'] = df['vefa'].apply(lambda x: 1 if x == True else (0 if x==False else np.nan))
        df["nb_accomodations"] = df["nblocmai"] + df["nblocapt"]
        df["surf"] = df["sbatmai"] + df["sbatapt"] #df["sbati"] + df["sbatact"]
        df["arro"] = df["l_codinsee"].apply(lambda s: s[-4:-2] if s[2:4]=="75" else 0).astype("int")
        df["nb_rooms"] = (df["nbapt1pp"] 
                           + 2 * df["nbapt2pp"]
                           + 3 * df["nbapt3pp"]
                           + 4 * df["nbapt4pp"]
                           + 5 * df["nbapt5pp"]
                           + df["nbmai1pp"] 
                           + 2 * df["nbmai2pp"]
                           + 3 * df["nbmai3pp"]
                           + 4 * df["nbmai4pp"]
                           + 5 * df["nbmai5pp"]
                          ) / df["nb_accomodations"]
        return df[base_columns]
    
    @staticmethod
    def create_min_max_scaler():
        """
        Create Min Max Scaler

        Returns:
            StandardScaler: instance of StandardScaler from scikit-learn library.
        """
        scaler = StandardScaler()
        return scaler

    @staticmethod
    def min_max_scale(df, scaler):
        """
        Transform the values in a single column of a pandas DataFrame using a StandardScaler object.

        Parameters:
            df (pandas DataFrame): The DataFrame to be transformed.
            scaler (StandardScaler): The StandardScaler object to use for the transformation.

        Returns:
            None
        """        

        df['valeurfonc'] = scaler.fit_transform(df['valeurfonc'].values.reshape(-1, 1))

    @staticmethod
    def min_max_rescale(list_, scaler):
        """
        Rescale a list using the specified MinMaxScaler

        Parameters:
            list_ (list): List to be rescaled
            scaler (StandardScaler): Scaler used for the rescale

        Returns
            rescaled_list (list):Rescaled list
        """     
        list_ = scaler.inverse_transform(list_.reshape(-1, 1))
        return list_
    
    @staticmethod
    def X_y_split(df):
        """
        Splits the data into feature and target variables.

        Inputs:
            df : pandas dataframe
            A dataframe with the data to be split into features and target variables.

        Outputs:
            X (pandas dataframe): A dataframe with feature variables.
            y (pandas series): A series with target variable.
    
        """
        features = ["anneemut", "moismut", "coddep", "vefa", "sterr",
                        "nblocdep", "latitude", "longitude", "nb_accomodations", "surf",
                        "arro", "nb_rooms"
                       ]
        X = df[features]
        y = df["valeurfonc"]
        return X, y
    
    def preprocessing_and_splitting(self):
        """
        Preprocesses the mutation data and splits it into train and test sets

        Returns:
            X_train (pandas.DataFrame): Training features data.
            X_test (pandas.DataFrame): Testing features data.
            Y_train (pandas.Series): Training target data.
            Y_test (pandas.Series): Testing target data.
            scaler (StandardScaler): Scaler instance used to scale the "valeurfonc" feature.
        """        
        df_1 = self.sort_csv_files()
        mutation = self.exclude_lines(df_1)
        self.add_arrondissement(mutation)
        # self.sum_surface(mutation)
        mutation = self.feature_selection(mutation)
        scaler = self.create_min_max_scaler()
        self.min_max_scale(mutation, scaler)
        X, y = self.X_y_split(mutation)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)
        return X_train, X_test, Y_train, Y_test, scaler

    @staticmethod
    def training(X_train, Y_train, save=False, pkl_file_path=None): 
        """
        Train an XGBoost regressor model using the provided training data.

        Parameters:
            X_train (pandas.DataFrame): The training data for the model's independent variables.
            Y_train (pandas.Series): The training data for the model's dependent variable.
            save (bool, optional): Whether to save the trained model to a file. Defaults to False.
            pkl_file_path (str, optional): The file path to save the trained model to, if save is True.
                Defaults to None.

        Returns:
            xgboost (xgboost.XGBRegressor): The trained XGBoost model.
        """
        from xgboost import XGBRegressor
        xgboost = XGBRegressor()
        xgboost.fit(X_train, Y_train)
        if save:
            joblib.dump(xgboost, pkl_file_path)
        return xgboost

    @staticmethod
    def predict(xgboost, X_test):
        """
        make predictions using an XGBoost model

        Parameters:
            xgboost (XGBRegressor): Trained XGBoost model
            X_test (DataFrame): Test data features

        Returns:
            y_pred (ndarray): Predictions
        """
        y_pred = xgboost.predict(X_test)
        return y_pred

    @staticmethod
    def load_model(pkl_file_path):
        """
        Load an XGBoost model saved as a pickle file.

        Parameters:
            pkl_file_path (str): The file path of the pickled XGBoost model.

        Returns:
            xgboost (XGBRegressor): The loaded XGBoost model.
        """
        xgboost = joblib.load(pkl_file_path)
        return xgboost
    
    
    # R²
    @staticmethod
    def rsqr_score(test, pred):
        """Calculate Root Mean Square Error score 

        Args:
            test -- test data
            pred -- predicted data

        Returns:
            Root Mean Square Error score
        """
        rsqr_ = r2_score(test, pred)
        return rsqr_
    
    @staticmethod
    def rmse_score(test, pred):
        """Calculate Root Mean Square Error score 

        Args:
            test -- test data
            pred -- predicted data

        Returns:
            Root Mean Square Error score
        """
        rmse_ = np.sqrt(mean_squared_error(test, pred))
        return rmse_

    @staticmethod
    def mse_score(test, pred):
        """Calculate Root Mean Square Error score 

        Args:
            test -- test data
            pred -- predicted data

        Returns:
            Root Mean Square Error score
        """
        mse_ = mean_squared_error(test, pred)
        return mse_
    
    @staticmethod
    def mae_score(test, pred):
        """Calculate Mean Absolute Error score 

        Args:
            test -- test data
            pred -- predicted data

        Returns:
            Mean Absolute Error score
        """
        mae_ = mean_absolute_error(test, pred)
        return mae_

    # Print the scores
    def print_score(self, test, pred):
        """Print calculated score 

        Args:
            test -- test data
            pred -- predicted data

        Returns:
            print the R squared score
            print Root Mean Square Error score
            print Mean Square Error score
            print Mean Absolute Error score
        """

        print(f"R²: {self.rsqr_score(test, pred)}")
        print(f"RMSE: {self.rmse_score(test, pred)}")
        print(f"MSE: {self.mse_score(test, pred)}")
        print(f"MAE: {self.mae_score(test, pred)}")
