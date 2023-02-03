# Data commercial proposal with Eleven - The right price

This application was developed for a player in the real estate industry who wants to optimize its whole value chain, and in particular the estimation of the purchase/sale price.


## Quick start

Clone this github repository or upload all of its files to the folder where you want to place this project.
You can install the necessary packages from the requirements.txt file provided with this repository. In the terminal, replacing path with the path of your dedicated folder:
```
pip install -r path/requirements.txt
```

Then, to launch the application:
```
streamlit run path/server_configuration/Home.py
```


## Features

This application offers the following three features, each on one page:
- explore the state of the real estate market in Ile-de-France and analyze the different temporal or geographical trends;
- predict prices using a machine learning model;
- predict the best opportunities to save time and money.


## Organization of the repository

The server_configuration/home.py file and the server_configuration/pages folder contain the application structure.

The notebooks folder contains the various notebooks used for data collection, data augmentation, data exploration, and the model.


## Background

This project was created for a week-long challenge run by [eleven](https://eleven-strategy.com). It aimed to offer a solution to a client wanting to correctly estimate the price of land in order to identify the best opportunities.
