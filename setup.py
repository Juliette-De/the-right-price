"""Setup file"""

from distutils.core import setup

setup(
    name="TheRightPrice",
    version="1.0",
    description="Optimize the whole value chain of a player in real estate industry, and in particular the estimation of the purchase/sale price",
    author="Team 3",
    author_email="juliette.demoly@hec.edu",
    packages=["TheRightPrice"],
    install_requires=[
        "numpy",
        "pandas",
        "streamlit",
        "streamlit_folium",
        "pandas", 
        "numpy", 
        "pillow",
        "geopandas",
        "folium",
        "plotly",
        "sklearn",
        "xgboost",
        "joblib",
    ],
)
