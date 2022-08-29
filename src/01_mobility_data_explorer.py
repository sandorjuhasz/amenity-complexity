import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import h3
from shapely.geometry import Point, Polygon, LineString
from geopy import distance
import time
import math
from parquetranger import TableRepo


"""
create tables for home / work / third places based on stop detection results
"""


class ub_explorer:
    """
    class to create tables for key locations -- supporting explorative work
    """

    def __init__(self, period="2020-01"):
        self.period = period
        
        print("Loading data..", end="")
        self.data = self.load_data()
        print(" DONE")

    def load_data(self):
        df = pd.read_parquet(
            "/mnt/common-ssd/anet-shares/mobility-data/repartitioned-semantic-stops/"+
            self.period+
            ".parquet",
            engine="pyarrow"
        )
        
        return df

    def place_of_the_month(self, stop_detection_data, location_type, h3_resolution):
        """home / work location of the month based on coord means"""

        if location_type == "home":
            df = stop_detection_data[stop_detection_data["home__identified"] == 1]
        elif location_type == "work":
            df = stop_detection_data[stop_detection_data["work__identified"] == 1]
        else:
            print("INVALID location type")

        # filter places with -1 label (movement)
        df = df[df["place_label"] > 0]

        df = (
            df.groupby(["device_id", "year_month"])
            .agg(
                mean_lon=pd.NamedAgg("center__lon", "mean"),
                mean_lat=pd.NamedAgg("center__lat", "mean"),
                std_lon=pd.NamedAgg("center__lon", "std"),
                std_lat=pd.NamedAgg("center__lat", "std"),
                nr_stops=pd.NamedAgg("stop_number", "count"),
            )
            .reset_index()
        )

        # mean is based on 10 stops at least
        df = df[df["nr_stops"] >= 10]

        # drop cases with above threshold std
        df = df[(df["std_lon"] <= 0.001) & (df["std_lat"] <= 0.001)]

        # add h3 hex based on h3_resolution
        df["h3"] = df.apply(lambda r: h3.geo_to_h3(r["mean_lat"], r["mean_lon"], h3_resolution), axis=1)

        return df

    def place_to_szlok(self, userdf, coord_col_names, mapdf):
        """spatial join places to map"""
        coordf = userdf[[c for c in userdf.columns if c in coord_col_names]]
        userdf["geometry"] = coordf.apply(lambda r: Point(r.iloc[0], r.iloc[1]), axis=1)
        userdf = userdf.set_geometry("geometry")
        userdf = userdf.set_crs("epsg:4326")

        # spatial join
        place_szlok = gpd.sjoin(userdf, mapdf, "left", "within")
        place_szlok = place_szlok.dropna(subset=["district"])

        return place_szlok

    def third_places_table(self, data, h3_resolution):
        # manipulation
        data["dayofmonth"] = data["interval__start"].dt.day

        # third place filter
        third_df = data[(data["home__identified"]==0) &
                        (data["work__identified"]==0) &
                        (data["place_label"] > 0)]
        
        third_df = (
            third_df.groupby(["device_id", "year_month", "place_label"]))\
            .agg(
                nr_visits = pd.NamedAgg("stop_number", "count"),
                nr_days = pd.NamedAgg("dayofmonth", "nunique"),
                mean_lon = pd.NamedAgg("center__lon", "mean"),
                mean_lat = pd.NamedAgg("center__lat", "mean"),
                std_lon = pd.NamedAgg("center__lon", "std"),
                std_lat = pd.NamedAgg("center__lat", "std"),
                mean_duration = pd.NamedAgg("duration", "mean")
            ).reset_index()
        
        # drop unstable third places -- would be a HUGE drop
        # third_df = third_df[(third_df["std_lon"] <= 0.001) & (third_df["std_lat"] <= 0.001)]
        
        # add price group to third place
        third_df["h3"] = third_df.apply(lambda r: h3.geo_to_h3(r["mean_lat"], r["mean_lon"], h3_resolution), axis=1)
        
        return third_df

    