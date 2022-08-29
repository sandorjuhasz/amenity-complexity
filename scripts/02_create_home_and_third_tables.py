import pandas as pd
import numpy as np
import geopandas as gpd
import h3
import h3pandas
from geopy import distance
from haversine import haversine_vector, Unit
import time
import sys

sys.path.insert(0, '../scripts/')
from ub_explorer import ub_explorer

"""
create tables for home / work / third places based on stop detection results
"""


periods = ["2019-05",
           "2019-06",
           "2019-07",
           "2019-08",
           "2019-09",
           "2019-10",
           "2019-11",
           "2019-12",
           "2020-01",
           "2020-02",
           "2020-03",
           "2020-04",
           "2020-05",
           "2020-06",
           "2020-07",
           "2020-08",
           "2020-09",
           "2020-10",
           "2020-11",
           "2020-12",
           "2021-01",
           "2021-02",
           "2021-03",
           "2021-04",
           "2021-05"]

h3_resolution = 10
bp_h3 = pd.read_csv("outputs/bp_hex_prices.csv")
home_output = pd.DataFrame()
third_output = pd.DataFrame()

start_time = time.time()
for p in periods:
    ub = ub_explorer(period=p)
    
    # create home / work tables
    home_table = ub.place_of_the_month(ub.data, "home", h3_resolution)
    
    # add price group info to HOME
    home_table = pd.merge(
        home_table,
        bp_h3,
        left_on="h3",
        right_on="h3_polyfill",
        how="left"
    ).dropna(subset="price_group")
    
    # append -- home
    home_output = pd.concat([home_output, home_table], ignore_index=True)
    
    # create third places table
    third_df = ub.third_places_table(ub.data, h3_resolution)

    # add price group to third places
    third_df = pd.merge(
        third_df,
        bp_h3,
        left_on="h3",
        right_on="h3_polyfill",
        how="left"
    ).dropna(subset="price_group")

    # add home price group to third places table
    third_df = pd.merge(
        third_df,
        home_table,
        on = ["device_id", "year_month"],
        how = "left",
        suffixes=["_third", "_home"]
    )

    # HUGE drop -- remove stops of users with NO identified home in year_month
    third_df = third_df.dropna(subset=["h3_polyfill_home"])
    
    # distance of third places
    third_df["home_coords"] = list(zip(third_df["mean_lat_home"], third_df["mean_lon_home"]))
    third_df["third_coords"] = list(zip(third_df["mean_lat_third"], third_df["mean_lon_third"]))
    third_df["distance"] = haversine_vector(third_df["home_coords"].tolist(), third_df["third_coords"].tolist())

    # append -- third
    third_output = pd.concat([third_output, third_df], ignore_index=True)

# export
home_output.to_csv("outputs/home_table_full.csv", index=False)
third_output.to_csv("outputs/third_table_full.csv", index=False)

print("--- %s seconds ---" % round((time.time() - start_time), 3))