import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from ecomplexity import ecomplexity


"""
complexity of locations and pois in Budapest
"""


# parameters
path_location_file = "../data/shape_files/neighborhoods_admin10.shp"
path_poi_file = "../data/google_pois_2021_enriched.geojson"
selected_cat = "category_78"
filter_out = ["ATM", "Parking"]
min_categories_per_location = 2
min_pois_per_category = 2


class poi_complexity:
    """
    class to create location / poi category table with complexity related variables
    """

    def __init__(self):
        print("Preparing location data.. ", end="")
        self.location_data = self.prep_location_data()
        print(" location_data READY")

        print("Preparing POI data.. ", end="")
        self.raw_poi_data = self.prep_poi_data()
        print(" raw_poi_data READY")

        print("Cleaning POI data.. ", end="")
        self.poi_data = self.filter_poi_data(self.raw_poi_data, filter_out)
        print(" FILTERED for", filter_out)

        print("ADD location to POIs.. ", end="")
        self.poi_data = self.pois_locations(self.poi_data, self.location_data)
        print(" poi_data READY")

        print("Preparing aggregated location category table..", end="")
        self.category_location_table = self.filter_category_location_table(
            self.create_category_location_table(self.poi_data, selected_cat),
            min_categories_per_location,
            min_pois_per_category
            )
        print(" category_location_table READY")

        print("Measuring complexity.. ", end="")
        self.complexity_df = self.create_complexity_df(self.category_location_table)
        print(" complexity_df READY")

        print("Creating location complexity table.. ", end="")
        self.location_complexity = self.create_location_complexity_table(self.location_data, self.complexity_df)
        print(" location_complexity READY")

        print("Creating POI complexity table.. ", end="")
        self.poi_complexity = self.create_poi_complexity_table(self.complexity_df)
        print(" poi_complexity READY")


    # data prep functions
    def prep_location_data(self):
        locations = gpd.read_file(path_location_file)
        locations = locations.set_geometry("geometry")
        locations = locations.to_crs("epsg:4326")
        return locations

    def prep_poi_data(self):
        raw_poi_data = gpd.read_file(path_poi_file)
        return raw_poi_data


    # poi filters
    def remove_atm(self, df):
        filtered = df[(df["amenity_category"] != "ATM") & ~(df["name"]).str.contains(" ATM ", case=False)]
        return filtered

    def remove_parking(self, df):
        filtered = df[df["amenity_category"] != "Parking"]
        return filtered

    def remove_malls(self, df):
        filtered = df[df["mall"].isna()]
        return filtered

    def mask_inside_malls_pois(self, df):
        df.loc[df["mall"].isna()==False, selected_cat] = "shopping mall"
        filtered = df
        return filtered

    def filter_poi_data(self, df, filter_list):
        """
        combine previously defined filters
        """
        filtered = df
        
        if any("ATM" in p for p in filter_list):
            filtered = self.remove_atm(filtered)
        else:
            filtered = filtered

        if any("Parking" in p for p in filter_list):
            filtered = self.remove_parking(filtered)
        else:
            filtered

        if any("mall" in p for p in filter_list):
            filtered = self.remove_malls(filtered)
        else:
            filtered

        if any("mask" in m for m in filter_list):
            filtered = self.mask_inside_malls_pois(filtered)
        else:
            filtered

        return filtered


    # spatial join POIs and locations
    def pois_locations(self, poi_gdf, location_gdf):
        poi_with_locations = gpd.sjoin(
            poi_gdf,
            location_gdf.rename({"NAME":"location_name"}, axis=1),
            "left",
            "within"
        )
        return poi_with_locations


    # create aggregate category location table
    def create_category_location_table(self, df, category_col):
        # create vars for future filter
        df["nr_categories"] = df.groupby(["location_name"])[category_col].transform("nunique")

        # POI category - area - count table
        cat_location_table = df.groupby(["location_name", category_col, "nr_categories"]).agg(
            poi_count = pd.NamedAgg("place_id", "count")
        ).reset_index()
        return cat_location_table

    def filter_category_location_table(self, df, min_categories, min_pois):
        df = df[df["nr_categories"] >= min_categories]
        df = df[df["poi_count"] >= min_pois]
        return df
    

    # measure complexity -- thanks Growth Lab
    def create_complexity_df(self, df):
        df["year"] = 2020
        
        key_cols = {
            "time" : "year",
            "loc" : "location_name",
            "prod" : selected_cat,
            "val" : "poi_count"
        }

        complexity_df = ecomplexity(df, key_cols)
        return complexity_df

    def create_location_complexity_table(self, location_data, complexity_data):
        location_complexity = pd.merge(
            location_data,
            complexity_data[["location_name", "eci", "diversity"]].drop_duplicates(),
            left_on="NAME",
            right_on="location_name",
            how="left"
            )

        ubi_temp = complexity_data[complexity_data["rca"] >= 1]\
            .groupby(["location_name"])\
            .agg(avg_ubiquity = pd.NamedAgg("ubiquity", "mean"))\
            .reset_index()

        location_complexity = pd.merge(
            location_complexity,
            ubi_temp,
            on="location_name",
            how="left"
            )
        
        # normalization
        location_complexity["eci_norm"] = (location_complexity["eci"] - location_complexity["eci"].min()) / (location_complexity["eci"].max() - location_complexity["eci"].min())
        location_complexity["div_norm"] = (location_complexity["diversity"] - location_complexity["diversity"].min()) / (location_complexity["diversity"].max() - location_complexity["diversity"].min())
        return location_complexity

    def create_poi_complexity_table(self, complexity_data):
        poi_complexity = complexity_data[[selected_cat, "pci", "ubiquity"]].drop_duplicates()

        # normalization
        poi_complexity["ubi_norm"] = (poi_complexity["ubiquity"] - poi_complexity["ubiquity"].min()) / (poi_complexity["ubiquity"].max() - poi_complexity["ubiquity"].min())        
        return poi_complexity