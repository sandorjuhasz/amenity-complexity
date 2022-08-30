import numpy as np
import pandas as pd
import geopandas as gpd
import h3
import h3pandas



def add_poi_complexity_to_full_poi_data(raw_poi_data, poi_complexity_data, h3_resolution):
    """add POI complexity to all POIs from the raw poi data"""

    poi_hex_c = pd.merge(
        raw_poi_data[["place_id", "category_78", "geometry"]],
        poi_complexity_data,
        on="category_78",
        how="left"
    )

    # add h3 hex based on h3_resolution
    poi_hex_c["h3"] = poi_hex_c.apply(lambda r: h3.geo_to_h3(r["geometry"].y, r["geometry"].x, h3_resolution), axis=1)

    return poi_hex_c


def hex_to_location_complexity(city_parts_hex, location_complexity):
    """add hexes to locations with complexity measures"""
    
    location_hex_c = gpd.GeoDataFrame(pd.merge(
        city_parts_hex[["NAME", "h3_polyfill", "geometry_hex"]],
        location_complexity.drop(columns=["NAME"]),
        left_on=["NAME"],
        right_on=["location_name"],
        how="left"
    )
    )
    location_hex_c = location_hex_c.rename(columns={"h3_polyfill":"h3_part"})

    return location_hex_c


def add_location_info_to_home_and_third_places(third_df, location_hex_c, complexity_df):
    """add city part level location info to homes and third places"""
    
    # join city part name and complexity to third places
    third_c = pd.merge(
        third_df,
        location_hex_c,
        left_on="h3_third",
        right_on="h3_part",
        how="left"
    )

    # merge city_part table to home
    third_c = pd.merge(
        third_c,
        location_hex_c,
        left_on="h3_home",
        right_on="h3_part",
        how="left",
        suffixes=["_third", "_home"]
    )

    # add avg ubiquity to third places
    avg_ubiquity = complexity_df[complexity_df["rca"] >= 1]\
        .groupby(["location_name"])\
        .agg(avg_ubiquity = pd.NamedAgg("ubiquity", "mean"))\
        .reset_index()

    third_c = pd.merge(
        third_c,
        avg_ubiquity,
        left_on="location_name_third",
        right_on="location_name",
        how="left"
    )
    
    return third_c


def add_szlok_level_prices_to_home_locations(bp_szlok, third_df, h3_resolution):
    """add szlok ids and predicted prices to home locations in home-third places table"""
    
    # geometry setting
    bp_szlok = bp_szlok.set_geometry("geometry")
    
    # real predicted prices
    bp_szlok["pred_real_price"] = np.exp(bp_szlok["pred_price"]).astype(int)

    # create price groups based on szamlalokorzet data
    bp_szlok["price_group"] = pd.qcut(bp_szlok["pred_real_price"], 10, labels = False)

    # explode multipolygons
    bp_szlok = bp_szlok.explode(index_parts=True)

    # fill szamlalokorzet polygons with h3 hexes
    h3_resolution = 10
    bp_h3 = bp_szlok.h3.polyfill(h3_resolution, explode=True)

    # drop ca 10% of rows with NA for h3_polyfill -- ??
    bp_h3 = bp_h3.dropna(subset=["h3_polyfill"])

    # add szlok info to home locations in bp_h3
    third_h = pd.merge(
        third_df,
        bp_h3,
        left_on="h3_home",
        right_on="h3_polyfill",
        how="left"
    )

    return third_h


# city part -- population
def population_to_locations(bp_szlok, pop_data, location_complexity):
    bp_szlok = bp_szlok.set_geometry("geometry")
    pop_szlok = pd.merge(
        bp_szlok,
        pop_data,
        left_on="TNev",
        right_on="szlok_id",
        how="left"
        )

    # join szloks to city parts
    part_pop = gpd.sjoin(location_complexity, pop_szlok, how="left")
    part_pop = part_pop.groupby(["location_name"])["laknep"].agg("sum").reset_index()

    return part_pop
