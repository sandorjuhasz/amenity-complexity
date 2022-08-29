import pandas as pd
import numpy as np
import geopandas as gpd
import h3
import h3pandas
from shapely.geometry import Point, Polygon
import pygeos
gpd.options.use_pygeos = False

# szamlalokorzet data
szlok = gpd.read_file("data/shape_files/szlok.shp")

# szlok transformation
crs_eov = "epsg:23700"
crs_map = "epsg:4326"
szlok = szlok.set_crs(crs_eov, allow_override=True)
szlok = szlok.to_crs(crs_map)
szlok = szlok.set_geometry("geometry")

# Budapest districts
bp_shape = gpd.read_file("data/shape_files/budapest_districts.shp")
bp_shape = bp_shape.set_geometry("geometry")

# spatial join
bp_szlok = gpd.sjoin(szlok, bp_shape, "left", "within")
bp_szlok = bp_szlok.drop(["index_right"], axis=1).rename(columns={"name": "district"})
bp_szlok = bp_szlok[bp_szlok["district"].isna() == 0]

# real estate prices in Budapest blocks -- by Gergely Monus
price_df = pd.read_csv(
    "data/szlok_unitprice_multi_level_model_predictions001.csv",
    sep=";",
    decimal=",",
)
price_df = price_df.rename(columns={"mean_r_slope_pred": "pred_price"})

# real values
price_df["full_price"] = np.exp(price_df["pred_price"])

# clean up szlok ids in the price data
def create_szlok_id(row):
    szlok_id = str(int(row["SZLOKID"]))[-4:]
    return szlok_id


price_df["szlok"] = price_df.apply(create_szlok_id, axis=1)

# join bp shape and price data
bp_szlok_prices = pd.merge(
    bp_szlok,
    price_df[["TNev", "szlok", "pred_price", "full_price"]],
    on=["TNev", "szlok"],
    how="left",
).drop_duplicates()

# fill NAs with mean price
bp_szlok_prices["pred_price"] = bp_szlok_prices["pred_price"].fillna(
    np.mean(bp_szlok_prices["pred_price"])
)
bp_szlok_prices["full_price"] = bp_szlok_prices["full_price"].fillna(
    np.mean(bp_szlok_prices["full_price"])
)

# save as shp
#bp_szlok_prices.to_file("outputs/bp_szlok_pred_price.shp")


# original shape file with predicted prices
#bp_szlok = gpd.read_file("data/shape_files/bp_szlok_pred_price.shp")
#bp_szlok_prices = bp_szlok_prices.set_geometry("geometry")

# real predicted prices
bp_szlok_prices["pred_real_price"] = np.exp(bp_szlok_prices["pred_price"]).astype(int)

# create price groups based on szamlalokorzet data
bp_szlok_prices["price_group"] = pd.qcut(bp_szlok_prices["pred_real_price"], 10, labels = False)

# explode multipolygons
bp_szlok_prices = bp_szlok_prices.explode(index_parts=True)

# fill szamlalokorzet polygons with h3 hexes
h3_resolution = 10
bp_h3 = bp_szlok_prices.h3.polyfill(h3_resolution, explode=True)

# drop ca 10% of rows with NA for h3_polyfill -- ??
bp_h3 = bp_h3.dropna(subset=["h3_polyfill"])

def add_geometry(row):
  """create hex geometry"""
  points = h3.h3_to_geo_boundary(row["h3_polyfill"], True)
  return Polygon(points)

bp_h3['geometry_hex'] = bp_h3.apply(add_geometry, axis=1)
bp_h3 = bp_h3.set_geometry("geometry_hex")

# save the key columns
bp_h3_price = bp_h3[["h3_polyfill", "pred_price", "price_group", "pred_real_price"]].drop_duplicates()
bp_h3_price.to_csv("outputs/bp_hex_price.csv", index=False)