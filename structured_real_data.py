import os
import polars as pl
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

mypath="Carteras"

Carteras=os.listdir(mypath)
dirname = os.path.dirname(__file__)

dfs = []
for cartera in Carteras:
    try:
        print(cartera)
        df = pl.read_csv(dirname+"\\Carteras\\"+cartera, separator=";",infer_schema_length=1000000,encoding="utf8-lossy")
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {cartera}: {e}")

if dfs:
    combined_df = pl.concat(dfs,how='diagonal_relaxed')
else:
    combined_df = pl.DataFrame() # Or handle the case where no files were read
path= "Carteras.parquet"
combined_df.write_parquet(path)