"""
Title: Loader

Author: Masha Khanina
Reviewer: Nir Landerer

Description: Module for loading MovieLens files into storage, and
parsing them to database.
"""
import os
import shutil
import zipfile

import duckdb
import requests
import pandas as pd


GENRES = [
    "unknown", "action", "adventure", "animation", "children",
    "comedy", "crime", "documentary", "drama",
    "fantasy", "film_noir", "horror", "musical",
    "mystery", "romance", "sci_fi",
    "thriller", "war", "western",
]


TABLES_SCHEMAS = {
    "u.user": {
        "table_name": "users",
        "columns": ["user_id", "age", "gender", "occupation", "zip_code"],
    },

    "u.item": {
        "table_name": "movies",
        "columns": [
            "movie_id", "title", "release_date", "vid_release", 
            "IMDB_URL", 
        ] + GENRES,
    },

    "u.data": {
        "table_name": "ratings",
        "columns": ["user_id", "movie_id", "rating", "timestamp"]
    }
}


def load_raw_files(dest_dir: str, url: str = ""):
    """Load MovieLens files with users, movies, and ratings."""
    os.makedirs(dest_dir, exist_ok=True)
    temp_zip_dest = "/tmp/ml100k.zip"
    if not url:
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    
    with requests.get(url, stream=True) as response:
        with open(temp_zip_dest, mode="wb") as zip_file:
            for chunk in response.iter_content(8192):
                zip_file.write(chunk)
    with zipfile.ZipFile(temp_zip_dest, mode="r") as zip_file:
        for file in ["u.user", "u.data", "u.item"]:
            file_path = os.path.join("ml-100k", file)
            with zip_file.open(file_path) as src:
                with open(os.path.join(dest_dir, file), "wb") as dest:
                    shutil.copyfileobj(src, dest)


def load_table(file_path: str, sep: str = "|"):
    """
        Parse csv file to a database. 
        Fits for large files (1_000_000_000 rows).
    """
    db_path = os.environ["DB_PATH"]
    chunk_size = int(os.environ["LOADING_DB_CHUNK_SIZE"])
    file_name = os.path.basename(file_path)
    table_scheme = TABLES_SCHEMAS[file_name]

    with duckdb.connect(db_path) as db_conn:
        table_reader = pd.read_csv(
            filepath_or_buffer=file_path,
            sep=sep,
            names=table_scheme["columns"],
            chunksize=chunk_size,
            encoding="latin-1",
            encoding_errors="replace",
        )
        try:
            first_chunk = next(table_reader)
        except StopIteration:
            return None
        table_name = table_scheme["table_name"]
        db_conn.execute(
            f"DROP TABLE IF EXISTS {table_name}"
        )
        db_conn.execute(
            f"CREATE TABLE {table_name} "
            "AS SELECT * FROM first_chunk WHERE 1=0",
        )
        db_conn.execute(
            f"INSERT INTO {table_name} SELECT * FROM first_chunk",
        )
        
        for chunk in table_reader:
            db_conn.execute(
                f"INSERT INTO {table_name} SELECT * FROM chunk",
            )
