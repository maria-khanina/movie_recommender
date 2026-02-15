"""
Title: Loader

Author: Masha Khanina
Reviewer: Shani Gendelis

Description: Module for loading MovieLens files into storage and
parsing them to database.

Infinity Labs R&D AI #5.
"""
import os
from pathlib import Path
import shutil
from typing import Literal
import zipfile

import duckdb
import requests


GENRES = [
    "unknown", "action", "adventure", "animation", "children",
    "comedy", "crime", "documentary", "drama",
    "fantasy", "film_noir", "horror", "musical",
    "mystery", "romance", "sci_fi",
    "thriller", "war", "western",
]


RATINGS_COLS = ["user_id", "movie_id", "rating", "timestamp"]


TABLES_SCHEMAS = {
    "users": {
        "file_name": {"100k": "u.user", "1m": "users.dat"},
        "separator": {"100k": "|", "1m": "::"}, 
        "columns": {
            "100k": ["user_id", "age", "gender", "occupation", "zip_code"],
            "1m": ["user_id", "gender", "age", "occupation", "zip_code"],
        },
    },

    "movies": {
        "file_name": {"100k": "u.item", "1m": "movies.dat"},
        "separator": {"100k": "|", "1m": "::"},
        "columns": {
            "100k": [
                "movie_id", "title", "release_date", "vid_release", 
                "IMDB_URL", 
            ] + GENRES,
            "1m": ["movie_id", "title", "genres"],
        },
    },

    "ratings": {
        "file_name": {"100k": "u.data", "1m": "ratings.dat"},
        "separator": {"100k": "\t", "1m": "::"},
        "columns": {
            "100k": RATINGS_COLS,
            "1m": RATINGS_COLS,
        },
    }
}


def load_raw_files(dest_dir: str, dataset_name: Literal["100k", "1M"]):
    """Load MovieLens files with users, movies, and ratings."""
    os.makedirs(dest_dir, exist_ok=True)
    temp_zip_dest = os.path.join(dest_dir, f"ml-{dataset_name}.zip")
    url = f"https://files.grouplens.org/datasets/movielens/ml-{dataset_name}.zip"
    
    with requests.get(url, stream=True) as response:
        with open(temp_zip_dest, mode="wb") as zip_file:
            for chunk in response.iter_content(8192):
                zip_file.write(chunk)
    with zipfile.ZipFile(temp_zip_dest, mode="r") as zip_file:
        for dest_name in TABLES_SCHEMAS:
            source_name = TABLES_SCHEMAS[dest_name]["file_name"][dataset_name]
            file_path = os.path.join(f"ml-{dataset_name}", source_name)
            with zip_file.open(file_path) as src:
                with open(os.path.join(
                        dest_dir, 
                        f"{dest_name}.csv"),
                        "wb",
                ) as dest:
                    shutil.copyfileobj(src, dest)
    os.remove(temp_zip_dest)


def load_table(file_path: str, dataset_name: Literal["100k", "1m"]):
    """
        Parse csv file to a database. 
        Possible file names are: users.csv, movies.csv, ratings.csv.
        Fits for large files (1_000_000_000 rows).
    """
    db_path = os.environ["DB_PATH"]
    table_name = Path(file_path).stem
    columns = TABLES_SCHEMAS[table_name]["columns"][dataset_name]
    sep = TABLES_SCHEMAS[table_name]["separator"][dataset_name]

    with duckdb.connect(db_path) as db_conn:
        db_conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} AS
            SELECT * FROM read_csv_auto(
                '{file_path}',
                delim='{sep}',
                header=false,
                column_names={columns},
                encoding='latin-1',
                parallel=True
            )
        """)


if __name__ == "__main__":
    DATASET = "1m"
    FILES_DIR = os.path.join(os.environ["DATA_DIR"], "files")
    load_raw_files(FILES_DIR, DATASET)
    for table in TABLES_SCHEMAS:
        csv_file = os.path.join(FILES_DIR, f"{table}.csv")
        load_table(csv_file, DATASET)
