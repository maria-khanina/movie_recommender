import os

import duckdb

from loader import load_raw_files, load_table


os.environ["DB_PATH"] = "/tmp/database.duckdb"
load_raw_files("/tmp/movies_files", "1m")
load_table("/tmp/movies_files/ratings.csv", "1m")

load_table("/tmp/movies_files/movies.csv", "1m")
