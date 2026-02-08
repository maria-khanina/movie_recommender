import os

import duckdb

from loader import load_raw_files, load_table

dir = "/home/maria-khanina/mkgit/movies_project/data"
files = os.path.join(dir, "files")

os.environ["DB_PATH"] = os.path.join(dir, "database.duckdb")
load_raw_files(files, "1m")
load_table(os.path.join(files, "ratings.csv"), "1m")

load_table(os.path.join(files, "movies.csv"), "1m")
