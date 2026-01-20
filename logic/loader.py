import os

import duckdb
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


def load_raw_files():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    


def init_counter_table(db_conn: duckdb.DuckDBPyConnection):
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS rows_counter(
            file_path TEXT PRIMARY KEY,
            last_row_loaded BIGINT
        )
    """)


def load_table(file_path: str, sep: str = "|"):
    db_path = os.environ["DB_PATH"]
    chunk_size = int(os.environ["LOADING_DB_CHUNK_SIZE"])
    file_name = os.path.basename(file_path)
    table_scheme = TABLES_SCHEMAS[file_name]

    with duckdb.connect(db_path) as db_conn:
        init_counter_table(db_conn)
        start_row = db_conn.execute(
            "SELECT last_row_loaded FROM rows_counter WHERE "
            "file_path=?", [file_path]
        ).fetchone()
        if start_row is None:
            start_row = 0
        else:
            start_row = start_row[0]
        table_reader = pd.read_csv(
            filepath_or_buffer=file_path,
            sep=sep,
            names=table_scheme["columns"],
            skiprows=start_row,
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
            f"CREATE TABLE IF NOT EXISTS {table_name} "
            "AS SELECT * FROM first_chunk WHERE 1=0",
        )
        db_conn.execute(
            f"INSERT INTO {table_name} SELECT * FROM first_chunk",
        )
        
        rows_added = len(first_chunk)
        for chunk in table_reader:
            db_conn.execute(
                f"INSERT INTO {table_name} SELECT * FROM chunk",
            )
            rows_added = len(chunk)
        
        stop_row = start_row + rows_added
        db_conn.execute(
            "INSERT OR REPLACE INTO rows_counter VALUES (?, ?)",
            [file_path, stop_row]
        )
