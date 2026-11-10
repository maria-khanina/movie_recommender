from typing import Literal, TypedDict

DatasetName = Literal["100k", "1m"]
TableName = Literal["users", "movies", "ratings"]


class TableSchema(TypedDict):
    file_name: dict[DatasetName, str]
    separator: dict[DatasetName, str]
    columns: dict[DatasetName, list[str]]


DATASET_NAME: DatasetName = "1m"
ARCHIVE_URL_TEMPLATE = "https://files.grouplens.org/datasets/movielens/ml-{name}.zip"
ARCHIVE_DIRECTORY_TEMPLATE = "ml-{name}"
DOWNLOAD_CHUNK_BYTES = 8192
DOWNLOAD_TIMEOUT_SECONDS = 60
RAW_FILES_SUBDIRECTORY = "files"
RAW_FILE_EXTENSION = ".csv"
RAW_FILE_ENCODING = "latin-1"
DATA_DIRECTORY_VARIABLE = "DATA_DIR"
DATABASE_PATH_VARIABLE = "DB_PATH"

GENRES_100K: list[str] = [
    "unknown",
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film_noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci_fi",
    "thriller",
    "war",
    "western",
]
RATINGS_COLUMNS: list[str] = ["user_id", "movie_id", "rating", "timestamp"]

TABLE_SCHEMAS: dict[TableName, TableSchema] = {
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
                "movie_id",
                "title",
                "release_date",
                "vid_release",
                "IMDB_URL",
                *GENRES_100K,
            ],
            "1m": ["movie_id", "title", "genres"],
        },
    },
    "ratings": {
        "file_name": {"100k": "u.data", "1m": "ratings.dat"},
        "separator": {"100k": "\t", "1m": "::"},
        "columns": {"100k": RATINGS_COLUMNS, "1m": RATINGS_COLUMNS},
    },
}

GENRE_SEPARATOR = "|"
RELEASE_YEAR_PATTERN = r"\((\d{4})\)\s*$"
RELEASE_PERIOD_YEARS = 5
RELEASE_PERIOD_COLUMN = "release_period"
NON_GENRE_MOVIE_COLUMNS = ("title", RELEASE_PERIOD_COLUMN)
UNKNOWN_FEATURE_CODE = 0
USER_FEATURE_COLUMNS: list[str] = ["gender", "age", "occupation"]
