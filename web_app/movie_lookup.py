"""
Title: Movie web app

Author: Masha Khanina

Description: Module that provides different information about movies.
"""
from abc import ABC, abstractmethod
import os

import aiohttp
import asyncio
from dotenv import load_dotenv
import pandas as pd


REVIEWS_MIN = 100
DATA_DIR = os.environ["DATA_DIR"]


class Movie(ABC):
    """Abstract class representing a movie, with attribute title."""
    @property
    @abstractmethod
    def title(self) -> str:
        """Movie title"""
        pass


class StrMovie(Movie):
    """Class representing movie, initialized by title string."""
    def __init__(self, title: str):
        self._title = title

    @property
    def title(self) -> str:
        """Movie title"""
        return self._title


class ConfigurationError(ValueError):
    """Error to raise when configuration is invalid."""
    pass


class MovieDataBase(ABC):
    """
        Abstract class representing a database wih movies.
        .get_movie_rating(Movie) method returns tuple with
        movie title and rating from the database.
        .get_movie_info(Movie) method returns json with information
        about movie from the database.
    """
    def __init__(self, **flags):
        self._flags = flags

    @abstractmethod
    async def get_movie_rating(
            self, 
            target_movie: Movie,
            session: aiohttp.ClientSession | None = None,
    ) -> tuple[str, str]:
        """
            For given movie, return tuple with movie title and
            rating from the database.
        """
        pass

    @abstractmethod
    async def get_movie_info(
            self,
            target_movie: Movie,
            session: aiohttp.ClientSession | None = None,
    ):
        """
            For given movie, return json with information about it
            from the database.
        """
        pass


class OMDB(MovieDataBase):
    """Class representing OMDB movie database."""
    load_dotenv()
    uri_prefix = f"https://www.omdbapi.com/?apikey={os.getenv('OMDB_KEY')}&t="

    @classmethod
    async def get_movie_info(
            cls, 
            target_movie: Movie,
            session: aiohttp.ClientSession | None = None,
    ):
        if session is None:
            raise ValueError("OMDB database requires aiohttp.ClientSession.")
        async with session.get(cls.uri_prefix + target_movie.title) as response:
            return await response.json()

    @classmethod
    async def get_movie_rating(
            cls, 
            target_movie: Movie,
            session: aiohttp.ClientSession | None = None,
    ) -> tuple[str, str]:
        try:
            movie_info = await cls.get_movie_info(target_movie, session)
            return (
                movie_info["Title"],
                movie_info["imdbRating"],
            )
        except KeyError:
            return ("not found", "-")


def _get_info_movie_lens() -> pd.DataFrame:
    """
        Make pandas dataframe from MovieLens movies file,
        separate title from year.
    """
    movies_categories = [
        "movie_id", "title+year", "release_date",   
        "video_release_date", "imdb_url", "unknown",
        "action", "adventure", "animation", "children",
        "comedy", "crime", "documentary", "drama",
        "fantasy", "film_noir", "horror", "musical",
        "mystery", "romance", "sci_fi",
        "thriller", "war", "western",
    ]
    movies_df = pd.read_table(
        f"{DATA_DIR}/files/movies.csv",
        sep="|",
        names=movies_categories,
        index_col=False,
        encoding_errors="replace",
    )
    movies_df = movies_df.fillna("n/a")

    title_pattern = r"(.+)\s\((\d+)\)"
    movies_df[["title", "release_year"]] = \
        movies_df["title+year"].str.extract(title_pattern)
    movies_df = movies_df.drop(columns="title+year")
    return movies_df


def _average_grades(grades_df: pd.DataFrame) -> pd.DataFrame:
    """
        Extract average grade and amount of reviews for
        each movie from given database with grades.
    """
    grades_df = grades_df.groupby("movie_id", as_index=False).agg(
        rating=("rating", "mean"),
        reviews=("rating", "sum"),
    )
    grades_df["rating"] = 2 * grades_df["rating"].round(2)
    return grades_df


def _preprocess_data() -> pd.DataFrame:
    """
        Make pandas dataframe with movies, their info, average
        ratings and amount of reviews from MovieLens databases.
    """
    info_df = _get_info_movie_lens()
    grades_df = pd.read_table(
        f"{DATA_DIR}/files/ratings.csv",
        sep="\t",
        names=["movie_id", "rating"],
        index_col=False,
        usecols=[1, 2],
    )
    grades_df = grades_df.fillna("n/a")
    rating_df = _average_grades(grades_df)
    main_df = pd.merge(rating_df, info_df, on="movie_id", how="left")
    return main_df.sort_values(by="rating", ascending=False)


class MovieLens(MovieDataBase):
    """Class representing MovieLens movie database."""
    _rated_df = None

    @classmethod
    def _get_by_title(cls, target_movie: Movie) -> pd.DataFrame:
        """Return dataframe with 1 movie which title contains
        given movie's title and biggest amount of reviews."""
        cls._rated_df = _preprocess_data()
        mask = cls._rated_df["title"].str.contains(
            target_movie.title, 
            na=False, 
            case=False,
        )
        return cls._rated_df[mask].nlargest(n=1, columns="reviews")

    @classmethod
    async def get_movie_info(
            cls, 
            target_movie: Movie,
            session: aiohttp.ClientSession | None = None,
    ) -> dict[str, str]:
        try:
            return cls._get_by_title(target_movie).to_dict(orient="records")[0]
        except IndexError:
            return {"title": "not found"}

    @classmethod
    async def get_movie_rating(
            cls, 
            target_movie: Movie,
            session: aiohttp.ClientSession | None = None,
    ) -> tuple[str, str]:
        try:
            return (
                cls._get_by_title(target_movie).iloc[0]["title"],
                str(cls._get_by_title(target_movie).iloc[0]["rating"]),
            )
        except IndexError:
            return ("not found", "-")

    @classmethod
    def get_top_n(cls, n: int = 10, **filters) -> list[dict[str, str]]:
        """
            Return json with n top-rated movies matching given filters
            with at least REVIEWS_MIN reviews.
        """
        if cls._rated_df is None:
            cls._rated_df = _preprocess_data()
        filtered_df = cls._rated_df.copy()
        for label, value in filters.items():
            filtered_df = filtered_df[filtered_df[label] == value]
        popularity_mask = filtered_df["reviews"] > REVIEWS_MIN
        popular_top_df = filtered_df[popularity_mask].head(n)
        return popular_top_df.to_dict(orient="records")


class Config:
    """
        Class representing database.
        Receive string representing database and optional flags
        at instantiation. Flags are passed to database class.
        Method .supported_databases() returns list of names
        of accepted databases.
    """
    _supported_databases = {
        "omdb": OMDB,
        "movie_lens": MovieLens,
    }

    def __init__(self, database: str, **flags):
        try:
            self._database = Config._supported_databases[database](**flags)
        except KeyError:
            raise ConfigurationError("Unknown database")
        self._flags = flags

    @property
    def database(self) -> MovieDataBase:
        """Return configuration's database."""
        return self._database

    @classmethod
    def supported_databases(cls) -> list[str]:
        """
            Return list of databases names that
            can be passed at instantiation.
        """
        return list(Config._supported_databases.keys())


async def get_rating(
        target_movie: Movie, 
        config: Config,
        session: aiohttp.ClientSession | None = None,
) -> tuple[str, str]:
    """
        Return average rating of given movie on
        given configuration's database.
    """
    return await config.database.get_movie_rating(target_movie, session)


async def get_movie_info(
        target_movie: Movie, 
        config: Config,
        session: aiohttp.ClientSession | None = None,
):
    """
        Return json representation of given movie
        received from given configuration's database.
    """
    return await config.database.get_movie_info(target_movie, session)
