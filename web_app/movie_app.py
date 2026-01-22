"""
Title: Movie web app

Author: Masha Khanina

Description: Web application for learning about movies and
getting personal recommendations.
"""
import os
from typing import Any, Literal

import aiohttp
from contextlib import asynccontextmanager
import duckdb
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


import movie_lookup
from users_actions import log_movie_search, log_top_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session = aiohttp.ClientSession()
    yield
    await app.state.session.close()


app = FastAPI(lifespan=lifespan)


app.mount(
    "/static",
    StaticFiles(directory="frontend/static"),
    name="static",
)


@app.get("/")
def root():
    """Load main html page."""
    return FileResponse("frontend/index.html")


@app.get("/recommend/{user_id}")
def recommend_movies(user_id: str) -> list[str]:
    """
        Return 5 titles of movies recommmended for given MovieLens user.
    """
    if user_id == "":
        return ["Enter user id, please"]
    try:
        user_id = int(user_id)
    except ValueError:
        return ["Enter number, please"]
    try:
        with duckdb.connect(os.environ["DB_PATH"], read_only=True) as db_conn:
            results = db_conn.execute(
                "SELECT title FROM recommendations WHERE user_id=?",
                [user_id],
            ).fetchall()
    except (duckdb.CatalogException, duckdb.IOException):
        return ["Recommendation isn't ready yet, try later."]

    titles = [row[0] for row in results]
    return titles


@app.get("/{database}/{title}/rating")
async def get_rating(
        database: str, 
        title: str,
) -> dict[Literal["movie_title", "rating"], str] | Literal["unavailable"]:
    """Return rating of movie with given title on given database."""
    try:
        real_movie = await _get_full_title_movie(title)
    except FileNotFoundError:
        return "unavailable"
    
    rating = await movie_lookup.get_rating(
            real_movie,
            movie_lookup.Config(database),
            app.state.session,
    )
    log_movie_search(
        title=title, 
        database=database, 
        request="rating", 
        response={rating[0]: rating[1]}
    )
    return {"movie_title": rating[0], "rating": rating[1]}
    

@app.get("/movie_lens/top/")
def get_top_n(
        n: str = "", 
        genre: str = "", 
        release_year: str = ""
) -> list[dict[str, str | int | float]] | Literal["unavailable"]:
    """
        Return n top-rated movies with given genre and release_year
        and at least 100 reviews on MovieLens.
    """
    filters = {}

    if genre:
        filters["_".join(genre.lower().split(" "))] = 1
    if release_year and release_year.strip():
        filters["release_year"] = release_year
    try:
        n = int(n)
    except ValueError:
        n = 10

    try:
        response = movie_lookup.MovieLens.get_top_n(
            n=n,
            **filters,
        )
    except FileNotFoundError:
        return "unavailable"

    logged_response = [{
        "movie_id": top_movie["movie_id"], 
        "title": top_movie["title"]}
        for top_movie in response
    ]
    log_top_search(filters | {"n": n}, logged_response)
    return response


@app.get("/movies/{title}")
async def get_info(
    title: str,
) -> list[tuple[str, str]] | Literal["unavailable"]:
    """
        Return list of (key, value) pairs representing information
        about the movie received from MovieLens and OMDB.
    """
    omdb_labels = ["RunTime", "Genre", "Director", "Writer", "Rated"]
    m_lens_labels = [
        "release_year", "release_date", "video_release_date",
        "imdb_url",
    ]
    try:
        real_movie = await _get_full_title_movie(title)
    except FileNotFoundError:
        return "unavailable"
    movie_info = {"Title": real_movie.title}
    _add_info(
        movie_info,
        m_lens_labels,
        await movie_lookup.get_movie_info(
            real_movie, 
            movie_lookup.Config("movie_lens"), 
            app.state.session,
        ),
    )
    _add_info(
        movie_info,
        omdb_labels,
        await movie_lookup.get_movie_info(
            real_movie, 
            movie_lookup.Config("omdb"), 
            app.state.session,
        ),
    )

    log_movie_search(
        title=title,
        request="info",
        response=movie_info.items()
    )
    return list(movie_info.items())


async def _get_full_title_movie(title: str) -> movie_lookup.StrMovie:
    """
        Return a movie with full title fitting to the given partial title.
    """
    init_movie = movie_lookup.StrMovie(title)
    movie_info = await movie_lookup.get_movie_info(
        init_movie,
        movie_lookup.Config("movie_lens"),
        app.state.session,
    )
    movie_title = movie_info["title"]
    if movie_title == "not found":
        try:
            movie_info = await movie_lookup.get_movie_info(
                init_movie,
                movie_lookup.Config("omdb"),
                app.state.session,
            )
            movie_title = movie_info["Title"]
        except KeyError:
            pass
    return movie_lookup.StrMovie(movie_title)


def _add_info(
        dest_info: dict[str, str], 
        labels: list[str],
        source_info: dict[str, Any]
) -> None:
    """
        Add items from source_info to dest_info if item key is in
        labels list.
    """
    for label in labels:
        formatted_label = " ".join(label.split("_")).capitalize()
        try:
            dest_info[formatted_label] = source_info[label]
        except KeyError:
            continue
