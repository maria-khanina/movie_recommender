"""
Title: Movie web app

Author: Masha Khanina

Description: Saving users' requests and movie app responses to a database.
"""
import os
from datetime import datetime

from pymongo import MongoClient


user = os.environ['DB_USER']
password = os.environ['DB_PASSWORD']
host = os.environ['DB_SERVICE']

url = f"mongodb://{user}:{password}@{host}:27017/"
client = MongoClient(url)

searches_db = client["searches_db"]
movies_searched = searches_db["movies_searches"]
top_searched = searches_db["tops_searched"]


def log_movie_search(title, request, response, **kwargs):
    """Save request to get rating/info about movies to a database."""
    movies_searched.insert_one({
        "timestamp": datetime.now().timestamp(),
        "title": title,
        "request": request,
        "response": dict(response),
        **kwargs
    })

def log_top_search(filters, response):
    """Save top-n request and response to a database."""
    top_searched.insert_one({
        "timestamp": datetime.now().timestamp(),
        "response": response,
        **filters
    })
