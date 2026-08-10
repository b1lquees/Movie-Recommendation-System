"""
FastAPI backend for the MovieLens recommendation system.

Run:
    uvicorn main:app --reload --port 8000

Expects an artifacts/ directory (see recommenders.py for the exact contract,
produced by the training notebook's export cell) next to this file, or set
ARTIFACT_DIR env var to point elsewhere.
"""
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from recommenders import MovieNotFoundError, RecommenderService

service: Optional[RecommenderService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    artifact_dir = os.environ.get("ARTIFACT_DIR", "artifacts")
    try:
        service = RecommenderService(artifact_dir)
    except FileNotFoundError as e:
        # Let the app boot so /health reports the problem cleanly instead of crashing.
        print(f"[startup] WARNING: {e}")
        service = None
    yield


app = FastAPI(
    title="Movie Recommendation API",
    description="Popularity, content-based, item-based collaborative filtering, and "
                "hybrid movie recommendations served from pre-trained artifacts.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_service() -> RecommenderService:
    if service is None:
        artifact_dir = os.environ.get("ARTIFACT_DIR", "artifacts")
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts not loaded from '{artifact_dir}'. "
                   "Run the training notebook's export cell and restart the API.",
        )
    return service

# response models
class Movie(BaseModel):
    movieId: int
    title: str
    genres: str
    year: Optional[float] = None
    in_cf_model: Optional[bool] = None
    tmdbId: Optional[int] = None


class ScoredMovie(BaseModel):
    movieId: int
    title: str
    genres: str
    score: Optional[float] = None
    similarity: Optional[float] = None
    n_ratings: Optional[int] = None
    avg_rating: Optional[float] = None
    tmdbId: Optional[int] = None


class RecommendationResponse(BaseModel):
    strategy: str
    count: int
    results: List[ScoredMovie]


class HealthResponse(BaseModel):
    status: str
    artifacts_loaded: bool
    n_users_modelled: Optional[int] = None
    n_items_modelled: Optional[int] = None


def df_to_scored(df: pd.DataFrame) -> List[dict]:
    df = df.replace({np.nan: None})
    return df.to_dict(orient="records")


# endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    if service is None:
        return HealthResponse(status="degraded", artifacts_loaded=False)
    return HealthResponse(
        status="ok",
        artifacts_loaded=True,
        n_users_modelled=service.metadata.get("n_users_modelled"),
        n_items_modelled=service.metadata.get("n_items_modelled"),
    )


@app.get("/model-info")
def model_info():
    svc = require_service()
    return svc.metadata


@app.get("/movies/search", response_model=List[Movie])
def search_movies(
    q: str = Query(..., min_length=1, description="Substring to search for in movie titles"),
    limit: int = Query(10, ge=1, le=50),
):
    svc = require_service()
    df = svc.search_movies(q, limit=limit)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No movies found matching '{q}'")
    return df_to_scored(df)


@app.get("/recommend/popular", response_model=RecommendationResponse)
def recommend_popular(
    n: int = Query(10, ge=1, le=100),
    genre: Optional[str] = Query(None, description="Filter to a genre, e.g. 'Comedy'"),
):
    svc = require_service()
    df = svc.top_popular(n=n, genre=genre)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No movies found for genre '{genre}'" if genre else "No movies available",
        )
    return RecommendationResponse(
        strategy="popularity (Bayesian-weighted rating)",
        count=len(df),
        results=df_to_scored(df),
    )


@app.get("/recommend/similar", response_model=RecommendationResponse)
def recommend_similar(
    movie_id: int = Query(..., description="movieId to find similar movies for"),
    n: int = Query(10, ge=1, le=100),
    genre: Optional[str] = Query(None),
):
    svc = require_service()
    try:
        df = svc.similar_movies(movie_id, n=n, genre=genre)
    except MovieNotFoundError:
        raise HTTPException(status_code=404, detail=f"movieId {movie_id} not found. Try /movies/search first.")
    if df.empty:
        raise HTTPException(status_code=404, detail="No similar movies found matching the given filters")
    return RecommendationResponse(
        strategy="content-based (TF-IDF genres/tags/title + tag genome, cosine similarity)",
        count=len(df),
        results=df_to_scored(df),
    )


@app.get("/recommend/user", response_model=RecommendationResponse)
def recommend_user(
    user_id: int = Query(..., description="userId to generate personalised recommendations for"),
    n: int = Query(10, ge=1, le=100),
    genre: Optional[str] = Query(None),
    preferred_genres: Optional[str] = Query(
        None, description="Comma-separated genres, used only when user_id is unknown (cold start)"
    ),
):
    svc = require_service()
    pref_list = [g.strip() for g in preferred_genres.split(",")] if preferred_genres else None
    df, strategy = svc.recommend_for_user(user_id, n=n, genre=genre, preferred_genres=pref_list)
    if df.empty:
        raise HTTPException(status_code=404, detail="No recommendations found matching the given filters")
    return RecommendationResponse(strategy=strategy, count=len(df), results=df_to_scored(df))
