"""
Basic API tests. Builds a tiny synthetic artifacts/ directory (5 movies, 3 users)
matching the notebook's export contract, so the suite runs in under a second
without needing the real MovieLens 20M dataset.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    rng = np.random.default_rng(0)

    movies = pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5, 6],
        "title": ["Toy Story (1995)", "Jumanji (1995)", "Heat (1995)",
                  "GoldenEye (1995)", "Sabrina (1995)", "Casino (1995)"],
        "clean_title": ["Toy Story", "Jumanji", "Heat", "GoldenEye", "Sabrina", "Casino"],
        "year": [1995.0] * 6,
        "genres": ["Animation|Comedy", "Adventure|Fantasy", "Action|Crime",
                   "Action|Adventure", "Comedy|Romance", "Crime|Drama"],
        "in_cf_model": [True, True, True, True, True, True],
        "tmdbId": [862, 8844, 949, 710, 11860, 8963],
    })
    movies.to_csv(d / "catalogue.csv", index=False)

    n_items, n_users, n_features = 6, 3, 8

    content = sp.random(n_items, n_features, density=0.5, format="csr",
                         random_state=0, data_rvs=rng.random).astype(np.float32)
    sp.save_npz(d / "content_matrix.npz", content)

    item_sim = sp.random(n_items, n_items, density=0.4, format="csr",
                          random_state=1, data_rvs=rng.random).astype(np.float32)
    sp.save_npz(d / "item_similarity.npz", item_sim)

    # user 0 (raw id 10): 2 liked items -> full hybrid path
    # user 1 (raw id 20): 1 liked item, but < 5 total ratings -> sparse-user path
    # user 2 (raw id 30): 0 ratings at all -- still "known" (present in index_maps),
    #                     exercises the sparse path with an empty profile
    user_likes = sp.csr_matrix(
        (np.array([1, 1], dtype=np.float32), (np.array([0, 1]), np.array([0, 2]))),
        shape=(n_users, n_items),
    )
    sp.save_npz(d / "user_likes.npz", user_likes)

    # residuals: user 0 has 5 ratings out of 6 movies (full hybrid eligible, movie
    # index 5 left unseen so there's something to recommend), user 1 has 2 ratings
    # (sparse path), user 2 has 0
    residual_rows = [0, 0, 0, 0, 0, 1, 1]
    residual_cols = [0, 1, 2, 3, 4, 0, 2]
    residual_vals = rng.random(len(residual_rows)).astype(np.float32)
    user_residuals = sp.csr_matrix(
        (residual_vals, (residual_rows, residual_cols)), shape=(n_users, n_items)
    )
    sp.save_npz(d / "user_residuals.npz", user_residuals)

    np.savez_compressed(
        d / "mf_factors.npz",
        P=rng.random((n_users, 4)).astype(np.float32),
        Q=rng.random((n_items, 4)).astype(np.float32),
        b_user=np.zeros(n_users, dtype=np.float32),
        b_item=np.zeros(n_items, dtype=np.float32),
        mu=np.float32(3.5),
    )

    np.savez_compressed(
        d / "popularity.npz",
        scores=np.array([4.1, 3.2, 3.8, 2.9, 3.5, 3.6], dtype=np.float32),
        counts=np.array([100, 40, 60, 20, 55, 70]),
        means=np.array([4.2, 3.1, 3.9, 2.8, 3.4, 3.7], dtype=np.float32),
    )

    np.savez(
        d / "index_maps.npz",
        user_ids=np.array([10, 20, 30]),
        movie_ids=np.array([1, 2, 3, 4, 5, 6]),
    )

    metadata = {
        "trained_at": "2026-01-01T00:00:00",
        "dataset": "synthetic-test-fixture",
        "n_users_modelled": n_users,
        "n_items_modelled": n_items,
        "config": {"W_CF": 0.4, "W_CONTENT": 0.2, "W_POP": 0.4},
        "rating_metrics": {},
        "ranking_metrics": {},
    }
    with open(d / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return str(d)


@pytest.fixture(scope="module")
def client(artifact_dir):
    os.environ["ARTIFACT_DIR"] = artifact_dir
    from main import app  # imported after ARTIFACT_DIR is set, so startup picks it up
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["artifacts_loaded"] is True
    assert body["n_items_modelled"] == 6


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["dataset"] == "synthetic-test-fixture"


def test_search_movies_valid(client):
    r = client.get("/movies/search", params={"q": "toy"})
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 1
    assert results[0]["title"] == "Toy Story (1995)"


def test_search_movies_no_match(client):
    r = client.get("/movies/search", params={"q": "nonexistent-movie-xyz"})
    assert r.status_code == 404


def test_popular_recommendations(client):
    r = client.get("/recommend/popular", params={"n": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    scores = [row["score"] for row in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_popular_recommendations_genre_filter(client):
    r = client.get("/recommend/popular", params={"n": 5, "genre": "Comedy"})
    assert r.status_code == 200
    for row in r.json()["results"]:
        assert "comedy" in row["genres"].lower()


def test_similar_valid_movie(client):
    r = client.get("/recommend/similar", params={"movie_id": 1, "n": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] <= 3
    assert all(row["movieId"] != 1 for row in body["results"])  # never recommends itself


def test_similar_invalid_movie_id(client):
    r = client.get("/recommend/similar", params={"movie_id": 999999})
    assert r.status_code == 404


def test_recommend_full_hybrid_user(client):
    r = client.get("/recommend/user", params={"user_id": 10, "n": 3})
    assert r.status_code == 200
    assert "full hybrid" in r.json()["strategy"]


def test_recommend_sparse_user(client):
    r = client.get("/recommend/user", params={"user_id": 20, "n": 3})
    assert r.status_code == 200
    assert "sparse" in r.json()["strategy"]


def test_recommend_unknown_user_cold_start(client):
    r = client.get("/recommend/user", params={"user_id": 999999, "n": 3})
    assert r.status_code == 200
    body = r.json()
    assert "cold-start" in body["strategy"]


def test_recommend_unknown_user_with_preferred_genres(client):
    r = client.get("/recommend/user", params={
        "user_id": 999999, "n": 5, "preferred_genres": "Comedy,Romance",
    })
    assert r.status_code == 200
    for row in r.json()["results"]:
        genres = row["genres"].lower()
        assert "comedy" in genres or "romance" in genres


def test_output_format_fields(client):
    r = client.get("/recommend/popular", params={"n": 1})
    row = r.json()["results"][0]
    for field in ["movieId", "title", "genres", "score"]:
        assert field in row


def test_missing_required_param_is_validated(client):
    r = client.get("/recommend/similar")  # movie_id is required
    assert r.status_code == 422

    r = client.get("/recommend/user")  # user_id is required
    assert r.status_code == 422
