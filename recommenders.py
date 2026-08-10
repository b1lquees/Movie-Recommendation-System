"""
Inference-only service that mirrors the notebook's models and scoring functions
exactly -- same weighting, same cold-start tiers -- loading everything from the
artifacts/ directory the notebook's export cell produces. No retraining here.

    artifacts/
        item_similarity.npz    S_item: item-item CF similarity (CF item order)
        content_matrix.npz     C_all: TF-IDF (genres/tags/title) + tag genome (full catalogue order)
        user_likes.npz         L: binary "rated >= LIKE_THRESHOLD in train" (n_users x n_items, CF order)
        user_residuals.npz     R_res: bias-corrected train ratings (n_users x n_items, CF order) --
                                needed to score item-based CF for a known user at request time
        mf_factors.npz         P, Q, b_user, b_item, mu
        popularity.npz         scores, counts, means
        index_maps.npz         user_ids, movie_ids: matrix-index -> raw ID (CF space)
        catalogue.csv          movieId, title, clean_title, year, genres, in_cf_model
        metadata.json          training config + evaluation results

"CF space" = the users/movies that survived MIN_MOVIE_RATINGS / USER_SAMPLE
filtering during training. A movieId can exist in the catalogue (searchable,
similar-movies works) without existing in CF space (no personalised score).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize


class MovieNotFoundError(KeyError):
    pass


REQUIRED_ARTIFACTS = [
    "item_similarity.npz", "content_matrix.npz", "user_likes.npz",
    "user_residuals.npz", "mf_factors.npz", "popularity.npz",
    "index_maps.npz", "catalogue.csv", "metadata.json",
]


class RecommenderService:
    def __init__(self, artifact_dir):
        self.dir = Path(artifact_dir)
        self._load()

    def _load(self):
        d = self.dir
        missing = [f for f in REQUIRED_ARTIFACTS if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifact(s) in {d}: {missing}. "
                "Run the training notebook's export cell first."
            )

        self.catalogue = pd.read_csv(d / "catalogue.csv")
        self.catalogue["genres"] = self.catalogue["genres"].fillna("")
        if "tmdbId" in self.catalogue.columns:
            self.catalogue["tmdbId"] = self.catalogue["tmdbId"].astype("Int64")
        else:
            self.catalogue["tmdbId"] = pd.array([None] * len(self.catalogue), dtype="Int64")
        self._title_lower = self.catalogue["title"].str.lower()
        self._row_by_movie_id = {m: i for i, m in enumerate(self.catalogue["movieId"].to_numpy())}

        with open(d / "metadata.json") as f:
            self.metadata = json.load(f)

        idx = np.load(d / "index_maps.npz")
        self.cf_user_ids = idx["user_ids"]
        self.cf_movie_ids = idx["movie_ids"]
        self._user_row = {u: i for i, u in enumerate(self.cf_user_ids)}
        self._cf_row_by_movie_id = {m: i for i, m in enumerate(self.cf_movie_ids)}
        cat_by_id = self.catalogue.set_index("movieId")
        self._cf_title = cat_by_id["title"].reindex(self.cf_movie_ids).to_numpy()
        self._cf_genres = cat_by_id["genres"].reindex(self.cf_movie_ids).fillna("").to_numpy()
        self._cf_tmdb = cat_by_id["tmdbId"].reindex(self.cf_movie_ids).to_numpy()

        pop = np.load(d / "popularity.npz")
        self.pop_scores = pop["scores"].astype(np.float32)
        self.pop_counts = pop["counts"]
        self.pop_means = pop["means"]

        self.content_matrix = sp.load_npz(d / "content_matrix.npz").tocsr()   # full catalogue order
        content_rows = [self._row_by_movie_id[m] for m in self.cf_movie_ids]
        self.content_cf = normalize(self.content_matrix[content_rows]).tocsr().astype(np.float32)

        self.item_sim = sp.load_npz(d / "item_similarity.npz").tocsr()
        self.item_sim_T = self.item_sim.T.tocsr()

        self.user_likes = sp.load_npz(d / "user_likes.npz").tocsr()
        self.user_residuals = sp.load_npz(d / "user_residuals.npz").tocsr()

        mf = np.load(d / "mf_factors.npz")
        self.P, self.Q = mf["P"], mf["Q"]
        self.b_user, self.b_item = mf["b_user"], mf["b_item"]
        self.mu = float(mf["mu"])

        # NOTE: taste profiles are built per-user, on demand (see _content_scores) --
        # never eagerly for all users at once. A full (n_users x n_features) matmul
        # here is exactly the memory blow-up the notebook's user_likes.npz export was
        # meant to avoid, and it's enough to OOM a small deployment box.

        cfg = self.metadata.get("config", {})
        self.w_cf = float(cfg.get("W_CF", 0.4))
        self.w_content = float(cfg.get("W_CONTENT", 0.2))
        self.w_pop = float(cfg.get("W_POP", 0.4))

        self.pop_z = self._z1(self.pop_scores)

    @staticmethod
    def _z1(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-8)

    def _genre_mask_cf(self, genre: Optional[str]):
        if not genre:
            return None
        g = genre.lower()
        return np.array([g in gg.lower() for gg in self._cf_genres])

    def _genre_mask_catalogue(self, genre: Optional[str]):
        if not genre:
            return None
        return self.catalogue["genres"].str.lower().str.contains(genre.lower(), regex=False).to_numpy()

    def movie_row_catalogue(self, movie_id: int):
        return self._row_by_movie_id.get(movie_id)

    def user_row(self, user_id: int):
        return self._user_row.get(user_id)

    def _topk(self, scores: np.ndarray, n: int) -> np.ndarray:
        n = min(n, int(np.isfinite(scores).sum()))
        if n <= 0:
            return np.array([], dtype=int)
        top = np.argpartition(-scores, n - 1)[:n]
        return top[np.argsort(-scores[top])]

    # --- item-based CF and content scoring, matching the notebook's functions ---
    def _itemcf_scores(self, u: int) -> np.ndarray:
        return np.asarray((self.user_residuals[u] @ self.item_sim_T).todense()).ravel()

    def _user_profile(self, u: int):
        """This user's liked-item centroid, built on demand -- one sparse row-vector
        matmul, not the full (n_users x n_features) matrix."""
        profile = self.user_likes[u] @ self.content_cf
        nnz = profile.nnz if sp.issparse(profile) else np.count_nonzero(profile)
        if nnz == 0:
            return profile, 0
        return normalize(profile), nnz

    def _content_scores(self, u: int) -> np.ndarray:
        profile, _ = self._user_profile(u)
        return np.asarray((profile @ self.content_cf.T).todense()).ravel()

    # ------------------------------------------------------------------ #
    def search_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        hit = self._title_lower.str.contains(query.lower(), regex=False)
        return self.catalogue[hit].head(limit)[
            ["movieId", "title", "genres", "year", "in_cf_model", "tmdbId"]
        ].reset_index(drop=True)

    def top_popular(self, n: int = 10, genre: Optional[str] = None) -> pd.DataFrame:
        mask = self._genre_mask_cf(genre)
        scores = self.pop_scores.copy()
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        top = self._topk(scores, n)
        return pd.DataFrame({
            "movieId": self.cf_movie_ids[top],
            "title": self._cf_title[top],
            "genres": self._cf_genres[top],
            "tmdbId": self._cf_tmdb[top],
            "n_ratings": self.pop_counts[top].astype(int),
            "avg_rating": self.pop_means[top].round(3),
            "score": scores[top].round(4),
        })

    def similar_movies(self, movie_id: int, n: int = 10, genre: Optional[str] = None) -> pd.DataFrame:
        row = self.movie_row_catalogue(movie_id)
        if row is None:
            raise MovieNotFoundError(f"movieId {movie_id} not found in catalogue")
        sims = np.asarray((self.content_matrix[row] @ self.content_matrix.T).todense()).ravel()
        sims[row] = -np.inf
        mask = self._genre_mask_catalogue(genre)
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
        top = self._topk(sims, n)
        out = self.catalogue.iloc[top][["movieId", "title", "genres", "year", "tmdbId"]].copy()
        out["similarity"] = sims[top].round(4)
        return out.reset_index(drop=True)

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 10,
        genre: Optional[str] = None,
        preferred_genres: Optional[list] = None,
    ):
        """Mirrors the notebook's recommend(): switches strategy on how much is known
        about the user. Returns (recommendations_df, strategy_used)."""
        genre_mask = self._genre_mask_cf(genre)
        u = self.user_row(user_id)

        # Tier 1: unseen user -- never appeared in training at all.
        if u is None:
            combined_mask = genre_mask
            if preferred_genres:
                pref_mask = np.zeros(len(self.cf_movie_ids), dtype=bool)
                for g in preferred_genres:
                    pref_mask |= self._genre_mask_cf(g)
                combined_mask = pref_mask if combined_mask is None else (combined_mask & pref_mask)
                strategy = f"cold-start: new user, popularity within preferred genres {preferred_genres}"
            else:
                strategy = "cold-start: new user, global popularity (no genre signal given)"
            scores = self.pop_scores.copy()
            if combined_mask is not None:
                scores = np.where(combined_mask, scores, -np.inf)
            top = self._topk(scores, n)
            out = pd.DataFrame({
                "movieId": self.cf_movie_ids[top],
                "title": self._cf_title[top],
                "genres": self._cf_genres[top],
                "tmdbId": self._cf_tmdb[top],
                "score": scores[top].round(4),
            })
            return out, strategy

        seen = set(self.user_residuals[u].indices)
        _, liked_count = self._user_profile(u)

        # Tier 2: known user, but too little signal for a reliable content profile.
        if len(seen) < 5 or liked_count == 0:
            strategy = f"content + popularity (sparse user, {len(seen)} ratings)"
            s = 0.5 * self._z1(self._content_scores(u)) + 0.5 * self.pop_z
        # Tier 3: full hybrid.
        else:
            strategy = f"full hybrid ({len(seen)} ratings)"
            s = (
                self.w_cf * self._z1(self._itemcf_scores(u))
                + self.w_content * self._z1(self._content_scores(u))
                + self.w_pop * self.pop_z
            )

        s = s.copy()
        s[list(seen)] = -np.inf   # never recommend something the user already rated
        if genre_mask is not None:
            s = np.where(genre_mask, s, -np.inf)

        top = self._topk(s, n)
        out = pd.DataFrame({
            "movieId": self.cf_movie_ids[top],
            "title": self._cf_title[top],
            "genres": self._cf_genres[top],
            "tmdbId": self._cf_tmdb[top],
            "score": s[top].round(4),
        })
        return out, strategy