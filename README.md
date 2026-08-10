# Movie Recommendation System

An end-to-end movie recommendation product built on the MovieLens 20M dataset:
a training notebook, a FastAPI backend that serves the trained models, and a
Streamlit dashboard on top of the API.

## What it does

Given a movie, it finds similar ones by content. Given a user, it produces
personalized recommendations — blending collaborative filtering, content
similarity, and popularity — and falls back gracefully when a user has little
or no rating history. A poster-grid dashboard sits on top, letting you search,
browse by genre, view a movie's details, and get recommendations

Dataset Link: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset

Deployed Project Link: https://dashboardpy-vxcaswmmgzrpt3nzb8xziw.streamlit.app/?view=home

## Dataset details

[MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) — 20,000,263 ratings and
465,564 tag applications applied to 27,278 movies by 138,493 users between 1995 and 2015.
Files used:

| File | Rows | Used for |
|---|---|---|
| `movie.csv` | 27,278 | titles, genres, catalogue |
| `rating.csv` | 20,000,263 | popularity, collaborative filtering, MF-SVD |
| `tag.csv` | 465,564 | content-based free-text tags |
| `genome_scores.csv` / `genome_tags.csv` | 11.7M / 1,128 | tag-genome relevance weights, folded into the content vector |
| `link.csv` | 27,278 | `movieId -> tmdbId` mapping, used only by the dashboard for poster art |

For training cost, the notebook works on a **filtered subset**: movies with fewer than
`MIN_MOVIE_RATINGS = 100` ratings and a random sample of `USER_SAMPLE = 20,000` users are
excluded from the collaborative-filtering / matrix-factorization matrices (this is the "CF
space" described in `recommenders.py`). The full, unfiltered catalogue (all 27,278 movies)
still supports search and content-based similarity — only the personalized/CF-based scoring
is restricted to the sampled subset. Resulting model footprint: 20,000 users x 8,546 items.

## Architecture

architecture.png

## Project flow

```
MovieLens 20M CSVs (movie.csv, rating.csv, tag.csv, genome_scores.csv, link.csv)
        │
        ▼
movie_recommendation_system.ipynb
   1. Load, clean, explore (EDA)
   2. Popularity baseline (Bayesian-weighted rating)
   3. Content-based model (TF-IDF on genres/tags/title + tag genome, cosine similarity)
   4. Item-based collaborative filtering (shrunk cosine similarity on bias-corrected residuals)
   5. Matrix factorization (truncated SVD)
   6. Hybrid model (weighted blend + 3-tier cold-start routing)
   7. Evaluation (RMSE, MAE, Precision@10, Recall@10, Hit Rate, MAP@10, NDCG@10)
   8. Export cell -> writes artifacts/
        │
        ▼
   artifacts/  (item_similarity.npz, content_matrix.npz, user_likes.npz,
                user_residuals.npz, mf_factors.npz, popularity.npz,
                index_maps.npz, catalogue.csv, metadata.json)
        │
        ▼
movie_recommendation_api/   FastAPI backend — loads artifacts once, never retrains
   /health  /model-info  /movies/search
   /recommend/popular  /recommend/similar  /recommend/user
        │
        ▼
movie_recommendation_dashboard/   Streamlit UI — calls the API over HTTP
   Home (search/browse) -> Details (poster + recommendations) -> For You (personalized)
```

The notebook runs once, offline, to produce `artifacts/`. Everything after that
is inference-only — the API never touches the raw dataset or retrains anything.
TMDB is used **only** by the dashboard, purely to fetch poster/backdrop/overview
art for display; it plays no role in any recommendation, score, or search result.

## Features

- **Popularity baseline** — Bayesian-weighted rating (shrinks small-sample
  averages toward the global mean), with genre filtering.
- **Content-based** — TF-IDF over genres, tags, and title, plus the MovieLens
  tag genome, combined and cosine-compared. Works for any catalogued movie,
  including ones with too few ratings to have a personalized score.
- **Item-based collaborative filtering** — shrunk cosine similarity between
  items, computed on bias-corrected residuals.
- **Matrix factorization** — truncated SVD on the same residuals.
- **Hybrid** — weighted blend of item-CF, content similarity, and popularity,
  z-normalized so the three scales combine sensibly.
- **Poster-grid dashboard** — search, genre-filtered browsing, a details page
  with TMDB art + two recommendation rails, and a personalized/cold-start page.


## Project structure

```
movie_recommendation_system.ipynb   training notebook — the only thing that
                                     touches the raw dataset or fits a model

main.py            FastAPI app: endpoints, request validation, response models
recommenders.py    loads artifacts/, holds all scoring logic (mirrors the
                      notebook's functions exactly)
test_main.py        pytest suite with a synthetic artifacts/ fixture
requirements.txt    full dev requirements (API + dashboard + tests)
requirements-api.txt  lean subset for deployment (no streamlit/pytest)

dashboard.py         Streamlit app: Home / For You / Details views
  (uses the same requirements.txt above)

data/            raw MovieLens CSVs -- gitignored, only needed to run the
                 notebook once; never read by the deployed API or dashboard
artifacts/       notebook output --this is what the deployed API actually loads
docs/            architecture.png (+ .dot source) referenced above
.gitignore
```
## Model evaluation results

Evaluated on a held-out split of the CF-space ratings (20,000 users x 8,546 items).
Numbers below are from `artifacts/metadata.json`, produced by the notebook's evaluation
cell (section 7).

**Rating prediction** (lower is better)

| Model | RMSE | MAE |
|---|---|---|
| Global mean (baseline) | 1.035 | 0.821 |
| Item mean, shrunk | 0.931 | 0.722 |
| Bias baseline (user + item bias) | 0.860 | 0.657 |
| MF-SVD (k=64) | 0.844 | 0.643 |
| **Item-based CF** | **0.831** | **0.622** |

**Top-10 ranking** (higher is better)

| Model | Precision@10 | Recall@10 | Hit Rate@10 | MAP@10 | NDCG@10 |
|---|---|---|---|---|---|
| Popularity | 0.042 | 0.040 | 0.262 | 0.028 | 0.057 |
| Content-based | 0.036 | 0.035 | 0.257 | 0.022 | 0.049 |
| MF-SVD | 0.059 | 0.056 | 0.356 | 0.038 | 0.078 |
| Item-based CF | 0.079 | 0.078 | 0.435 | 0.054 | 0.106 |
| **Hybrid** | **0.090** | **0.089** | **0.468** | **0.061** | **0.119** |


## API usage

Base URL when running locally: `http://localhost:8000`.

```bash
# Health check
curl http://localhost:8000/health
# {"status":"ok","artifacts_loaded":true,"n_users_modelled":20000,"n_items_modelled":8546}

# Search for a movie
curl "http://localhost:8000/movies/search?q=toy+story&limit=5"

# Similar movies (content-based), filtered to Comedy
curl "http://localhost:8000/recommend/similar?movie_id=1&n=10&genre=Comedy"

# Top popular movies overall
curl "http://localhost:8000/recommend/popular?n=10"

# Personalized recommendations for a known user
curl "http://localhost:8000/recommend/user?user_id=123&n=10"

# Cold-start: unknown user with declared genre preferences
curl "http://localhost:8000/recommend/user?user_id=999999999&n=10&preferred_genres=Comedy,Action"
```

Every `/recommend/*` response has the shape:

```json
{
  "strategy": "full hybrid (37 ratings)",
  "count": 10,
  "results": [
    {"movieId": 1, "title": "Toy Story (1995)", "genres": "Animation|Comedy",
     "score": 1.284, "similarity": null, "n_ratings": 49695,
     "avg_rating": 3.921, "tmdbId": 862}
  ]
}
```

`strategy` always reports which tier served the request (e.g. `cold-start: new user,
popularity within preferred genres [...]`, `content + popularity (sparse user, N
ratings)`, or `full hybrid (N ratings)`), so callers can tell a personalized result from
a fallback one. Invalid `movieId`/`userId` combinations, or filters that produce no
results, return `404` with a JSON `detail` message instead of an empty `200`.

## Assumptions and limitations

- **CF space is a sample, not the full dataset.** Only movies with >= 100 ratings and a
  20,000-user sample are eligible for item-CF / MF-SVD / hybrid scoring. Movies or users
  outside that sample still work for search and content-based similarity, but never get a
  personalized `score` — they're served through the cold-start / content path instead.
- **Cold-start is popularity-within-genre, not personalized.** A brand-new user gets the
  most popular movies in their chosen genres, not a taste-matched list — it's a
  reasonable default, not a substitute for learned preferences, and will look the same
  for every new user who picks the same genres.
- **The "sparse user" tier (< 5 ratings) blends content + popularity only** — it
  deliberately skips item-CF, since collaborative signal is unreliable with that little
  history. This trades some potential accuracy for stability.
- **TMDB poster/overview data is best-effort.** If a `tmdbId` is missing, the API request
  fails, or no key is configured, the dashboard falls back to a placeholder — it never
  blocks or errors on missing poster art.
- **Single-machine, in-memory service.** Artifacts are loaded fully into memory on
  startup; there's no sharding or external model store, so deployment size is bounded by
  the artifacts directory fitting in the host's RAM.

## Setup & run (brief)

1. Run the notebook end-to-end (needs `data/` populated) to produce `artifacts/`.
2. API: `pip install -r requirements.txt` then
   `python -m uvicorn main:app --reload --port 8000`.
3. Dashboard: `pip install -r requirements.txt` then
   `python -m streamlit run dashboard.py`.
4. Set `TMDB_API_KEY` via `.env` locally, or via Streamlit secrets when deployed — it's
   never entered or shown in the running app.

## Demonstration video



