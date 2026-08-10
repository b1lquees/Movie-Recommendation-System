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
.gitignore
```
## Setup & run (brief)

1. Run the notebook end-to-end (needs `data/` populated) to produce `artifacts/`.
2. API: `pip install -r requirements.txt` then
   `python -m uvicorn main:app --reload --port 8000`.
3. Dashboard: `pip install -r requirements.txt` then
   `python -m streamlit run dashboard.py`.
4. Set Api Key` via `.env` locally, or via secrets when deployed — it's
   never entered or shown in the running app.


