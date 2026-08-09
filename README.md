# Movie Recommendation System

An end-to-end movie recommendation product built on the MovieLens 20M dataset:
a training notebook, a FastAPI backend that serves the trained models, and a
Streamlit dashboard on top of the API. 

## 1. Produce the artifacts (one-time)

Run `movie_recommendation_system.ipynb` end to end, with a `data/` folder next
to it containing the extracted MovieLens 20M CSVs (`movie.csv`, `rating.csv`,
`tag.csv`, `genome_scores.csv`, ...). The final cell ("Exporting artifacts for
the API layer") writes an `artifacts/` folder containing:

```
item_similarity.npz    content_matrix.npz    user_likes.npz
user_residuals.npz      mf_factors.npz         popularity.npz
index_maps.npz          catalogue.csv           metadata.json
```

Copy that `artifacts/` folder into `movie_recommendation_api/`, next to `main.py`.

This step needs real compute/RAM (20M ratings) — expect it to take a while on a
laptop, faster on Colab/Kaggle with more RAM.

## 2. Run the API

```bash
cd movie_recommendation_api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

If `artifacts/` is missing, the app still boots — `/health` reports
`"artifacts_loaded": false` and every other endpoint returns `503` with a clear
message, instead of crashing at startup.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + whether artifacts loaded |
| `GET /model-info` | Training config + evaluation metrics from `metadata.json` |
| `GET /movies/search?q=&limit=` | Title substring search |
| `GET /recommend/popular?n=&genre=` | Bayesian-weighted popularity baseline |
| `GET /recommend/similar?movie_id=&n=&genre=` | Content-based (TF-IDF genres/tags/title + tag genome, cosine) |
| `GET /recommend/user?user_id=&n=&genre=&preferred_genres=` | Hybrid, with cold-start routing |

### Cold-start behaviour (`/recommend/user`)

Mirrors the notebook's `recommend()` exactly — three tiers based on how much is
known about the user:

1. **Unseen `user_id`** (never appeared in training): falls back to popularity,
   optionally filtered to `preferred_genres` (comma-separated).
2. **Known user with fewer than 5 ratings, or no "liked" items to build a taste
   profile from**: blends content score with popularity, 50/50.
3. **Known user with 5+ ratings and a usable taste profile**: full hybrid —
   item-based CF + content profile + popularity, weighted per `metadata.json`'s
   config (`W_CF` / `W_CONTENT` / `W_POP`, defaults 0.4 / 0.2 / 0.4).

Every response includes a `strategy` field naming which tier was used.

**Limitation:** there's no live-write path for a brand-new user to rate a few
movies and get an immediate profile — this is a read-only service over a static
training snapshot. The dashboard papers over this with a genre picker as a
first-session taste signal until an online-update path exists.

### Tests

```bash
cd movie_recommendation_api
pytest test_main.py -v
```

The suite builds a tiny synthetic `artifacts/` fixture (6 movies, 3 users, one
user in each cold-start tier) so it runs in ~1s without needing the real 20M-row
dataset. Also smoke-tested against artifacts produced by an actual (small) run
of the training notebook, not just the hand-built fixture. Covers: health check,
valid recommendation requests, invalid movie titles, all three
user-recommendation tiers, and output-format/field validation.

## 3. Run the dashboard

Make sure the API is running first, then:

```bash
cd movie_recommendation_dashboard
pip install -r requirements.txt
streamlit run dashboard.py
```

By default it looks for the API at `http://localhost:8000`. Override with:

```bash
API_BASE_URL=http://your-api-host:8000 streamlit run dashboard.py
```

...or just edit the "API base URL" field in the sidebar once it's open.

### What's on each page

- **Search & Similar Movies** — type part of a title, pick from the matches,
  get content-based similar movies. Genre filter included.
- **Personalized Recommendations** — enter a user ID for hybrid recommendations,
  or toggle "I'm a new user" for the cold-start flow (pick a few favorite genres,
  get popular movies within them). The `strategy` field from the API is always
  shown above the results, so it's visible which tier served the answer.
- **Popular & Top-Rated** — the Bayesian-weighted popularity baseline, with a
  genre filter and an adjustable result count (sidebar slider, applies to every
  page).

The sidebar also shows live API health (users/movies modelled) and a "Model
info" expander with the full `metadata.json` (training config + evaluation
metrics). If the API is unreachable or its artifacts aren't loaded, every page
shows a clear error instead of a stack trace.

## Assumptions and limitations

- Only movies/users that survive `MIN_MOVIE_RATINGS` / `USER_SAMPLE` filtering
  during training get a personalized (CF/hybrid) score; movies outside that
  filtered set can still be searched and used for content-based similarity, but
  not for `/recommend/user`.
- Item-based CF and SVD matrix factorization are the two collaborative-filtering
  methods implemented; user-based CF was evaluated and deliberately excluded at
  full 20M scale (the notebook explains why).
- New-user cold start is genre-based popularity, not a "rate a few movies and
  get an instant profile" flow — that would need a live write path into the
  training artifacts, which this read-only service doesn't have.
