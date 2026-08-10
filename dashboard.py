"""
Streamlit dashboard for the Movie Recommendation API, styled as a poster-grid
browsing experience: search or browse -> click a poster -> details page with
overview + two recommendation rails.

Run:
    streamlit run dashboard.py

Config (env vars, or edit in the sidebar once running):
    API_BASE_URL   -- our FastAPI backend, default http://localhost:8000
    TMDB_API_KEY   -- TMDB v3 API key, used ONLY for poster/backdrop/overview art.
                      Get one free at https://www.themoviedb.org/settings/api
                      Recommendations, search, and scoring all come from our own
                      API -- TMDB is purely a decoration layer here.
"""
import os

import requests
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed -- env vars set directly in the shell still work


def get_config(name: str, default: str = "") -> str:
    """Streamlit Cloud secrets first (st.secrets), then env vars / .env, then default."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # no secrets.toml configured (normal for local runs) -- fall through
    return os.environ.get(name, default)


DEFAULT_API_BASE_URL = get_config("API_BASE_URL", "http://localhost:8000")
DEFAULT_TMDB_API_KEY = get_config("TMDB_API_KEY", "")

# Safe fallbacks so these names always exist even if something upstream errors
# before the sidebar widgets that normally set them run.
base_url = DEFAULT_API_BASE_URL
tmdb_api_key = DEFAULT_TMDB_API_KEY

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
.small-muted { color:#6b7280; font-size: 0.92rem; }
.movie-title { font-size: 0.9rem; line-height: 1.15rem; height: 2.3rem; overflow: hidden; margin-top: 4px; }
.card { border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.7); }
.strategy-badge { color:#6b7280; font-size: 0.85rem; font-style: italic; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# routing (single-file, query-param based, same pattern as the reference app)
# --------------------------------------------------------------------------- #
if "view" not in st.session_state:
    st.session_state.view = "home"          # home | details | for_you
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None  # dict: movieId, title, genres, tmdbId

qp_view = st.query_params.get("view")
if qp_view in ("home", "details", "for_you"):
    st.session_state.view = qp_view


def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "movieId" in st.query_params:
        del st.query_params["movieId"]
    st.rerun()


def goto_for_you():
    st.session_state.view = "for_you"
    st.query_params["view"] = "for_you"
    st.rerun()


def goto_details(movie: dict):
    st.session_state.view = "details"
    st.session_state.selected_movie = movie
    st.query_params["view"] = "details"
    st.query_params["movieId"] = str(movie["movieId"])
    st.rerun()


# --------------------------------------------------------------------------- #
# our recommendation API
# --------------------------------------------------------------------------- #
def api_get(base_url: str, path: str, params: dict | None = None):
    """GET against our API. Returns (ok, data_or_error_message)."""
    try:
        r = requests.get(f"{base_url}{path}", params=params or {}, timeout=10)
    except requests.exceptions.RequestException as e:
        return False, f"Couldn't reach the API at {base_url} -- {e}"
    if r.status_code == 200:
        return True, r.json()
    try:
        detail = r.json().get("detail", r.text)
    except Exception:
        detail = r.text
    return False, f"{r.status_code}: {detail}"


# --------------------------------------------------------------------------- #
# TMDB (posters / backdrop / overview only -- never recommendations or scoring)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_movie_details(tmdb_id, api_key: str):
    if not tmdb_id or not api_key:
        return None
    try:
        r = requests.get(f"{TMDB_BASE}/movie/{int(tmdb_id)}",
                         params={"api_key": api_key}, timeout=8)
        if r.status_code != 200:
            return None
        d = r.json()
        return {
            "poster_url": f"{TMDB_IMG}{d['poster_path']}" if d.get("poster_path") else None,
            "backdrop_url": f"{TMDB_IMG}{d['backdrop_path']}" if d.get("backdrop_path") else None,
            "overview": d.get("overview"),
            "release_date": d.get("release_date"),
            "genres": [g["name"] for g in d.get("genres", [])],
            "vote_average": d.get("vote_average"),
        }
    except requests.exceptions.RequestException:
        return None


def poster_url_for(tmdb_id, api_key: str):
    details = tmdb_movie_details(tmdb_id, api_key) if tmdb_id else None
    return details["poster_url"] if details else None


# --------------------------------------------------------------------------- #
# shared UI pieces
# --------------------------------------------------------------------------- #
def poster_grid(movies, api_key: str, cols: int = 6, key_prefix: str = "grid"):
    """movies: list of dicts with at least movieId, title, genres, tmdbId."""
    if not movies:
        st.info("No movies to show.")
        return

    rows = (len(movies) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(movies):
                break
            m = movies[idx]
            idx += 1

            with colset[c]:
                poster = poster_url_for(m.get("tmdbId"), api_key)
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='height:220px; background:#e5e7eb; border-radius:8px; "
                        "display:flex; align-items:center; justify-content:center; color:#9ca3af;'>"
                        "🎬 No poster</div>",
                        unsafe_allow_html=True,
                    )

                if st.button("Open", key=f"{key_prefix}_{r}_{c}_{idx}_{m['movieId']}"):
                    goto_details(m)

                st.markdown(f"<div class='movie-title'>{m['title']}</div>", unsafe_allow_html=True)

                badge = m.get("score") if m.get("score") is not None else m.get("similarity")
                label = "score" if m.get("score") is not None else "similarity"
                if badge is not None:
                    st.caption(f"{label}: {badge:.3f}")


GENRE_OPTIONS = [
    "Any", "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🎬 Menu")
    if st.button("🏠 Home"):
        goto_home()
    if st.button("⭐ For You"):
        goto_for_you()

    st.markdown("---")
    base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL)
    # TMDB key is intentionally NOT shown or editable here -- it's loaded silently
    # from Streamlit Cloud secrets / .env and never rendered in the UI.
    tmdb_api_key = DEFAULT_TMDB_API_KEY

    ok, health = api_get(base_url, "/health")
    if ok and health.get("artifacts_loaded"):
        st.success(
            f"Connected -- {health.get('n_users_modelled', '?'):,} users, "
            f"{health.get('n_items_modelled', '?'):,} movies modelled"
        )
    elif ok:
        st.error("API is up but artifacts aren't loaded. Run the notebook's export "
                  "cell and restart the API with artifacts/ next to it.")
    else:
        st.error(health)

    if not tmdb_api_key:
        st.caption("No TMDB key configured -- posters will show as placeholders.")
    else:
        st.caption("Poster images enabled.")

    st.markdown("---")
    st.markdown("### 🏠 Home feed")
    home_genre = st.selectbox("Category (genre)", GENRE_OPTIONS, index=0)
    home_sort = st.radio("Sort by", ["Popular (weighted)", "Top rated (avg rating)"])
    grid_cols = st.slider("Grid columns", 3, 8, 6)

    with st.expander("Model info"):
        info_ok, info = api_get(base_url, "/model-info")
        st.json(info) if info_ok else st.caption(info)


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
st.title("🎬 Movie Recommender")
st.markdown(
    "<div class='small-muted'>Search or browse -> open a movie -> details + recommendations</div>",
    unsafe_allow_html=True,
)
st.divider()


# ============================================================================ #
# VIEW: HOME
# ============================================================================ #
if st.session_state.view == "home":
    typed = st.text_input("Search by movie title", placeholder="Type: toy story, godfather, matrix...")
    st.divider()

    if typed.strip():
        ok, results = api_get(base_url, "/movies/search", {"q": typed.strip(), "limit": 24})
        if not ok:
            st.error(results)
        elif not results:
            st.info("No movies found.")
        else:
            st.markdown("### Results")
            poster_grid(results, tmdb_api_key, cols=grid_cols, key_prefix="search")
    else:
        st.markdown(f"### 🏠 Home — {home_genre.title() if home_genre != 'Any' else 'All genres'} "
                    f"({'Popular' if 'Popular' in home_sort else 'Top rated'})")

        ok, rec = api_get(base_url, "/recommend/popular",
                          {"n": max(grid_cols * 4, 24), "genre": None if home_genre == "Any" else home_genre})
        if not ok:
            st.error(rec)
        else:
            movies = rec["results"]
            if "Top rated" in home_sort:
                movies = sorted(movies, key=lambda m: m.get("avg_rating") or 0, reverse=True)
            poster_grid(movies[:grid_cols * 4], tmdb_api_key, cols=grid_cols, key_prefix="home_feed")


# ============================================================================ #
# VIEW: FOR YOU (personalized, with cold-start)
# ============================================================================ #
elif st.session_state.view == "for_you":
    st.header("Personalized recommendations")

    is_new_user = st.toggle("I'm a new user (no rating history)")
    genre_filter = st.selectbox("Filter by genre", GENRE_OPTIONS, key="fy_genre")

    if is_new_user:
        st.write("Pick a few genres you enjoy and we'll recommend popular movies within them.")
        preferred = st.multiselect("Genres you like", GENRE_OPTIONS[1:])
        user_id = 999_999_999  # outside any real training sample -> guaranteed cold-start
        if st.button("Get recommendations", type="primary"):
            ok, rec = api_get(base_url, "/recommend/user", {
                "user_id": user_id, "n": grid_cols * 3, "genre": None if genre_filter == "Any" else genre_filter,
                "preferred_genres": ",".join(preferred) if preferred else None,
            })
            if not ok:
                st.error(rec)
            else:
                st.markdown(f"<div class='strategy-badge'>Strategy: {rec['strategy']}</div>",
                           unsafe_allow_html=True)
                poster_grid(rec["results"], tmdb_api_key, cols=grid_cols, key_prefix="for_you_new")
    else:
        user_id = st.number_input("Your user ID", min_value=1, step=1, value=1)
        if st.button("Get recommendations", type="primary"):
            ok, rec = api_get(base_url, "/recommend/user", {
                "user_id": int(user_id), "n": grid_cols * 3,
                "genre": None if genre_filter == "Any" else genre_filter,
            })
            if not ok:
                st.error(rec)
            else:
                st.markdown(f"<div class='strategy-badge'>Strategy: {rec['strategy']}</div>",
                           unsafe_allow_html=True)
                poster_grid(rec["results"], tmdb_api_key, cols=grid_cols, key_prefix="for_you_known")


# ============================================================================ #
# VIEW: DETAILS
# ============================================================================ #
else:
    movie = st.session_state.selected_movie
    if not movie:
        st.warning("No movie selected. Go back to Home and click a poster's 'Open' button.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    top_a, top_b = st.columns([3, 1])
    with top_a:
        st.markdown("### 📄 Movie Details")
    with top_b:
        if st.button("← Back to Home"):
            goto_home()

    details = tmdb_movie_details(movie.get("tmdbId"), tmdb_api_key)

    left, right = st.columns([1, 2.4], gap="large")
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if details and details.get("poster_url"):
            st.image(details["poster_url"], use_container_width=True)
        else:
            st.write("🎬 No poster available")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {movie['title']}")
        release = (details or {}).get("release_date") or "-"
        genres = (details or {}).get("genres")
        genres_str = ", ".join(genres) if genres else movie.get("genres", "-")
        st.markdown(f"<div class='small-muted'>Release: {release}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Genres: {genres_str}</div>", unsafe_allow_html=True)
        if details and details.get("vote_average"):
            st.markdown(f"<div class='small-muted'>TMDB rating: {details['vote_average']}/10</div>",
                       unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Overview")
        st.write((details or {}).get("overview") or "No overview available (no TMDB match for this title).")
        st.markdown("</div>", unsafe_allow_html=True)

    if details and details.get("backdrop_url"):
        st.markdown("#### Backdrop")
        st.image(details["backdrop_url"], use_container_width=True)

    st.divider()
    st.markdown("### ✅ Recommendations")

    ok, sim = api_get(base_url, "/recommend/similar", {"movie_id": movie["movieId"], "n": grid_cols * 2})
    st.markdown("#### 🔎 Similar Movies (content-based)")
    if ok:
        poster_grid(sim["results"], tmdb_api_key, cols=grid_cols, key_prefix="details_similar")
    else:
        st.info(sim)

    primary_genre = movie.get("genres", "").split("|")[0] if movie.get("genres") else None
    if primary_genre and primary_genre != "(no genres listed)":
        ok2, pop = api_get(base_url, "/recommend/popular", {"genre": primary_genre, "n": grid_cols * 2})
        st.markdown(f"#### 🎭 More Like This ({primary_genre})")
        if ok2:
            poster_grid(
                [m for m in pop["results"] if m["movieId"] != movie["movieId"]],
                tmdb_api_key, cols=grid_cols, key_prefix="details_genre",
            )
        else:
            st.info(pop)
