"""
Streamlit dashboard for the Movie Recommendation API: search or browse a
poster grid, open a movie for details + recommendations, or get personalized
recommendations with cold-start handling.

Run:
    streamlit run dashboard.py

Config: set API_BASE_URL and TMDB_API_KEY via Streamlit Cloud secrets, a
.env file, or shell env vars. Neither is shown or editable in the UI.
"""
import os

import requests
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_config(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


API_BASE_URL = get_config("API_BASE_URL", "http://localhost:8000")
TMDB_API_KEY = get_config("TMDB_API_KEY", "")
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

if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

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


def api_get(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params or {}, timeout=10)
    except requests.exceptions.RequestException as e:
        return False, f"Couldn't reach the API -- {e}"
    if r.status_code == 200:
        return True, r.json()
    try:
        detail = r.json().get("detail", r.text)
    except Exception:
        detail = r.text
    return False, f"{r.status_code}: {detail}"


@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_movie_details(tmdb_id, api_key: str):
    if not tmdb_id or not api_key:
        return None
    try:
        r = requests.get(f"{TMDB_BASE}/movie/{int(tmdb_id)}", params={"api_key": api_key}, timeout=8)
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


def poster_url_for(tmdb_id):
    details = tmdb_movie_details(tmdb_id, TMDB_API_KEY) if tmdb_id else None
    return details["poster_url"] if details else None


def poster_grid(movies, cols: int = 6, key_prefix: str = "grid"):
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
                poster = poster_url_for(m.get("tmdbId"))
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


def render_model_info():
    ok, info = api_get("/model-info")
    if not ok:
        st.caption(info)
        return

    st.table([
        {"Field": "Dataset", "Value": str(info.get("dataset", "-"))},
        {"Field": "Trained at", "Value": str(info.get("trained_at", "-"))},
        {"Field": "Users modelled", "Value": str(info.get("n_users_modelled", "-"))},
        {"Field": "Items modelled", "Value": str(info.get("n_items_modelled", "-"))},
        {"Field": "Movies in catalogue", "Value": str(info.get("n_movies_in_catalogue", "-"))},
    ])

    config = info.get("config", {})
    if config:
        st.markdown("**Training config**")
        st.table([{"Parameter": k, "Value": str(v)} for k, v in config.items()])

    rating_metrics = info.get("rating_metrics", {})
    if rating_metrics:
        st.markdown("**Rating prediction**")
        st.table([{"Model": name, **vals} for name, vals in rating_metrics.items()])

    ranking_metrics = info.get("ranking_metrics", {})
    if ranking_metrics:
        st.markdown("**Top-10 ranking**")
        st.table([{"Model": name, **vals} for name, vals in ranking_metrics.items()])


GENRE_OPTIONS = [
    "Any", "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]

with st.sidebar:
    st.markdown("## 🎬 Menu")
    if st.button(" Home"):
        goto_home()
    if st.button(" For You"):
        goto_for_you()

    st.markdown("---")
    ok, health = api_get("/health")
    if ok and health.get("artifacts_loaded"):
        st.success(
            f"Connected -- {health.get('n_users_modelled', '?'):,} users, "
            f"{health.get('n_items_modelled', '?'):,} movies modelled"
        )
    else:
        st.error(health if not ok else "API is up but artifacts aren't loaded.")

    st.markdown("---")
    st.markdown("###  Home feed")
    home_genre = st.selectbox("Category (genre)", GENRE_OPTIONS, index=0)
    home_sort = st.radio("Sort by", ["Popular (weighted)", "Top rated (avg rating)"])
    grid_cols = st.slider("Grid columns", 3, 8, 6)

    with st.expander("Model info"):
        render_model_info()


st.title("🎬 Movie Recommender")
st.markdown(
    "<div class='small-muted'>Search or browse -> open a movie -> details + recommendations</div>",
    unsafe_allow_html=True,
)
st.divider()


if st.session_state.view == "home":
    typed = st.text_input("Search by movie title", placeholder="Type: toy story, godfather, matrix...")
    st.divider()

    if typed.strip():
        ok, results = api_get("/movies/search", {"q": typed.strip(), "limit": 24})
        if not ok:
            st.error(results)
        elif not results:
            st.info("No movies found.")
        else:
            st.markdown("### Results")
            poster_grid(results, cols=grid_cols, key_prefix="search")
    else:
        st.markdown(f"###  Home — {home_genre.title() if home_genre != 'Any' else 'All genres'} "
                    f"({'Popular' if 'Popular' in home_sort else 'Top rated'})")

        ok, rec = api_get("/recommend/popular",
                          {"n": max(grid_cols * 4, 24), "genre": None if home_genre == "Any" else home_genre})
        if not ok:
            st.error(rec)
        else:
            movies = rec["results"]
            if "Top rated" in home_sort:
                movies = sorted(movies, key=lambda m: m.get("avg_rating") or 0, reverse=True)
            poster_grid(movies[:grid_cols * 4], cols=grid_cols, key_prefix="home_feed")


elif st.session_state.view == "for_you":
    st.header("Personalized recommendations")

    is_new_user = st.toggle("I'm a new user (no rating history)")
    genre_filter = st.selectbox("Filter by genre", GENRE_OPTIONS, key="fy_genre")

    if is_new_user:
        st.write("Pick a few genres you enjoy and we'll recommend popular movies within them.")
        preferred = st.multiselect("Genres you like", GENRE_OPTIONS[1:])
        if st.button("Get recommendations", type="primary"):
            ok, rec = api_get("/recommend/user", {
                "user_id": 999_999_999, "n": grid_cols * 3,
                "genre": None if genre_filter == "Any" else genre_filter,
                "preferred_genres": ",".join(preferred) if preferred else None,
            })
            if not ok:
                st.error(rec)
            else:
                st.markdown(f"<div class='strategy-badge'>Strategy: {rec['strategy']}</div>",
                           unsafe_allow_html=True)
                poster_grid(rec["results"], cols=grid_cols, key_prefix="for_you_new")
    else:
        user_id = st.number_input("Your user ID", min_value=1, step=1, value=1)
        if st.button("Get recommendations", type="primary"):
            ok, rec = api_get("/recommend/user", {
                "user_id": int(user_id), "n": grid_cols * 3,
                "genre": None if genre_filter == "Any" else genre_filter,
            })
            if not ok:
                st.error(rec)
            else:
                st.markdown(f"<div class='strategy-badge'>Strategy: {rec['strategy']}</div>",
                           unsafe_allow_html=True)
                poster_grid(rec["results"], cols=grid_cols, key_prefix="for_you_known")


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

    details = tmdb_movie_details(movie.get("tmdbId"), TMDB_API_KEY)

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
        st.write((details or {}).get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    if details and details.get("backdrop_url"):
        st.markdown("#### Backdrop")
        st.image(details["backdrop_url"], use_container_width=True)

    st.divider()
    st.markdown("###  Recommendations")

    ok, sim = api_get("/recommend/similar", {"movie_id": movie["movieId"], "n": grid_cols * 2})
    st.markdown("####  Similar Movies (content-based)")
    if ok:
        poster_grid(sim["results"], cols=grid_cols, key_prefix="details_similar")
    else:
        st.info(sim)

    primary_genre = movie.get("genres", "").split("|")[0] if movie.get("genres") else None
    if primary_genre and primary_genre != "(no genres listed)":
        ok2, pop = api_get("/recommend/popular", {"genre": primary_genre, "n": grid_cols * 2})
        st.markdown(f"####  More Like This ({primary_genre})")
        if ok2:
            poster_grid(
                [m for m in pop["results"] if m["movieId"] != movie["movieId"]],
                cols=grid_cols, key_prefix="details_genre",
            )
        else:
            st.info(pop)
