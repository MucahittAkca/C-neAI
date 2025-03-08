from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import random
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
import os

from Gradio_UI import GradioUI

API_KEY = os.getenv("TMDB_API_KEY")  
TMDB_BASE_URL = "https://api.themoviedb.org/3"

@tool
def get_movie_recommendations(genre: str) -> str:
    """Fetches top-rated, well-known movies from TMDb based on a given genre.

    Args:
        genre (str): The movie genre (e.g., 'Action', 'Drama', 'Comedy').

    Returns:
        str: A list of recommended movies.
    """
    genre_map = {
        "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
        "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
        "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
        "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
        "Thriller": 53, "War": 10752, "Western": 37
    }

    if genre not in genre_map:
        return "Invalid genre. Please try categories like Action, Drama, Comedy, etc."
    
    # **Min vote count ekledik (min 1000 oy alan filmleri al)**
    url = f"{TMDB_BASE_URL}/discover/movie?api_key={API_KEY}&with_genres={genre_map[genre]}&sort_by=vote_average.desc&vote_count.gte=1000"

    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            movies = data["results"][:5]  # İlk 5 filmi getir
            recommendations = "\n".join([f"{movie['title']} ({movie['vote_average']}/10)" for movie in movies])
            return f"Here are some top-rated {genre} movies:\n{recommendations}"
        else:
            return f"Couldn't fetch movie data. Error: {data.get('status_message', 'Unknown error')}"
    except Exception as e:
        return f"Error fetching movie recommendations: {str(e)}"

@tool
def get_similar_tv_shows(show_name: str) -> str:
    """Finds similar TV shows based on the given show's name.

    Args:
        show_name (str): The name of the TV show for which similar shows are requested.

    Returns:
        str: A list of similar TV shows with their ratings.
    """
    url = f"{TMDB_BASE_URL}/search/tv?api_key={API_KEY}&query={show_name}"
    
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200 and data["results"]:
            show_id = data["results"][0]["id"]
            similar_url = f"{TMDB_BASE_URL}/tv/{show_id}/similar?api_key={API_KEY}"
            similar_response = requests.get(similar_url)
            similar_data = similar_response.json()
            
            similar_shows = similar_data["results"][:5]
            recommendations = "\n".join([f"{show['name']} ({show['vote_average']}/10)" for show in similar_shows])
            return f"If you liked '{show_name}', you might also enjoy:\n{recommendations}"
        else:
            return f"Couldn't find recommendations for '{show_name}'."
    except Exception as e:
        return f"Error fetching similar shows: {str(e)}"

@tool
def get_latest_popular_movies() -> str:
    """Fetches the latest popular movies from TMDb."""
    url = f"{TMDB_BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page=1"

    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            movies = data["results"][:5]
            recommendations = "\n".join([f"{movie['title']} ({movie['vote_average']}/10)" for movie in movies])
            return f"Here are the latest popular movies:\n{recommendations}"
        else:
            return f"Couldn't fetch latest movies. Error: {data.get('status_message', 'Unknown error')}"
    except Exception as e:
        return f"Error fetching latest movies: {str(e)}"

@tool
def get_movie_by_mood(mood: str) -> str:
    """Recommends a single movie based on user's mood.

    Args:
        mood (str): The current mood of the user (e.g., 'happy', 'sad', 'excited', 'scared', 'thoughtful').

    Returns:
        str: A single recommended movie based on the mood.
    """
    mood_map = {
        "happy": ["Comedy", "Adventure", "Family"],
        "sad": ["Drama", "Romance"],
        "excited": ["Action", "Thriller"],
        "scared": ["Horror", "Mystery"],
        "thoughtful": ["Science Fiction", "Documentary"]
    }

    mood_lower = mood.lower()
    if "düşük" in mood_lower or "üzgün" in mood_lower or "depresif" in mood_lower:
        normalized_mood = "sad"
    elif "mutlu" in mood_lower or "neşeli" in mood_lower:
        normalized_mood = "happy"
    elif "korkmuş" in mood_lower or "gergin" in mood_lower:
        normalized_mood = "scared"
    elif "heyecanlı" in mood_lower or "enerjik" in mood_lower:
        normalized_mood = "excited"
    elif "düşünceli" in mood_lower or "felsefi" in mood_lower:
        normalized_mood = "thoughtful"
    else:
        return "I couldn't recognize that mood. Try happy, sad, excited, scared, or thoughtful."

    # Rastgele bir tür seç
    genre = random.choice(mood_map[normalized_mood])
    movies_list = get_movie_recommendations(genre)

    # Listeyi parçala ve ilk filmi seç
    movies = movies_list.split('\n')[1:]  # İlk satırı atla ("Here are some top-rated X movies")
    if movies:
        recommended_movie = movies[0].strip()  # İlk filmi al
        return {"movie": recommended_movie}  # Sadece string yerine JSON formatı döndür
    else:
        return {"error": "Couldn't find a good movie for your mood, try again!"}



final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer,get_movie_recommendations, get_similar_tv_shows, get_latest_popular_movies, get_movie_by_mood], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()