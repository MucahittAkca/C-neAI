from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import random
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

API_KEY = "c172a615aaeb6ba28fa8a91bedfd8ebe"  # TMDb API anahtarını buraya ekle
TMDB_BASE_URL = "https://api.themoviedb.org/3"

@tool
def get_movie_recommendations(genre: str) -> str:
    """Fetches top-rated movies from IMDb (via TMDb API) based on a given genre.
    Args:
        genre: The movie genre (e.g., 'Action', 'Drama', 'Comedy').
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
    
    url = f"{TMDB_BASE_URL}/discover/movie?api_key={API_KEY}&with_genres={genre_map[genre]}&sort_by=vote_average.desc"

    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            movies = data["results"][:5]
            recommendations = "\n".join([f"{movie['title']} ({movie['vote_average']}/10)" for movie in movies])
            return f"Here are some top-rated {genre} movies:\n{recommendations}"
        else:
            return f"Couldn't fetch movie data. Error: {data.get('status_message', 'Unknown error')}"
    except Exception as e:
        return f"Error fetching movie recommendations: {str(e)}"

@tool
def get_similar_tv_shows(show_name: str) -> str:
    """Finds similar TV shows based on the given show's name."""
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
    """Recommends a movie based on user's mood."""
    mood_map = {
        "happy": ["Comedy", "Adventure", "Family"],
        "sad": ["Drama", "Romance"],
        "excited": ["Action", "Thriller"],
        "scared": ["Horror", "Mystery"],
        "thoughtful": ["Science Fiction", "Documentary"]
    }

    if mood not in mood_map:
        return "I couldn't recognize that mood. Try happy, sad, excited, scared, or thoughtful."
    
    genre = random.choice(mood_map[mood])
    return get_movie_recommendations(genre)





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
    tools=[final_answer, get_weather, get_city_info, find_hotels, get_current_time_in_timezone], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()