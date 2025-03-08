from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI



@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

@tool
def get_city_info(city: str) -> str:
    """Fetches basic information about a city from Wikipedia.
    
    Args:
        city: The name of the city.
    
    Returns:
        A short summary about the city.
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{city}"

    try:
        response = requests.get(url)
        data = response.json()
        if "extract" in data:
            return f"🏙️ {city} hakkında bilgi: {data['extract']}"
        else:
            return f"❌ {city} hakkında Wikipedia'da bilgi bulunamadı."
    except Exception as e:
        return f"⚠️ Şehir bilgisi alırken hata oluştu: {str(e)}"




@tool
def find_hotels(city: str) -> str:
    """Finds top hotels in a given city using the free HotelAPI (no API key required).
    
    Args:
        city: The name of the city.
    
    Returns:
        A list of popular hotels in the city.
    """
    url = f"https://api.hotelapi.co/free?location={city}"

    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200 and "hotels" in data:
            hotels = data["hotels"][:5]  # İlk 5 oteli alalım
            hotel_list = "\n".join([f"🏨 {hotel['name']} - ⭐ {hotel['rating']} - 📍 {hotel['address']}" for hotel in hotels])
            return f"📍 {city} içindeki popüler oteller:\n{hotel_list}"
        else:
            return f"❌ {city} için otel bilgisi bulunamadı. Hata: {data.get('message', 'Bilinmeyen hata')}"
    except Exception as e:
        return f"⚠️ Otel bilgisi alırken hata oluştu: {str(e)}"



@tool
def get_weather(city: str) -> str:
    """Fetches the current weather for a given city.
    
    Args:
        city: The name of the city.
    
    Returns:
        A string describing the temperature and weather conditions.
    """
    API_KEY = "b83f226242ec9f5ae14ca6a19918787e"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            temp = data["main"]["temp"]
            weather_desc = data["weather"][0]["description"]
            return f"🌤️ {city} için güncel hava durumu: {temp}°C, {weather_desc}."
        else:
            return f"❌ {city} için hava durumu bilgisi alınamadı. Hata: {data.get('message', 'Bilinmeyen hata')}"
    except Exception as e:
        return f"⚠️ Hava durumu sorgularken hata oluştu: {str(e)}"








final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
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