import requests

def get_weather(city):
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    geocoding_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }
    try:
        response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10
        )
        response.raise_for_status()
        location_data = response.json()
    except requests.RequestException as e:
        return {
            "error": f"Could not connect to the weather service: {e}"
        }
    if "results" not in location_data:
        return {
            "error": f"Could not find the city '{city}'."
        }
    if not location_data["results"]:
        return {
            "error": f"Could not find the city '{city}'."
        }
    location = location_data["results"][0]
    latitude = location["latitude"]
    longitude = location["longitude"]
    city_name = location["name"]
    country = location.get("country", "")
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "weather_code,"
            "wind_speed_10m"
        ),
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh"
    }
    try:
        response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )
        response.raise_for_status()
        weather_data = response.json()
    except requests.RequestException as e:
        return {
            "error": f"Could not get weather data: {e}"
        }
    current = weather_data.get("current")

    if not current:
        return {
            "error": "Weather data not available."
        }
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Heavy rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    weather_code = current.get("weather_code")

    condition = weather_codes.get(
        weather_code,
        "Unknown"
    )
    return {
        "city": city_name,
        "country": country,
        "temperature_c": current.get("temperature_2m"),
        "feels_like_c": current.get("apparent_temperature"),
        "humidity_percent": current.get("relative_humidity_2m"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "condition": condition
    }
# these weather APIs are used from geocoding free weather API keys
'''
if __name__ == "__main__":
    result = get_weather("Delhi")
    print(result)
or
if __name__ == "__main__":
    result = get_weather("Mumbai")
    print(result)
or
if __name__ == "__main__":
    result = get_weather("America")
    print(result)
'''