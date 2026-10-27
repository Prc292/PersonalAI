
import os

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather(city: str | None = None) -> dict:
    """Get current weather, using the configured home city when none is given."""
    city = (city or os.getenv("PERSONALAI_HOME_CITY", "")).strip()
    if not city:
        return {
            "error": "No weather location was specified and PERSONALAI_HOME_CITY is not configured."
        }

    with httpx.Client(timeout=10.0) as client:
        location_response = client.get(
            GEOCODING_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
        )
        location_response.raise_for_status()
        locations = location_response.json().get("results", [])

        if not locations:
            return {"error": f"I couldn't find a location called {city}."}

        place = locations[0]

        weather_response = client.get(
            WEATHER_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "auto",
                "forecast_days": 1,
            },
        )
        weather_response.raise_for_status()
        data = weather_response.json()

    current = data["current"]
    daily = data["daily"]

    return {
        "location": f'{place["name"]}, {place.get("admin1", "")}'.strip(", "),
        "temperature_f": current["temperature_2m"],
        "feels_like_f": current["apparent_temperature"],
        "humidity_percent": current["relative_humidity_2m"],
        "wind_mph": current["wind_speed_10m"],
        "precipitation_mm": current["precipitation"],
        "high_f": daily["temperature_2m_max"][0],
        "low_f": daily["temperature_2m_min"][0],
        "precipitation_chance_percent": daily["precipitation_probability_max"][0],
        "weather_code": current["weather_code"],
        "updated_at": current["time"],
    }
