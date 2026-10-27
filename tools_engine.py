
import json
import re

import weather


MAX_TOOL_ROUNDS = 3


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get current weather and today's forecast for a city. "
            "Use this for questions about current conditions or forecasts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city and optional region or country.",
                }
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


def run_tool(name: str, arguments: dict) -> dict:
    """Run an approved tool with validated arguments."""

    if name != "get_weather":
        return {"error": f"Unknown tool: {name}"}

    city = arguments.get("city")

    if not isinstance(city, str) or not city.strip():
        return {"error": "Please provide a valid city name."}

    city = city.strip()

    if len(city) > 120:
        return {"error": "City name is too long."}

    try:
        result = weather.get_weather(city)
        return result
    except Exception:
        return {"error": "Weather service is temporarily unavailable."}


def chat_with_tools(
    request_client,
    model_name: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 700,
) -> str:
    """Run a bounded tool-calling conversation."""

    conversation = list(messages)

    for _ in range(MAX_TOOL_ROUNDS + 1):
        response = request_client.chat.completions.create(
            model=model_name,
            messages=conversation,
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            return message.content or "(The model returned an empty response.)"

        conversation.append(message.model_dump(exclude_none=True))

        for call in tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be an object.")
                result = run_tool(call.function.name, arguments)
            except (ValueError, TypeError, json.JSONDecodeError):
                result = {"error": "The model supplied invalid tool arguments."}

            conversation.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return (
        "I couldn't finish the lookup within the tool-call limit. "
        "Please try your question again."
    )
