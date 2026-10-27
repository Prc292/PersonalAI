import json
import re

from ddgs import DDGS

import weather


MAX_TOOL_ROUNDS = 3
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information, recent events, "
            "news, and facts that may be missing from your knowledge. "
            "Use this when the user asks to search, look up, or verify "
            "information online."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise web search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


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

    if name == "web_search":
        return web_search(arguments.get("query"))

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


def web_search(query: str) -> dict:
    """Search the web and return useful, article-focused results."""

    if not isinstance(query, str) or not query.strip():
        return {"error": "Please provide a search query."}

    query = query.strip()

    if len(query) > 500:
        return {"error": "Search query is too long."}

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=8))

        if not results:
            return {"results": [], "message": "No results found."}

        cleaned_results = []

        for item in results:
            title = item.get("title", "")
            url = item.get("href", "")
            snippet = item.get("body", "")

            # Skip results that lack a usable title or URL.
            if not title or not url:
                continue

            cleaned_results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "date": item.get("date", ""),
            })

        return {
            "results": cleaned_results,
            "count": len(cleaned_results),
            "message": (
                "Use these search results as evidence. Prefer specific "
                "articles or announcements over topic pages. Do not invent "
                "article details that are not supported by the results."
            ),
        }

    except Exception as exc:
        print(f"Web search error: {exc}")
        return {"error": "Web search is temporarily unavailable."}



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
            tools=[WEATHER_TOOL, WEB_SEARCH_TOOL],
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = response.choices[0].message
        print("DEBUG tool calls:", message.tool_calls)
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
                print("DEBUG tool result:", json.dumps(result)[:2000])
            except (ValueError, TypeError, json.JSONDecodeError):
                result = {"error": "The model supplied invalid tool arguments."}

            conversation.append({
    "role": "tool",
    "tool_call_id": call.id,
    "content": json.dumps(result, ensure_ascii=False),
})

    return (
        "I couldn't finish the lookup within the tool-call limit. "
        "Please try your question again."
    )
