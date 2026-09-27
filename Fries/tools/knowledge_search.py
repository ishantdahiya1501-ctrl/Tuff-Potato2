import requests
from bs4 import BeautifulSoup

def wikipedia_search(query):
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "exintro": True,
        "titles": query,
        "redirects": 1
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers={
                "User-Agent": "FlexAI/1.0"
            },
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        pages = data.get("query", {}).get("pages", {})

        for page in pages.values():

            if "missing" in page:
                return None

            title = page.get("title", "")
            extract = page.get("extract", "").strip()

            if not extract:
                return None

            sentences = extract.split(". ")

            if len(sentences) > 8:
                extract = ". ".join(sentences[:8]) + "."

            return {
                "source": "Wikipedia",
                "title": title,
                "content": extract,
                "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            }
    except Exception:
        return None
    return None
def web_search(query, max_results=5):
    url = "https://html.duckduckgo.com/html/"
    try:
        response = requests.post(
            url,
            data={
                "q": query
            },
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=10
        )
        response.raise_for_status()
        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )
        results = []
        for result in soup.select(".result"):
            title = result.select_one(".result__title")
            link = result.select_one(".result__url")
            snippet = result.select_one(".result__snippet")
            if not title or not link:
                continue
            result_data = {
                "title": title.get_text(" ", strip=True),
                "url": link.get("href"),
                "snippet": ""
            }
            if snippet:
                result_data["snippet"] = snippet.get_text(
                    " ",
                    strip=True
                )
            results.append(result_data)
            if len(results) >= max_results:
                break
        return results
    except Exception:
        return []
def knowledge_search(query):
    wikipedia = wikipedia_search(query)
    if wikipedia:
        content = wikipedia.get(
            "content",
            ""
        )
        if len(content) >= 200:
            return {
                "query": query,
                "source_used": "Wikipedia",
                "wikipedia": wikipedia,
                "web_search": []
            }
    web_results = web_search(query)
    return {
        "query": query,
        "source_used": "Web Search",
        "wikipedia": wikipedia,
        "web_search": web_results
    }