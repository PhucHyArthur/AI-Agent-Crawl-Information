from smolagents import tool

@tool
def fetch_netflix_movie_info(urls: list[str]) -> list[tuple[str, str, str]]:
    """
    Fetch latest Netflix movie titles, descriptions, and release years from a list of URLs.

    Args:
        urls (list[str]): List of Netflix movie page URLs.

    Returns:
        list[tuple[str, str, str]]: List of (title, description, release year).
    """
    import requests
    from bs4 import BeautifulSoup

    results = []

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        for item in soup.find_all("div", class_="title-card-container"):
            title_tag = item.find("span", class_="fallback-text")
            description_tag = item.find_next("p")
            year_tag = item.find_next("span", class_="year")

            title = title_tag.get_text(strip=True) if title_tag else "N/A"
            description = description_tag.get_text(strip=True) if description_tag else "N/A"
            year = year_tag.get_text(strip=True) if year_tag else "N/A"

            results.append((title, description, year))

    return results