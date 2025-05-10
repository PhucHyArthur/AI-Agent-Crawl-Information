from agents.netflix_movie_agent import build_movie_agent

def main():
    agent = build_movie_agent()
    task = "Scrape the latest movies on Netflix (title, description, release year) from multiple URLs."
    result = agent.run(task, max_steps=1)
    print("\n\n==== RESULTS ====")
    for title, description, year in result:
        print(f"Title: {title}\nYear: {year}\nDescription: {description}\n")

if __name__ == "__main__":
    main()