import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse, parse_qs
import logging
import os
import time
import argparse
import random

# Valid league options
VALID_LEAGUES = {
    "mlb",
    "nfl",
    "nba",
    "nhl",
    "college-football",
    "mens-college-basketball",
    "womens-college-basketball",
}


def validate_parameters(league, game_date):
    """
    Validate the input parameters.

    Args:
        league (str): The sports league to scrape
        game_date (str): The date to scrape in YYYYMMDD format

    Returns:
        tuple: (bool, str) indicating if validation passed and any error message
    """
    # Validate league
    if league not in VALID_LEAGUES:
        return (
            False,
            f"Invalid league. Must be one of: {', '.join(sorted(VALID_LEAGUES))}",
        )

    # Validate date format
    try:
        date_int = int(game_date)
        if len(game_date) != 8:
            return False, "Date must be exactly 8 digits in YYYYMMDD format"
    except ValueError:
        return False, "Date must be a valid integer in YYYYMMDD format"

    return True, ""


def extract_date_from_url(url):
    """Extract the date parameter from the URL."""
    try:
        parsed_url = urlparse(url)
        path_segments = parsed_url.path.split("/")
        # Find the dates segment and get the following value
        for i, segment in enumerate(path_segments):
            if segment == "dates":
                return path_segments[i + 1]
    except Exception as e:
        logging.error(f"Error extracting date from URL: {e}")
        return None


def scrape_espn_games(league, game_date):
    """
    Scrape game information from ESPN's where-to-watch page.

    Args:
        league (str): The sports league to scrape
        game_date (str): The date to scrape in YYYYMMDD format

    Returns:
        tuple: (DataFrame, str) where:
            - DataFrame containing game information with columns:
                - date: Date of the games (YYYYMMDD format)
                - league: League of the games
                - away_team: Name of the away team
                - home_team: Name of the home team
                - watch_options: Comma-separated list of watch options
            - str: The extracted date from the URL
    """
    try:
        # Validate parameters
        is_valid, error_message = validate_parameters(league, game_date)
        if not is_valid:
            raise ValueError(error_message)

        # Construct the URL
        url = (
            f"https://www.espn.com/where-to-watch/leagues/{league}/_/dates/{game_date}"
        )

        # Set up headers to mimic a browser request
        headers = get_headers()

        # Make the request to the URL
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the date from the URL
        date = extract_date_from_url(url)

        # Find all game divs
        game_divs = soup.find_all("div", class_="VZTD HNQqj FhNIO")

        # Lists to store the data
        games_data = []

        # Process each game
        for game_div in game_divs:
            # Find all team divs within the game
            team_divs = game_div.find_all("div", class_="ZLXw BJfX")

            if len(team_divs) >= 2:  # Ensure we have both teams
                # Extract team names from img alt attributes
                away_team_img = team_divs[0].find("img", class_="hsDdd qqXIK dzLvX")
                home_team_img = team_divs[1].find("img", class_="hsDdd qqXIK dzLvX")

                if away_team_img and home_team_img:
                    away_team = away_team_img.get("alt", "")
                    home_team = home_team_img.get("alt", "")

                    # Extract watch options for this game
                    watch_spans = game_div.find_all(
                        "span",
                        class_="VZTD mLASH kfeMl klTtn JspAS GpQCA kfPAQ FuEs SIxtI GNmIv tuAKv cmtRa vHUJ xTell zkpVE jIRH ImFQ sJGIv xVNgQ bhvGw",
                    )
                    watch_options = [
                        span.text.strip() for span in watch_spans if span.text.strip()
                    ]
                    watch_options_str = (
                        ",".join(watch_options) if watch_options else None
                    )

                    games_data.append(
                        {
                            "date": date,
                            "league": league,
                            "away_team": away_team,
                            "home_team": home_team,
                            "watch_options": watch_options_str,
                        }
                    )

        # Create DataFrame
        if not games_data:
            logging.info("No games were found for the specified date and league")
            # Create empty DataFrame with correct columns
            df = pd.DataFrame(
                columns=["date", "league", "away_team", "home_team", "watch_options"]
            )
        else:
            df = pd.DataFrame(games_data)
            # Ensure proper column order
            df = df[["date", "league", "away_team", "home_team", "watch_options"]]

        return df, date

    except requests.exceptions.RequestException as e:
        logging.error(f"Error making request to URL: {e}")
        raise
    except Exception as e:
        logging.error(f"Error scraping data: {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape ESPN game schedules and watch options."
    )

    parser.add_argument(
        "--league",
        type=str,
        required=True,
        choices=sorted(VALID_LEAGUES),
        help="The sports league to scrape (e.g., mlb, nba, nfl)",
    )

    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="The date to scrape in YYYYMMDD format (e.g., 20250409)",
    )

    return parser.parse_args()


def get_headers():
    """Get randomized browser-like headers."""
    user_agents = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        # Safari on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ]

    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    return headers


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Parse command line arguments
    args = parse_arguments()

    try:
        # Scrape the data
        df, scraped_date = scrape_espn_games(args.league, args.date)

        # Display the results
        print("\nScraped Game Schedule:")
        print(df.to_string(index=False))

        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)

        # Generate filename with league, date and timestamp
        current_ts = int(time.time())
        filename = f"espn_games_{args.league}_{scraped_date}_{current_ts}.csv"
        filepath = os.path.join(data_dir, filename)

        # Save to CSV
        if len(df) > 0:
            df.to_csv(filepath, index=False)
            print(f"\nData has been saved to '{filepath}'")
        else:
            print("\nNo data was saved as no games were found")

    except Exception as e:
        logging.error(f"Script execution failed: {e}")


if __name__ == "__main__":
    main()
