# ESPN Game Scraper

A set of Python scripts for scraping game schedules and watch options from ESPN's website. The scraper supports multiple sports leagues and can handle both single-day and bulk date range scraping operations.

## Features

- Scrape game schedules and watch options for multiple sports leagues
- Support for single-day and date range scraping
- Automatic retry mechanisms and rate limiting
- Comprehensive logging and error handling
- CSV output with detailed game information
- Summary reports for bulk scraping operations

## Supported Leagues

- MLB (Major League Baseball)
- NFL (National Football League)
- NBA (National Basketball Association)
- NHL (National Hockey League)
- College Football
- Men's College Basketball
- Women's College Basketball

## Requirements

```
requests
beautifulsoup4
pandas
```

## Installation

1. Clone the repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Single Day Scraping

Use `espn_game_scraper.py` to scrape games for a specific league and date:

```bash
python espn_game_scraper.py --league mlb --date 20250409
```

Arguments:
- `--league`: The sports league to scrape (required)
- `--date`: The date to scrape in YYYYMMDD format (required)

### Bulk Scraping

Use `espn_bulk_scraper.py` to scrape multiple leagues across a date range:

```bash
python espn_bulk_scraper.py --leagues mlb nba --start-date 20250409 --end-date 20250415 --delay 2.0
```

Arguments:
- `--leagues`: List of leagues to scrape (required)
- `--start-date`: Start date in YYYYMMDD format (required)
- `--end-date`: End date in YYYYMMDD format (required)
- `--delay`: Delay between requests in seconds (default: 2.0)
- `--output-dir`: Output directory for scraped data (default: "data")
- `--logs-dir`: Directory for log files (default: "logs")

## Output

The scrapers generate CSV files with the following columns:
- `date`: Date of the games (YYYYMMDD format)
- `league`: League of the games
- `away_team`: Name of the away team
- `home_team`: Name of the home team
- `watch_options`: Comma-separated list of watch options

### File Structure

```
project/
├── data/                       # Scraped data output directory
│   └── espn_games_*.csv        # CSV files with scraped data
├── logs/                       # Log files directory
│   ├── scraper.log             # Detailed logging information
│   └── scraping_summary_*.txt  # Summary reports
├── espn_game_scraper.py        # Single day scraping script
└── espn_bulk_scraper.py        # Bulk scraping script
```

## Rate Limiting and Best Practices

- The bulk scraper includes a configurable delay between requests to respect rate limits
- User-Agent headers are randomized to simulate browser-like requests
- Error handling includes logging of failed requests and retry mechanisms
- Summary reports track success rates and provide data overviews

## Error Handling

The scripts include comprehensive error handling for:
- Invalid input parameters
- Network request failures
- HTML parsing errors
- File system operations

All errors are logged to both console and log files for debugging.
