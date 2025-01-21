# Broadcast Rights Mapping

A comprehensive collection of Python scripts for scraping game schedules, watch options, and local market rights information from various sports websites and APIs. The toolkit includes support for multiple sports leagues and can handle both game schedule scraping from ESPN and local market rights determination from league-specific APIs.

## Features

### Game Schedule Scraping
- Scrape game schedules and watch options from ESPN for multiple sports leagues
- Support for single-day and date range scraping
- Automatic retry mechanisms and rate limiting
- Comprehensive logging and error handling
- CSV output with detailed game information
- Summary reports for bulk scraping operations

### Local Market Rights Lookup
- Determine local market broadcast rights for MLB, NBA, and NHL by ZIP code
- Unified interface for querying all leagues simultaneously
- Asynchronous processing with configurable batch sizes and rate limiting
- Automatic retry mechanisms with exponential backoff
- Detailed error tracking and reporting
- Support for bulk ZIP code processing

## Supported Leagues

### ESPN Game Schedules
- MLB (Major League Baseball)
- NFL (National Football League)
- NBA (National Basketball Association)
- NHL (National Hockey League)
- College Football
- Men's College Basketball
- Women's College Basketball

### Local Market Rights
- MLB
- NBA
- NHL

## Requirements & Installation

This project requires Python 3.12 or higher and uses the following dependencies:
- aiohttp >= 3.11.11
- argparse >= 1.4.0
- beautifulsoup4 (bs4) >= 0.0.2
- logging >= 0.4.9.6
- pandas >= 2.2.3
- requests >= 2.32.3
- tqdm >= 4.67.1

To install:

1. Clone the repository
2. Install the project and its dependencies:
```bash
pip install .
```

## Usage

### ESPN Game Schedule Scraping

#### Single Day Scraping

Use `espn_game_scraper.py` to scrape games for a specific league and date:

```bash
python espn_game_scraper.py --league mlb --date 20250409
```

Arguments:
- `--league`: The sports league to scrape (required)
- `--date`: The date to scrape in YYYYMMDD format (required)

#### Bulk Schedule Scraping

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

### Local Market Rights Lookup

#### Individual League Lookup

Separate scripts are available for each league:
- `mlb_rights_scraper.py`
- `nba_rights_scraper.py`
- `nhl_rights_scraper.py`

Example usage:
```bash
python mlb_rights_scraper.py --input-file zip_dma_mapping.csv --batch-size 10 --rate-limit 50
```

#### Unified League Lookup

Use `all_local_rights_scraper.py` to query all supported leagues simultaneously:

```bash
python all_local_rights_scraper.py --input-file zip_dma_mapping.csv --config-file config.json
```

Arguments:
- `--input-file`: CSV file containing ZIP codes (default: zip_dma_mapping.csv)
- `--config-file`: JSON configuration file (optional)
- `--batch-size`: Number of concurrent requests per league
- `--rate-limit`: Maximum requests per second per league
- `--output-dir`: Output directory for results
- `--logs-dir`: Directory for log files

## Output Format

### ESPN Game Schedule Output
CSV files with the following columns:
- `date`: Date of the games (YYYYMMDD format)
- `league`: League of the games
- `away_team`: Name of the away team
- `home_team`: Name of the home team
- `watch_options`: Comma-separated list of watch options

### Local Market Rights Output
CSV files with the following columns:
- `zip_code`: 5-digit ZIP code
- `in_market_teams`: Comma-separated list of teams with local broadcast rights
- `error_code`: Error code if lookup failed (NULL if successful)
- `error_message`: Detailed error message if lookup failed (NULL if successful)

## Project Structure

```
project/
├── data/                           # Scraped data output directory
│   ├── espn_games_*.csv           # ESPN game schedule data
│   ├── mlb_market_teams_*.csv     # MLB market rights data
│   ├── nba_market_teams_*.csv     # NBA market rights data
│   ├── nhl_market_teams_*.csv     # NHL market rights data
│   └── combined_results_*.csv     # Combined league rights data
├── logs/                          # Log files directory
│   ├── scraper.log               # ESPN scraper logs
│   ├── mlb_lookup.log           # MLB rights lookup logs
│   ├── nba_lookup.log           # NBA rights lookup logs
│   ├── nhl_lookup.log           # NHL rights lookup logs
│   ├── unified_scraper.log      # Combined rights lookup logs
│   └── *_summary_*.txt          # Various summary reports
├── espn_game_scraper.py          # Single day ESPN scraping
├── espn_bulk_scraper.py          # Bulk ESPN scraping
├── mlb_rights_scraper.py         # MLB rights lookup
├── nba_rights_scraper.py         # NBA rights lookup
├── nhl_rights_scraper.py         # NHL rights lookup
└── all_local_rights_scraper.py   # Unified rights lookup
```

## Rate Limiting and Best Practices

- Configurable delays between requests to respect rate limits
- Randomized User-Agent headers to simulate browser-like requests
- Exponential backoff retry mechanism for failed requests
- Concurrent processing with configurable batch sizes
- Comprehensive error handling and logging

## Error Handling

The scripts include robust error handling for:
- Invalid input parameters
- Network request failures
- Rate limiting responses
- API parsing errors
- File system operations
- Invalid ZIP codes
- International postal codes

All errors are logged to both console and log files for debugging, with detailed summary reports generated for each operation.

## Configuration

The unified rights lookup script supports JSON configuration files for setting:
- Batch sizes
- Rate limits
- Retry attempts
- Request timeouts
- Output directories
- Log directories

Example config.json:
```json
{
  "batch_size": 10,
  "rate_limit": 50,
  "max_retries": 3,
  "timeout": 30,
  "output_dir": "data",
  "logs_dir": "logs"
}
```
