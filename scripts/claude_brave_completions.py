import anthropic
import requests
import pandas as pd
import json
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import List, Dict, Optional, Union
from pydantic import BaseModel
from time import sleep
import argparse
from datetime import datetime
import sys
import logging
from logging.handlers import RotatingFileHandler


class WatchOption(BaseModel):
    watch_option: str
    availability: str
    market_team: str
    confidence_score: float


class Config:
    def __init__(self, env_path: str = None, logs_dir: str = "logs"):
        # Create logs directory
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)

        # Set up logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        if env_path:
            load_dotenv(env_path)
            self.logger.info(f"Loaded environment from: {env_path}")
        else:
            env_path = Path(__file__).parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                self.logger.info("Loaded environment from default .env file")
            else:
                self.logger.warning("No .env file found")

        try:
            self.anthropic_api_key = self._get_required_key("ANTHROPIC_API_KEY")
            self.brave_api_key = self._get_required_key("BRAVE_API_KEY")
            self.logger.info("Successfully loaded API keys")
        except ValueError as e:
            self.logger.error(f"Failed to load API keys: {str(e)}")
            raise

    def _setup_logging(self):
        """Configure logging with both file and console handlers."""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler with rotation
        log_file = os.path.join(self.logs_dir, "completions.log")
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def _get_required_key(self, key_name: str) -> str:
        value = os.getenv(key_name)
        if not value:
            raise ValueError(f"Missing required API key: {key_name}")
        return value


class BraveSearchClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
        self.logger = logging.getLogger(__name__)
        # Initialize cache as dictionary
        self._cache = {}
        # Track cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache performance statistics"""
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total_queries": self.cache_hits + self.cache_misses,
            "cache_size": len(self._cache),
        }

    def clear_cache(self):
        """Clear the search cache"""
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Perform a web search using Brave Search API with caching"""
        # Create cache key from query and max_results
        cache_key = f"{query}::{max_results}"

        # Check cache first
        if cache_key in self._cache:
            self.cache_hits += 1
            self.logger.debug(f"Cache hit for query: {query[:50]}...")
            return self._cache[cache_key]

        self.cache_misses += 1
        self.logger.info(f"Cache miss - performing search query: {query[:50]}...")

        try:
            response = requests.get(
                self.base_url,
                params={"q": query, "count": max_results},
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            sleep(1)  # Rate limiting
            results = response.json().get("web", {}).get("results", [])

            # Cache the results
            self._cache[cache_key] = results

            self.logger.info(f"Retrieved and cached {len(results)} search results")
            return results

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Search request failed: {str(e)}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse search response: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error during search: {str(e)}")
            return []


class WatchOptionsAnalyzer:
    def __init__(self, config: Config):
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.search_client = BraveSearchClient(config.brave_api_key)
        self.logger = logging.getLogger(__name__)
        self.error_count = 0
        self.success_count = 0

    def search_watch_option_for_team(
        self, watch_option: str, league: str, team: str
    ) -> str:
        """Search for information about a watch option for a specific team"""
        self.logger.info(f"Searching for {watch_option} rights for {team} in {league}")
        query = f"{watch_option} {league} {team} rights"
        results = self.search_client.search(query)

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f'<result index="{i}">\n'
                f'<url>{result.get("url", "")}</url>\n'
                f'<title>{result.get("title", "")}</title>\n'
                f'<description>{result.get("description", "")}</description>\n'
                f"</result>"
            )

        return "\n".join(formatted_results)

    def search_watch_option(
        self, watch_option: str, league: str, home_team: str, away_team: str
    ) -> Dict[str, str]:
        """Search for information about a watch option for both teams in a game"""
        self.logger.info(
            f"Searching for {watch_option} in {league} for {home_team} vs {away_team}"
        )

        # Get search results for both teams
        home_results = self.search_watch_option_for_team(
            watch_option, league, home_team
        )
        away_results = self.search_watch_option_for_team(
            watch_option, league, away_team
        )

        return {"home_team_results": home_results, "away_team_results": away_results}

    def analyze_watch_options(self, row: pd.Series) -> List[WatchOption]:
        """Analyze watch options for a single game with team-specific search augmentation"""
        game_id = f"{row['date']}_{row['away_team']}_at_{row['home_team']}"
        self.logger.info(f"Analyzing watch options for game: {game_id}")

        try:
            watch_options_list = [
                opt.strip() for opt in row["watch_options"].split(",")
            ]
            self.logger.info(
                f"Found {len(watch_options_list)} watch options to analyze"
            )

            all_search_results = {}
            for watch_option in watch_options_list:
                search_results = self.search_watch_option(
                    watch_option, row["league"], row["home_team"], row["away_team"]
                )
                all_search_results[watch_option] = search_results

            prompt = self._construct_analysis_prompt(row, all_search_results)

            try:
                response = self.client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )

                response_content = response.content[0].text
                watch_options_data = json.loads(response_content)
                watch_options = [WatchOption(**wo) for wo in watch_options_data]

                self.success_count += 1
                self.logger.info(
                    f"Successfully analyzed {len(watch_options)} watch options for {game_id}"
                )
                return watch_options

            except json.JSONDecodeError as e:
                self.error_count += 1
                self.logger.error(
                    f"Failed to parse Claude response for {game_id}: {str(e)}"
                )
                return []
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Claude API error for {game_id}: {str(e)}")
                return []

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error analyzing watch options for {game_id}: {str(e)}")
            return []

    def _construct_analysis_prompt(self, row: pd.Series, search_results: Dict) -> str:
        """Construct the prompt for Claude analysis with team-specific search results"""
        return f"""Analyze the watch options for this {row['league']} game:
Date: {row['date']}
Teams: {row['home_team']} (home) vs {row['away_team']} (away)
Watch options: {row['watch_options']}

Search results about broadcast rights:
{json.dumps(search_results, indent=2)}

Analysis should consider:
- Each watch option's relationship to both teams
- Regional sports networks (RSNs): NBC Sports [Region], Bally Sports [Region], Team Networks
- National broadcasts: ESPN, TNT, ABC, NHL Network, NBA TV
- For regional broadcasts, identify which team's market is served
- National broadcasts should be identified regardless of team rights

Confidence scoring:
- 0.9-1.0: Strong evidence of rights relationship
- 0.7-0.8: Clear pattern but some ambiguity
- 0.5-0.6: Reasonable assumption based on market patterns
- 0.3-0.4: Uncertain relationship

Return analysis in this JSON format:
[{{"watch_option": "string", "availability": "regional"/"national", "market_team": "team name"/"", "confidence_score": float}}]

Format as valid JSON only, no additional text."""

    def log_cache_stats(self):
        """Log the current cache statistics"""
        stats = self.search_client.get_cache_stats()
        self.logger.info("Search cache statistics:")
        self.logger.info(f"  Total queries: {stats['total_queries']}")
        self.logger.info(f"  Cache hits: {stats['hits']}")
        self.logger.info(f"  Cache misses: {stats['misses']}")
        self.logger.info(f"  Cache size: {stats['cache_size']}")
        if stats["total_queries"] > 0:
            hit_rate = (stats["hits"] / stats["total_queries"]) * 100
            self.logger.info(f"  Cache hit rate: {hit_rate:.1f}%")

    def create_summary_report(self, output_dir: str) -> str:
        """Create a summary report of the analysis operation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"analysis_summary_{timestamp}.txt")

        total_processed = self.success_count + self.error_count
        success_rate = (
            (self.success_count / total_processed * 100) if total_processed > 0 else 0
        )

        cache_stats = self.search_client.get_cache_stats()
        hit_rate = 0
        if cache_stats["total_queries"] > 0:
            hit_rate = (cache_stats["hits"] / cache_stats["total_queries"]) * 100

        with open(report_path, "w") as f:
            f.write("Watch Options Analysis Summary Report\n")
            f.write("==================================\n\n")
            f.write(f"Total games processed: {total_processed}\n")
            f.write(f"Successful analyses: {self.success_count}\n")
            f.write(f"Failed analyses: {self.error_count}\n")
            f.write(f"Success rate: {success_rate:.2f}%\n\n")
            f.write("Search Cache Statistics:\n")
            f.write(f"Total queries: {cache_stats['total_queries']}\n")
            f.write(f"Cache hits: {cache_stats['hits']}\n")
            f.write(f"Cache misses: {cache_stats['misses']}\n")
            f.write(f"Cache size: {cache_stats['cache_size']}\n")
            f.write(f"Cache hit rate: {hit_rate:.1f}%\n")

        return report_path


def get_output_filepath(input_filepath: str, output_dir: str) -> str:
    """Generate output filepath with timestamp"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_name = Path(input_filepath).stem
    output_filename = f"{input_name}_analysis_{timestamp}.json"

    return str(output_path / output_filename)


def process_games_file(
    input_filepath: str, output_dir: str, env_file: Optional[str] = None
) -> Dict[str, List[WatchOption]]:
    """Process entire games file and analyze watch options"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting processing of games file: {input_filepath}")

    try:
        # Load configuration
        config = Config(env_file, os.path.join(output_dir, "logs"))
        analyzer = WatchOptionsAnalyzer(config)

        # Read CSV file
        logger.info("Reading input CSV file")
        df = pd.read_csv(input_filepath)
        logger.info(f"Found {len(df)} games to process")

        # Store results by game
        results = {}
        total_games = len(df)

        # Process each row
        for index, row in df.iterrows():
            game_key = f"{row['date']}_{row['away_team']}_at_{row['home_team']}"
            logger.info(f"Processing game {index + 1}/{total_games}: {game_key}")

            watch_options = analyzer.analyze_watch_options(row)
            results[game_key] = watch_options

            # Log cache stats every 50 games
            if (index + 1) % 50 == 0:
                logger.info(f"Progress update after {index + 1} games:")
                analyzer.log_cache_stats()

        # Log final cache statistics
        logger.info("Final cache statistics:")
        analyzer.log_cache_stats()

        # Create summary report
        summary_path = analyzer.create_summary_report(output_dir)
        logger.info(f"Created summary report at: {summary_path}")

        return results

    except pd.errors.EmptyDataError:
        logger.error(f"Input file {input_filepath} is empty")
        return {}
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error processing games file: {str(e)}")
        return {}


def save_results(results: Dict[str, List[WatchOption]], output_file: str):
    """Save results to a JSON file"""
    logger = logging.getLogger(__name__)
    logger.info(f"Saving results to {output_file}")

    try:
        output_data = {
            game_key: [wo.model_dump() for wo in watch_options]
            for game_key, watch_options in results.items()
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info("Results successfully saved")

    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze watch options for sports games from CSV file"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input CSV file path containing games data"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output",
        help="Output directory for results (default: output)",
    )
    parser.add_argument("--env-file", help="Path to .env file (optional)")

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    try:
        # Process games
        results = process_games_file(args.input, args.output_dir, args.env_file)

        if not results:
            logging.error("No results generated from processing")
            sys.exit(1)

        # Generate output filepath
        output_file = get_output_filepath(args.input, args.output_dir)

        # Save results
        save_results(results, output_file)

        logging.info("Processing completed successfully")

    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
