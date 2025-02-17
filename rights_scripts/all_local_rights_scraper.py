from abc import ABC, abstractmethod
import asyncio
import aiohttp
import pandas as pd
import logging
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, AsyncGenerator, Any
from tqdm.asyncio import tqdm_asyncio
import argparse
from dataclasses import dataclass
from enum import Enum, auto
from collections import deque
import json
from pathlib import Path

# python all_local_rights_scraper.py --input-file zip_dma_mapping_testing.csv


@dataclass
class ScraperConfig:
    """Configuration class for scraper settings."""

    batch_size: int = 10
    rate_limit: int = 50
    max_retries: int = 5
    timeout: int = 30
    output_dir: str = "data"
    logs_dir: str = "logs"

    @classmethod
    def from_file(cls, filepath: str) -> "ScraperConfig":
        """Load configuration from a JSON file."""
        with open(filepath) as f:
            return cls(**json.load(f))

    def to_file(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=2)


class ErrorCode(Enum):
    """Enumeration of possible error codes."""

    SUCCESS = auto()
    RATE_LIMITED = auto()  # 429 Too Many Requests
    SERVER_ERROR = auto()  # 5xx errors
    CLIENT_ERROR = auto()  # 4xx errors (except 429)
    TIMEOUT = auto()  # Request timeout
    PARSE_ERROR = auto()  # JSON parsing error
    NETWORK_ERROR = auto()  # Connection issues
    INVALID_DATA = auto()  # Valid response but unexpected data structure
    INVALID_ZIP = auto()  # Invalid ZIP code format
    INTERNATIONAL = auto()  # Non-US postal code
    UNKNOWN = auto()  # Unhandled errors


@dataclass
class RequestResult:
    """Container for request results and metadata."""

    success: bool
    error_code: ErrorCode
    response_time: float
    retry_count: int
    data: Optional[str] = None
    error_message: Optional[str] = None


class RateLimiter:
    """Adaptive rate limiter with error-based adjustments."""

    def __init__(self, initial_rate: int = 50, window_size: int = 50):
        self.current_rate = initial_rate
        self.window_size = window_size
        self.error_window = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.min_rate = 10
        self.max_rate = 100

    def record_result(self, success: bool, response_time: float) -> None:
        """Record the result of a request."""
        self.error_window.append(not success)
        self.response_times.append(response_time)
        self._adjust_rate()

    def _adjust_rate(self) -> None:
        """Adjust the rate based on error rate and response times."""
        if len(self.error_window) >= self.window_size // 2:
            error_rate = sum(self.error_window) / len(self.error_window)
            avg_response_time = sum(self.response_times) / len(self.response_times)

            # Decrease rate if error rate is high
            if error_rate > 0.1:  # More than 10% errors
                self.current_rate = max(self.min_rate, int(self.current_rate * 0.8))
            # Increase rate if error rate is low and response times are good
            elif error_rate < 0.05 and avg_response_time < 1.0:
                self.current_rate = min(self.max_rate, int(self.current_rate * 1.2))

    @property
    def delay(self) -> float:
        """Get the current delay between requests."""
        return 1.0 / self.current_rate


class RetryStrategy:
    """Implements exponential backoff with jitter."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def execute(self, func: Any, *args: Any, **kwargs: Any) -> RequestResult:
        """Execute a function with retry logic."""
        retry_count = 0
        while retry_count <= self.max_retries:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time

                if isinstance(result, tuple):
                    success, data = result
                    return RequestResult(
                        success=success,
                        error_code=(
                            ErrorCode.SUCCESS if success else ErrorCode.INVALID_DATA
                        ),
                        response_time=elapsed,
                        retry_count=retry_count,
                        data=data,
                    )

            except aiohttp.ClientResponseError as e:
                elapsed = time.time() - start_time
                if e.status == 429:
                    error_code = ErrorCode.RATE_LIMITED
                elif 500 <= e.status < 600:
                    error_code = ErrorCode.SERVER_ERROR
                else:
                    error_code = ErrorCode.CLIENT_ERROR

                if (
                    retry_count == self.max_retries
                    or error_code == ErrorCode.CLIENT_ERROR
                ):
                    return RequestResult(
                        success=False,
                        error_code=error_code,
                        response_time=elapsed,
                        retry_count=retry_count,
                        error_message=str(e),
                    )

            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                if retry_count == self.max_retries:
                    return RequestResult(
                        success=False,
                        error_code=ErrorCode.TIMEOUT,
                        response_time=elapsed,
                        retry_count=retry_count,
                        error_message="Request timed out",
                    )

            except aiohttp.ClientError as e:
                elapsed = time.time() - start_time
                return RequestResult(
                    success=False,
                    error_code=ErrorCode.NETWORK_ERROR,
                    response_time=elapsed,
                    retry_count=retry_count,
                    error_message=str(e),
                )

            except json.JSONDecodeError as e:
                elapsed = time.time() - start_time
                return RequestResult(
                    success=False,
                    error_code=ErrorCode.PARSE_ERROR,
                    response_time=elapsed,
                    retry_count=retry_count,
                    error_message=str(e),
                )

            except Exception as e:
                elapsed = time.time() - start_time
                return RequestResult(
                    success=False,
                    error_code=ErrorCode.UNKNOWN,
                    response_time=elapsed,
                    retry_count=retry_count,
                    error_message=str(e),
                )

            # Calculate backoff with jitter
            delay = self.base_delay * (2**retry_count) * (0.5 + random.random())
            await asyncio.sleep(delay)
            retry_count += 1


@dataclass
class LeagueConfig:
    """Configuration class for league-specific settings."""

    name: str
    base_url: str


class SportsMarketLookup(ABC):
    """Abstract base class for sports market lookups."""

    def __init__(
        self,
        league_config: LeagueConfig,
        config: ScraperConfig,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self.league_config = league_config
        self.config = config
        self.session = session
        self.rate_limiter = RateLimiter(initial_rate=config.rate_limit)
        self.retry_strategy = RetryStrategy(max_retries=config.max_retries)
        self.error_counts = {code: 0 for code in ErrorCode}
        self.errors = []

        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{league_config.name}")

    def _validate_zip(self, zip_code: str) -> Tuple[bool, ErrorCode, str]:
        """Internal validation method."""
        if not zip_code.isdigit():
            return False, ErrorCode.INVALID_ZIP, "ZIP code must contain only digits"
        if len(zip_code) != 5:
            return False, ErrorCode.INVALID_ZIP, "ZIP code must be exactly 5 digits"
        if not zip_code.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
            return False, ErrorCode.INTERNATIONAL, "Non-US ZIP code detected"
        return True, ErrorCode.SUCCESS, ""

    def get_headers(self) -> Dict[str, str]:
        """Get randomized browser-like headers."""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        ]

        return {
            "User-Agent": random.choice(user_agents),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
        }

    def save_error_report(self) -> str:
        """Save error report to file and return the filepath."""
        if not self.errors:
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_report_{self.league_config.name}_{timestamp}.txt"
        filepath = os.path.join(self.config.logs_dir, filename)

        with open(filepath, "w") as f:
            f.write(f"Error Report for {self.league_config.name}\n")
            f.write("=" * (len(self.league_config.name) + 16) + "\n\n")
            for error in self.errors:
                f.write(f"ZIP: {error['zip_code']}\n")
                f.write(f"Error Code: {error['error_code']}\n")
                f.write(f"Message: {error['error_message']}\n")
                f.write("-" * 50 + "\n")

        return filepath

    def _deduplicate_teams(self, teams: List[str]) -> List[str]:
        """Deduplicate and sort team names."""
        # Convert to set to remove duplicates, then back to sorted list
        return sorted(set(teams))

    def _join_teams(self, teams: List[str]) -> str:
        """Join team names into a comma-separated string after deduplication."""
        deduped_teams = self._deduplicate_teams(teams)
        return ",".join(deduped_teams) if deduped_teams else ""

    @abstractmethod
    def get_request_url(self, zip_code: str) -> str:
        """Generate the request URL for a given ZIP code."""
        pass

    @abstractmethod
    async def parse_response(self, response_data: dict) -> Tuple[bool, str]:
        """Parse the API response and return success status and team names."""
        pass

    async def process_zip_codes(self, zip_codes: List[str]) -> pd.DataFrame:
        """Process multiple ZIP codes and return results as a DataFrame."""
        results = []
        processed = 0
        total = len(zip_codes)
        batches = [
            zip_codes[i : i + self.config.batch_size]
            for i in range(0, len(zip_codes), self.config.batch_size)
        ]

        for batch in batches:
            batch_results = []
            for zip_code in batch:
                # Validate ZIP code first
                is_valid, error_code, error_message = self._validate_zip(zip_code)
                if not is_valid:
                    error_data = {
                        "zip_code": zip_code,
                        "error_code": error_code.name,
                        "error_message": error_message,
                    }
                    self.errors.append(error_data)
                    batch_results.append(
                        {
                            "zip_code": zip_code,
                            "in_market_teams": None,
                            "error_code": error_code.name,
                            "error_message": error_message,
                        }
                    )
                    continue

                try:
                    url = self.get_request_url(zip_code)
                    async with self.session.get(
                        url, headers=self.get_headers()
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        success, teams = await self.parse_response(data)

                        result = {
                            "zip_code": zip_code,
                            "in_market_teams": teams if success else None,
                            "error_code": (
                                None if success else ErrorCode.INVALID_DATA.name
                            ),
                            "error_message": (
                                None if success else "Failed to parse response"
                            ),
                        }

                        if not success:
                            self.errors.append(
                                {
                                    "zip_code": zip_code,
                                    "error_code": ErrorCode.INVALID_DATA.name,
                                    "error_message": "Failed to parse response",
                                }
                            )

                        batch_results.append(result)

                except Exception as e:
                    error_data = {
                        "zip_code": zip_code,
                        "error_code": ErrorCode.UNKNOWN.name,
                        "error_message": str(e),
                    }
                    self.errors.append(error_data)
                    batch_results.append(
                        {
                            "zip_code": zip_code,
                            "in_market_teams": None,
                            "error_code": ErrorCode.UNKNOWN.name,
                            "error_message": str(e),
                        }
                    )

            processed += len(batch)
            if processed % 500 == 0:  # Status update every 100 ZIPs
                self.logger.info(
                    f"{self.league_config.name}: Processed {processed}/{total} ZIP codes"
                )

            await asyncio.sleep(self.rate_limiter.delay)
            results.extend(batch_results)

        if not results:
            self.logger.warning("No results were obtained")
            return pd.DataFrame(
                columns=["zip_code", "in_market_teams", "error_code", "error_message"]
            )

        return pd.DataFrame(results)


class MLBMarketLookup(SportsMarketLookup):
    """MLB-specific implementation of market lookup."""

    def get_request_url(self, zip_code: str) -> str:
        return f"{self.league_config.base_url}/{zip_code}.json"

    async def parse_response(self, response_data: dict) -> Tuple[bool, str]:
        if not response_data or "teams" not in response_data:
            return False, ""
        try:
            teams = response_data["teams"]
            return True, self._join_teams(teams)
        except (KeyError, TypeError):
            return False, ""


class NBAMarketLookup(SportsMarketLookup):
    """NBA-specific implementation of market lookup."""

    def get_request_url(self, zip_code: str) -> str:
        return f"{self.league_config.base_url}?zip={zip_code}"

    async def parse_response(self, response_data: dict) -> Tuple[bool, str]:
        if not response_data or "results" not in response_data:
            return False, ""
        # An empty list is considered a valid response
        if not response_data["results"]:
            return True, ""
        try:
            teams = [team["abbr"] for team in response_data["results"]]
            return True, self._join_teams(teams)
        except (KeyError, TypeError):
            return False, ""


class NHLMarketLookup(SportsMarketLookup):
    """NHL-specific implementation of market lookup."""

    def get_request_url(self, zip_code: str) -> str:
        return f"{self.league_config.base_url}/{zip_code}"

    async def parse_response(self, response_data: dict) -> Tuple[bool, str]:
        if not response_data:
            return False, ""
        try:
            teams = [team["teamName"]["default"] for team in response_data]
            return True, self._join_teams(teams)
        except (KeyError, TypeError):
            return False, ""


class UnifiedMarketLookup:
    """Unified processor for all leagues with enhanced error handling and resource management."""

    def __init__(self, config: ScraperConfig):
        self.config = config
        self.session = None
        self.connector = None

        # Create directory structure
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.logs_dir, exist_ok=True)

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler
        fh = logging.FileHandler(os.path.join(config.logs_dir, "unified_scraper.log"))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    async def __aenter__(self):
        """Set up resources when entering context."""
        self.connector = aiohttp.TCPConnector(
            limit=self.config.batch_size * 3
        )  # 3x batch size for 3 leagues
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
        )

        self.processors = {
            "MLB": MLBMarketLookup(
                LeagueConfig("MLB", "https://content.mlb.com/data/blackouts"),
                self.config,
                self.session,
            ),
            "NBA": NBAMarketLookup(
                LeagueConfig(
                    "NBA",
                    "https://content-api-prod.nba.com/public/1/leagues/nba/blackouts",
                ),
                self.config,
                self.session,
            ),
            "NHL": NHLMarketLookup(
                LeagueConfig("NHL", "https://api-web.nhle.com/v1/postal-lookup"),
                self.config,
                self.session,
            ),
        }
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()

    def export_combined_results(
        self, results: Dict[str, pd.DataFrame], filepath: str
    ) -> None:
        """Export combined results from all leagues to a single CSV file."""
        try:
            combined = pd.concat(
                [df.assign(league=league) for league, df in results.items()],
                ignore_index=True,
            )
            combined.to_csv(filepath, index=False)
            self.logger.info(f"Combined results exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export combined results: {str(e)}")

    def create_combined_report(
        self, results: Dict[str, Tuple[pd.DataFrame, str]]
    ) -> None:
        """Create a combined summary report for all leagues."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.config.logs_dir, f"combined_summary_{timestamp}.txt"
        )

        with open(report_path, "w") as f:
            f.write("Combined Sports Market Lookup Summary Report\n")
            f.write("=========================================\n\n")

            total_processed = 0
            total_successful = 0

            for league, (df, error_report) in results.items():
                f.write(f"\n{league} Summary:\n")
                f.write("-" * (len(league) + 9) + "\n")
                f.write(f"Total records: {len(df)}\n")

                # Safely check for successful lookups
                successful = 0
                if not df.empty:
                    successful = len(
                        df[df["error_code"].isna() | df["error_code"].isnull()]
                    )

                f.write(f"Successful lookups: {successful}\n")
                f.write(f"Failed lookups: {len(df) - successful}\n")
                if len(df) > 0:
                    f.write(f"Success rate: {(successful/len(df))*100:.2f}%\n")

                if error_report:
                    f.write(f"Detailed error report: {error_report}\n")

                total_processed += len(df)
                total_successful += successful

            f.write("\nOverall Summary:\n")
            f.write("---------------\n")
            f.write(f"Total records processed: {total_processed}\n")
            f.write(f"Total successful lookups: {total_successful}\n")
            if total_processed > 0:
                f.write(
                    f"Overall success rate: {(total_successful/total_processed)*100:.2f}%\n"
                )

    async def process_all_leagues(
        self, zip_codes: List[str]
    ) -> Dict[str, Tuple[pd.DataFrame, str]]:
        """Process ZIP codes for all leagues with proper resource management."""
        results = {}

        # Define the expected columns for our DataFrame
        empty_df = pd.DataFrame(
            columns=["zip_code", "in_market_teams", "error_code", "error_message"]
        )

        try:
            for league_name, processor in self.processors.items():
                self.logger.info(f"Processing {league_name}...")
                try:
                    df = await processor.process_zip_codes(zip_codes)
                    # Ensure DataFrame has all required columns
                    for col in empty_df.columns:
                        if col not in df.columns:
                            df[col] = None

                    error_report = None
                    if hasattr(processor, "save_error_report"):
                        error_report = processor.save_error_report()

                    results[league_name] = (df, error_report)
                    self.logger.info(f"Completed {league_name} processing")
                except Exception as e:
                    self.logger.error(f"Failed to process {league_name}: {str(e)}")
                    # Create DataFrame with same structure but error information
                    error_df = empty_df.copy()
                    # Add error records for each ZIP code
                    error_records = []
                    for zip_code in zip_codes:
                        error_records.append(
                            {
                                "zip_code": zip_code,
                                "in_market_teams": None,
                                "error_code": "PROCESSING_ERROR",
                                "error_message": str(e),
                            }
                        )
                    error_df = pd.DataFrame(error_records)
                    results[league_name] = (error_df, None)

            # Generate combined report and export
            self.create_combined_report(results)

            # Export combined results
            combined_file = os.path.join(
                self.config.output_dir,
                f"combined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )
            self.export_combined_results(
                {league: df for league, (df, _) in results.items()}, combined_file
            )

        except Exception as e:
            self.logger.error(f"Error in process_all_leagues: {str(e)}")
            raise

        return results


async def validate_input_file(file_path: str) -> List[str]:
    """Validate input CSV file and return list of ZIP codes."""
    try:
        df = pd.read_csv(file_path)
        if "zip_code" not in df.columns:
            raise ValueError("Input file must contain a 'zip_code' column")

        zip_codes = [str(int(zip_code)).zfill(5) for zip_code in df["zip_code"]]
        return zip_codes
    except Exception as e:
        raise ValueError(f"Error reading input file: {str(e)}")


async def main_async():
    """Main async entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Look up sports market teams for ZIP codes"
    )
    parser.add_argument(
        "--input-file",
        default="mapping_files/zip_dma_mapping_testing.csv",
        help="Input CSV file containing ZIP codes",
    )
    parser.add_argument("--config-file", help="Configuration file (JSON)", default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of concurrent requests per league (overrides config file)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        help="Maximum requests per second per league (overrides config file)",
    )
    parser.add_argument(
        "--output-dir", help="Output directory for results (overrides config file)"
    )
    parser.add_argument(
        "--logs-dir", help="Directory for log files (overrides config file)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        if args.config_file and os.path.exists(args.config_file):
            config = ScraperConfig.from_file(args.config_file)
        else:
            config = ScraperConfig()

        # Override config with command line arguments
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.rate_limit:
            config.rate_limit = args.rate_limit
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.logs_dir:
            config.logs_dir = args.logs_dir

        # Validate and read ZIP codes
        zip_codes = await validate_input_file(args.input_file)

        # Process all leagues using context manager for resource cleanup
        async with UnifiedMarketLookup(config) as processor:
            results = await processor.process_all_leagues(zip_codes)

            # Results are automatically saved and reports generated in process_all_leagues

    except Exception as e:
        logging.error(f"Script execution failed: {str(e)}")
        raise


def main():
    """Entry point for the script."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
