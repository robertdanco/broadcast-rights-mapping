import argparse
from datetime import datetime, timedelta
import logging
import os
import time
from typing import List, Tuple
import pandas as pd
from espn_game_scraper import scrape_espn_games, VALID_LEAGUES


class ESPNScraper:
    def __init__(self, output_dir: str = "data", logs_dir: str = "logs"):
        self.output_dir = output_dir
        self.logs_dir = logs_dir

        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(logs_dir, "scraper.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """Generate a list of dates in YYYYMMDD format between start and end dates."""
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        date_list = []
        current = start
        while current <= end:
            date_list.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)

        return date_list

    def scrape_data(self, league: str, date: str) -> Tuple[bool, pd.DataFrame]:
        """Scrape data for a specific league and date."""
        try:
            # Use existing scrape_espn_games function
            df, scraped_date = scrape_espn_games(league, date)

            if len(df) > 0:
                self.logger.info(
                    f"Successfully scraped {len(df)} games for {league} on {date}"
                )
                return True, df
            else:
                self.logger.info(f"No games found for {league} on {date}")
                return True, pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error scraping {league} for {date}: {str(e)}")
            return False, pd.DataFrame()

    def scrape_range(
        self, leagues: List[str], start_date: str, end_date: str, delay: float = 2.0
    ):
        """Scrape data for multiple leagues across a date range."""
        # Generate list of dates to scrape
        dates = self.generate_date_range(start_date, end_date)

        # Track success/failure statistics
        stats = {"total": len(dates) * len(leagues), "success": 0, "failure": 0}

        # Store all scraped data
        all_dfs = []

        for date in dates:
            for league in leagues:
                self.logger.info(f"Scraping {league} for date {date}")

                success, df = self.scrape_data(league, date)

                if success:
                    stats["success"] += 1
                    if not df.empty:
                        all_dfs.append(df)
                else:
                    stats["failure"] += 1

                # Respect rate limits
                time.sleep(delay)

        # Combine all DataFrames and save to single CSV
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"espn_games_combined_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            combined_df.to_csv(filepath, index=False)
            self.logger.info(f"Saved combined data to {filepath}")

            # Create summary report
            self.create_summary_report(stats, filepath, combined_df)
        else:
            self.logger.warning("No data was collected across all scraping attempts")
            self.create_summary_report(stats, None, pd.DataFrame())

        return stats

    def create_summary_report(self, stats: dict, output_file: str, df: pd.DataFrame):
        """Create a summary report of the scraping operation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.logs_dir, f"scraping_summary_{timestamp}.txt")

        with open(report_path, "w") as f:
            f.write("ESPN Scraping Summary Report\n")
            f.write("==========================\n\n")
            f.write(f"Total requests: {stats['total']}\n")
            f.write(f"Successful: {stats['success']}\n")
            f.write(f"Failed: {stats['failure']}\n")
            f.write(f"Success rate: {(stats['success']/stats['total'])*100:.2f}%\n\n")

            if not df.empty and output_file:
                f.write("Data Summary:\n")
                f.write(f"- Total games collected: {len(df)}\n")
                f.write(f"- Leagues represented: {', '.join(df['league'].unique())}\n")
                f.write(f"- Date range: {df['date'].min()} to {df['date'].max()}\n")
                f.write(f"- Output file: {output_file}\n")


def validate_date(date_str: str) -> bool:
    """Validate date string format (YYYYMMDD)."""
    try:
        datetime.strptime(date_str, "%Y%m%d")
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Scrape ESPN data for multiple leagues and dates"
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        choices=sorted(VALID_LEAGUES),
        help="List of leagues to scrape. Allowed values are "
        + ", ".join(sorted(VALID_LEAGUES)),
        metavar="",
    )
    parser.add_argument("--start-date", required=True, help="Start date (YYYYMMDD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYYMMDD)")
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Delay between requests in seconds"
    )
    parser.add_argument(
        "--output-dir", default="data", help="Output directory for scraped data"
    )
    parser.add_argument("--logs-dir", default="logs", help="Directory for log files")

    args = parser.parse_args()

    # Validate dates
    if not all(validate_date(date) for date in [args.start_date, args.end_date]):
        parser.error("Dates must be in YYYYMMDD format")

    # Initialize scraper
    scraper = ESPNScraper(output_dir=args.output_dir, logs_dir=args.logs_dir)

    # Run scraper
    stats = scraper.scrape_range(
        leagues=args.leagues,
        start_date=args.start_date,
        end_date=args.end_date,
        delay=args.delay,
    )

    print(f"\nScraping completed: {stats['success']}/{stats['total']} successful")


if __name__ == "__main__":
    main()
