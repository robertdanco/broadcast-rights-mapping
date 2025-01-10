import argparse
from datetime import datetime, timedelta
import logging
import os
import time
from typing import List, Tuple
import pandas as pd
from espn_game_scraper import scrape_espn_games, VALID_LEAGUES


class ESPNScraper:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(output_dir, "scraper.log")),
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

    def scrape_and_save(self, league: str, date: str) -> Tuple[bool, str]:
        """Scrape data for a specific league and date and save to CSV."""
        try:
            # Use existing scrape_espn_games function
            df, scraped_date = scrape_espn_games(league, date)

            if len(df) > 0:
                # Generate filename with league, date and timestamp
                current_ts = int(time.time())
                filename = f"espn_games_{league}_{scraped_date}_{current_ts}.csv"
                filepath = os.path.join(self.output_dir, filename)

                # Save to CSV
                df.to_csv(filepath, index=False)
                self.logger.info(f"Saved data to {filepath}")
                return True, filepath
            else:
                self.logger.info(f"No games found for {league} on {date}")
                return True, ""

        except Exception as e:
            self.logger.error(f"Error scraping {league} for {date}: {str(e)}")
            return False, str(e)

    def scrape_range(
        self, leagues: List[str], start_date: str, end_date: str, delay: float = 2.0
    ):
        """Scrape data for multiple leagues across a date range."""
        # Generate list of dates to scrape
        dates = self.generate_date_range(start_date, end_date)

        # Track success/failure statistics
        stats = {"total": len(dates) * len(leagues), "success": 0, "failure": 0}

        # Store all scraped data for summary
        all_data = []

        for date in dates:
            for league in leagues:
                self.logger.info(f"Scraping {league} for date {date}")

                success, result = self.scrape_and_save(league, date)

                if success:
                    stats["success"] += 1
                    if result:  # If there was data saved
                        all_data.append(
                            {"date": date, "league": league, "file": result}
                        )
                else:
                    stats["failure"] += 1
                    self.logger.error(f"Failed to scrape {league} for {date}: {result}")

                # Respect rate limits with random jitter
                time.sleep(delay)

        # Create summary report
        self.create_summary_report(stats, all_data)

        return stats

    def create_summary_report(self, stats: dict, data: List[dict]):
        """Create a summary report of the scraping operation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"scraping_summary_{timestamp}.txt")

        with open(report_path, "w") as f:
            f.write("ESPN Scraping Summary Report\n")
            f.write("==========================\n\n")
            f.write(f"Total requests: {stats['total']}\n")
            f.write(f"Successful: {stats['success']}\n")
            f.write(f"Failed: {stats['failure']}\n")
            f.write(f"Success rate: {(stats['success']/stats['total'])*100:.2f}%\n\n")

            if data:
                f.write("Files generated:\n")
                for entry in data:
                    f.write(f"- {entry['league']} ({entry['date']}): {entry['file']}\n")


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
        help="List of leagues to scrape",
    )
    parser.add_argument("--start-date", required=True, help="Start date (YYYYMMDD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYYMMDD)")
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Delay between requests in seconds"
    )
    parser.add_argument(
        "--output-dir", default="data", help="Output directory for scraped data"
    )

    args = parser.parse_args()

    # Validate dates
    if not all(validate_date(date) for date in [args.start_date, args.end_date]):
        parser.error("Dates must be in YYYYMMDD format")

    # Initialize scraper
    scraper = ESPNScraper(output_dir=args.output_dir)

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
