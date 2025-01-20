import asyncio
import aiohttp
import pandas as pd
import logging
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple, Union
from tqdm.asyncio import tqdm_asyncio
import argparse


def validate_zip_code(zip_code: str) -> Tuple[bool, str]:
    """Validate ZIP code format."""
    try:
        if not zip_code.isdigit() or len(zip_code) != 5:
            return False, "ZIP code must be exactly 5 digits"
        return True, ""
    except Exception as e:
        return False, str(e)


def get_headers() -> Dict[str, str]:
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


class NBAPostalLookup:
    def __init__(self, output_dir: str = "data", logs_dir: str = "logs"):
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        self.base_url = (
            "https://content-api-prod.nba.com/public/1/leagues/nba/blackouts"
        )

        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(logs_dir, "nba_lookup.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    async def process_zip_code(
        self, session: aiohttp.ClientSession, zip_code: str
    ) -> Tuple[bool, str]:
        """Process a single ZIP code through the NBA API."""
        is_valid, error_message = validate_zip_code(zip_code)
        if not is_valid:
            self.logger.error(f"Invalid ZIP code {zip_code}: {error_message}")
            return False, ""

        try:
            url = f"{self.base_url}?zip={zip_code}"
            async with session.get(url, headers=get_headers()) as response:
                if response.status != 200:
                    return False, ""

                data = await response.json()
                if not data or "results" not in data or not data["results"]:
                    return True, ""

                # Extract and sort team abbreviations
                teams = sorted(team["abbr"] for team in data["results"])
                return True, ",".join(teams)

        except Exception as e:
            self.logger.error(f"Error processing ZIP code {zip_code}: {str(e)}")
            return False, ""

    async def process_batch(
        self, session: aiohttp.ClientSession, zip_codes: List[str]
    ) -> List[dict]:
        """Process a batch of ZIP codes concurrently."""
        tasks = []
        for zip_code in zip_codes:
            tasks.append(self.process_zip_code(session, zip_code.zfill(5)))

        results = await asyncio.gather(*tasks)
        return [
            {"zip_code": zc, "in_market_teams": teams}
            for (zc, (success, teams)) in zip(zip_codes, results)
            if success
        ]

    async def process_zip_codes(
        self, zip_codes: List[str], batch_size: int = 10, rate_limit: int = 50
    ) -> pd.DataFrame:
        """
        Process multiple ZIP codes concurrently and return results as a DataFrame.

        Args:
            zip_codes: List of ZIP codes to process
            batch_size: Number of concurrent requests
            rate_limit: Maximum requests per second
        """
        results = []
        total_batches = (len(zip_codes) + batch_size - 1) // batch_size
        batches = [
            zip_codes[i : i + batch_size] for i in range(0, len(zip_codes), batch_size)
        ]

        # Configure connection pooling
        connector = aiohttp.TCPConnector(limit=batch_size)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            self.logger.info(
                f"Processing {len(zip_codes)} ZIP codes in {total_batches} batches"
            )

            pbar = tqdm_asyncio(
                range(total_batches), desc="Processing ZIP codes", unit="batch"
            )
            async for i in pbar:
                batch = batches[i]
                batch_results = await self.process_batch(session, batch)
                results.extend(batch_results)
                # Rate limiting
                await asyncio.sleep(len(batch_results) / rate_limit)

        if not results:
            self.logger.warning("No successful results were obtained")
            return pd.DataFrame(columns=["zip_code", "in_market_teams"])

        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame) -> str:
        """Save results to CSV file and return the filepath."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nba_market_teams_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        df.to_csv(filepath, index=False)
        self.logger.info(f"Results saved to {filepath}")

        return filepath

    def create_summary_report(
        self, total: int, success: int, filepath: Union[str, None]
    ) -> None:
        """Create a summary report of the lookup operation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.logs_dir, f"lookup_summary_{timestamp}.txt")

        with open(report_path, "w") as f:
            f.write("NBA Postal Code Lookup Summary Report\n")
            f.write("===================================\n\n")
            f.write(f"Total ZIP codes processed: {total}\n")
            f.write(f"Successful lookups: {success}\n")
            f.write(f"Failed lookups: {total - success}\n")
            f.write(f"Success rate: {(success/total)*100:.2f}%\n\n")

            if filepath:
                f.write(f"Output file: {filepath}\n")
            else:
                f.write(
                    "No output file was generated due to lack of successful results\n"
                )


async def main_async():
    parser = argparse.ArgumentParser(
        description="Look up NBA market teams for ZIP codes"
    )
    parser.add_argument(
        "--input-file",
        default="mapping_files/zip_dma_mapping_testing.csv",
        help="Input CSV file containing ZIP codes (default: mapping_files/zip_dma_mapping_testing.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=50,
        help="Maximum requests per second (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for results (default: data)",
    )
    parser.add_argument(
        "--logs-dir", default="logs", help="Directory for log files (default: logs)"
    )

    args = parser.parse_args()

    processor = NBAPostalLookup(output_dir=args.output_dir, logs_dir=args.logs_dir)

    try:
        processor.logger.info(f"Reading ZIP codes from {args.input_file}")
        df_input = pd.read_csv(args.input_file)
        zip_codes = [str(int(zip_code)).zfill(5) for zip_code in df_input["zip_code"]]
        processor.logger.info(f"Found {len(zip_codes)} ZIP codes to process")

        results_df = await processor.process_zip_codes(
            zip_codes, batch_size=args.batch_size, rate_limit=args.rate_limit
        )

        if len(results_df) > 0:
            output_file = processor.save_results(results_df)
        else:
            output_file = None
            processor.logger.warning("No results to save")

        processor.create_summary_report(
            total=len(zip_codes), success=len(results_df), filepath=output_file
        )

    except Exception as e:
        processor.logger.error(f"Script execution failed: {str(e)}")
        raise


def main():
    """Entry point for the script."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
