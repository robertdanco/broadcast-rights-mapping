import pandas as pd
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime


@dataclass
class WatchOption:
    """Represents a single watch option for a game"""

    watch_option: str
    availability: str  # 'national' or 'regional'
    market_team: str  # empty string if national broadcast
    confidence_score: float


@dataclass
class GameBroadcast:
    """Represents broadcast information for a single game"""

    date: str
    league: str
    home_team: str
    away_team: str
    national_options: List[str]
    home_team_options: List[str]
    away_team_options: List[str]

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "league": self.league,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "national_options": json.dumps(
                self.national_options
            ),  # Convert list to JSON string
            "home_team_options": json.dumps(self.home_team_options),
            "away_team_options": json.dumps(self.away_team_options),
        }


class DataIntegrator:
    def __init__(self, output_dir: str = "data", logs_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.logs_dir = Path(logs_dir)

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

    def _setup_logging(self):
        """Configure logging with both file and console handlers"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.logs_dir / "data_integration.log"),
                logging.StreamHandler(),
            ],
        )

    def load_schedule(self, filepath: str) -> pd.DataFrame:
        """Load and validate schedule data"""
        try:
            df = pd.read_csv(filepath)
            required_columns = [
                "date",
                "league",
                "home_team",
                "away_team",
                "watch_options",
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            return df

        except Exception as e:
            self.logger.error(f"Error loading schedule file: {str(e)}")
            raise

    def load_zip_mapping(self, filepath: str) -> pd.DataFrame:
        """Load and validate ZIP code mapping data"""
        try:
            df = pd.read_csv(filepath)
            required_columns = ["zip_code", "in_market_teams"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            return df

        except Exception as e:
            self.logger.error(f"Error loading ZIP mapping file: {str(e)}")
            raise

    def load_watch_metadata(self, filepath: str) -> Dict:
        """Load and validate watch option metadata"""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return data

        except Exception as e:
            self.logger.error(f"Error loading watch metadata file: {str(e)}")
            raise

    def get_game_key(self, row: pd.Series) -> str:
        """Generate consistent game key matching metadata format"""
        return f"{row['date']}_{row['away_team']}_at_{row['home_team']}"

    def analyze_game_broadcasts(
        self, schedule_row: pd.Series, metadata: Dict
    ) -> Optional[GameBroadcast]:
        """Analyze broadcast options for a single game"""
        try:
            game_key = self.get_game_key(schedule_row)

            if game_key not in metadata:
                self.logger.warning(f"No metadata found for game: {game_key}")
                return None

            watch_options = metadata[game_key]

            # Initialize empty lists for different types of watch options
            national_options = []
            home_team_options = []
            away_team_options = []

            # Categorize each watch option
            for option in watch_options:
                if option["availability"] == "national":
                    national_options.append(option["watch_option"])
                elif option["availability"] == "regional":
                    if option["market_team"] == schedule_row["home_team"]:
                        home_team_options.append(option["watch_option"])
                    elif option["market_team"] == schedule_row["away_team"]:
                        away_team_options.append(option["watch_option"])

            return GameBroadcast(
                date=schedule_row["date"],
                league=schedule_row["league"],
                home_team=schedule_row["home_team"],
                away_team=schedule_row["away_team"],
                national_options=national_options,
                home_team_options=home_team_options,
                away_team_options=away_team_options,
            )

        except Exception as e:
            self.logger.error(f"Error analyzing game broadcasts: {str(e)}")
            return None

    def postprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert JSON string lists back to Python lists for analysis"""
        list_columns = ["national_options", "home_team_options", "away_team_options"]
        for col in list_columns:
            df[col] = df[col].apply(json.loads)
        return df

    def process_data(
        self, schedule_file: str, zip_mapping_file: str, metadata_file: str
    ) -> Tuple[pd.DataFrame, str]:
        """Process and integrate all data sources"""
        try:
            # Load data
            self.logger.info("Loading data files...")
            schedule_df = self.load_schedule(schedule_file)
            zip_mapping_df = self.load_zip_mapping(zip_mapping_file)
            metadata = self.load_watch_metadata(metadata_file)

            # Process each game
            self.logger.info("Processing games...")
            results = []
            for _, row in schedule_df.iterrows():
                broadcast_info = self.analyze_game_broadcasts(row, metadata)
                if broadcast_info:
                    results.append(broadcast_info)

            # Generate output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"broadcast_analysis_{timestamp}.csv"

            # Convert results to DataFrame
            results_df = pd.DataFrame([r.to_dict() for r in results])

            # Save to CSV
            results_df.to_csv(output_file, index=False)

            # Convert back to Python lists for return value
            results_df = self.postprocess_dataframe(results_df)

            self.logger.info(f"Analysis complete. Processed {len(results)} games.")
            return results_df, str(output_file)

        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise


def main():
    """Main entry point for the script"""
    import argparse

    parser = argparse.ArgumentParser(description="Integrate sports broadcast data")
    parser.add_argument("--schedule", required=True, help="Path to schedule CSV file")
    parser.add_argument(
        "--zip-mapping", required=True, help="Path to ZIP mapping CSV file"
    )
    parser.add_argument(
        "--metadata", required=True, help="Path to watch metadata JSON file"
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--logs-dir", default="logs", help="Logs directory")

    args = parser.parse_args()

    try:
        integrator = DataIntegrator(args.output_dir, args.logs_dir)
        results_df, output_file = integrator.process_data(
            args.schedule, args.zip_mapping, args.metadata
        )

        print(f"\nProcessing complete!")
        print(f"Results saved to: {output_file}")
        print("\nSample of results:")
        print(results_df.head().to_string())

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
