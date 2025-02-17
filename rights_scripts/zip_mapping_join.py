import pandas as pd
import argparse
from datetime import datetime
import os

# python zip_mapping_join.py --zip-dma mapping_files/zip_dma_mapping.csv --results data/combined_results_20250129_162056.csv --output-dir data


def create_team_mapping(input_zip_dma: str, input_results: str, output_dir: str) -> str:
    """
    Create a combined mapping file with DMA and team information.

    Args:
        input_zip_dma (str): Path to the ZIP-to-DMA mapping CSV file
        input_results (str): Path to the results CSV file with team information
        output_dir (str): Directory where the output CSV will be saved

    Returns:
        str: Path to the created output file
    """
    # Read input files
    zip_dma_df = pd.read_csv(input_zip_dma)
    results_df = pd.read_csv(input_results)

    # Drop rows with errors
    clean_results = results_df[results_df["error_code"].isna()]

    # Create separate dataframes for each league
    mlb_teams = clean_results[clean_results["league"] == "MLB"][
        ["zip_code", "in_market_teams"]
    ]
    nba_teams = clean_results[clean_results["league"] == "NBA"][
        ["zip_code", "in_market_teams"]
    ]
    nhl_teams = clean_results[clean_results["league"] == "NHL"][
        ["zip_code", "in_market_teams"]
    ]

    # Rename columns to be more specific
    mlb_teams = mlb_teams.rename(columns={"in_market_teams": "mlb_teams"})
    nba_teams = nba_teams.rename(columns={"in_market_teams": "nba_teams"})
    nhl_teams = nhl_teams.rename(columns={"in_market_teams": "nhl_teams"})

    # Merge all dataframes
    # Start with ZIP/DMA mapping
    final_df = zip_dma_df.copy()

    # Convert zip_code to string with leading zeros for consistent merging
    final_df["zip_code"] = final_df["zip_code"].astype(str).str.zfill(5)
    mlb_teams["zip_code"] = mlb_teams["zip_code"].astype(str).str.zfill(5)
    nba_teams["zip_code"] = nba_teams["zip_code"].astype(str).str.zfill(5)
    nhl_teams["zip_code"] = nhl_teams["zip_code"].astype(str).str.zfill(5)

    # Merge with team data
    final_df = final_df.merge(mlb_teams, on="zip_code", how="left")
    final_df = final_df.merge(nba_teams, on="zip_code", how="left")
    final_df = final_df.merge(nhl_teams, on="zip_code", how="left")

    # Reorder columns to match requested format
    final_df = final_df[
        [
            "dma_code",
            "dma_description",
            "zip_code",
            "mlb_teams",
            "nba_teams",
            "nhl_teams",
        ]
    ]

    # Create timestamp and output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"zip_dma_team_mapping_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    final_df.to_csv(output_path, index=False)
    print(f"\nCreated mapping file: {output_path}")
    print(f"Total rows: {len(final_df)}")

    # Print some basic statistics
    print("\nCoverage statistics:")
    print(f"ZIP codes with MLB teams: {final_df['mlb_teams'].notna().sum()}")
    print(f"ZIP codes with NBA teams: {final_df['nba_teams'].notna().sum()}")
    print(f"ZIP codes with NHL teams: {final_df['nhl_teams'].notna().sum()}")

    return output_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a combined ZIP/DMA/Teams mapping file"
    )
    parser.add_argument(
        "--zip-dma", required=True, help="Path to the ZIP-to-DMA mapping CSV file"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to the results CSV file with team information",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where the output CSV will be saved (default: output)",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    try:
        # Create the mapping file
        output_file = create_team_mapping(args.zip_dma, args.results, args.output_dir)
        print(f"\nScript completed successfully!")

    except FileNotFoundError as e:
        print(f"\nError: Could not find input file - {str(e)}")
        exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
