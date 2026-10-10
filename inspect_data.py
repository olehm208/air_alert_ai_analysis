"""Run from the project root: python inspect_data.py"""
import argparse
from pathlib import Path

from data_loader import load_alerts, region_coverage, kyiv_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    for source in ("volunteer", "official"):
        df = load_alerts(source, args.data_dir)
        coverage = region_coverage(df)
        print(f"\n=== {source.upper()} ===")
        print(f"Unique rows: {len(df):,}")
        print(f"First start (Kyiv): {kyiv_time(df['started_at']).min()}")
        print(f"Last start (Kyiv):  {kyiv_time(df['started_at']).max()}")
        print(f"Estimated finish times: {df['estimated_end'].sum():,}")
        print("\nCrimea / Sevastopol / Luhansk:")
        print(coverage.loc[coverage['region'].str.contains(
            'Crimea|Sevastopol|Luhansk', case=False
        ), ['region', 'records', 'coverage']].to_string(index=False))


if __name__ == "__main__":
    main()
