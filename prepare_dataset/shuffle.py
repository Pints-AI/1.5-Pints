import os
import random
from pathlib import Path
from typing import Optional, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse

def read_file(filepath: Path, parquet_columns: Optional[List[str]] = None) -> pa.Table:
    """Read a single Parquet file."""
    contents = pq.read_table(filepath, columns=parquet_columns)
    return contents

def read_and_shuffle_all_files(directory: str) -> pa.Table:
    """Read all Parquet files in the directory, concatenate and shuffle them."""
    all_tables = []
    file_list = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    
    # Iterate over all files in the directory with a progress bar
    for filename in tqdm(file_list, desc="Reading and concatenating files", unit="file"):
        file_path = Path(directory) / filename
        file_contents = read_file(file_path)  # Read the entire file
        all_tables.append(file_contents)
    
    print("Concatenate all the tables into one ...")
    combined_table = pa.concat_tables(all_tables)
    
    print("Convert to pandas for shuffling ...")
    df = combined_table.to_pandas()
    
    print("Shuffle the DataFrame ...")
    df = df.sample(frac=1).reset_index(drop=True)
    
    print("Convert back to Arrow Table ...")
    shuffled_table = pa.Table.from_pandas(df)
    
    return shuffled_table

def save_to_parquet(table: pa.Table, batch_size: int, output_dir: str):
    """Save the shuffled table to Parquet files in batches."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Convert table to pandas DataFrame for batching ...")
    df = table.to_pandas()
    
    # Split the data into batches and save each batch as a separate parquet file with a progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="Saving shuffled data", unit="batch"):
        batch_df = df.iloc[i:i + batch_size]
        batch_table = pa.Table.from_pandas(batch_df)
        output_file = output_dir / f'shuffled_part_{i//batch_size}.parquet'
        pq.write_table(batch_table, output_file)

def main(directory_path: str, output_directory: str, batch_size: int):
    """Main function to shuffle and save Parquet files."""
    print(f"Reading from directory: {directory_path}")
    print(f"Shuffling and saving to: {output_directory}")
    print(f"Using batch size: {batch_size}")

    shuffled_table = read_and_shuffle_all_files(directory_path)
    save_to_parquet(shuffled_table, batch_size, output_directory)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Shuffle Parquet files and save them in batches.")
    parser.add_argument("--directory_path", type=str, required=True, help="Path to the input directory containing Parquet files.")
    parser.add_argument("--output_directory", type=str, required=True, help="Path to the output directory where shuffled files will be saved.")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for saving Parquet files. Default is 10000.")

    args = parser.parse_args()

    main(args.directory_path, args.output_directory, args.batch_size)
