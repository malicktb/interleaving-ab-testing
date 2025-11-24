import pandas as pd
import numpy as np
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq

def parse_line(line: str):
    parts = line.strip().split('|')
    record_id = int(parts[0])
    context_features = json.loads(parts[1])
    items_features = json.loads(parts[2])
    items_dense_features = json.loads(parts[3])
    items_dense_features2 = json.loads(parts[4])
    labels = json.loads(parts[5])
    
    return {
        'id': record_id,
        'context_features': context_features,
        'items_features': items_features,
        'items_dense_features': items_dense_features,
        'items_dense_features2': items_dense_features2,
        'labels': labels
    }

def process_chunk(lines, chunk_index: int, output_dir: str) -> str:
    records = [parse_line(line) for line in lines]
    df = pd.DataFrame(records)
    output_file = os.path.join(output_dir, f"chunk_{chunk_index:04d}.parquet")
    df.to_parquet(output_file, index=False, compression='snappy')
    return output_file

def split_and_convert(input_file: str, output_dir: str, chunk_size: int = 10000):
    os.makedirs(output_dir, exist_ok=True)
    chunk_index = 0
    current_chunk = []
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f):
            current_chunk.append(line)
            
            # When chunk is full, process it
            if len(current_chunk) >= chunk_size:
                output_file = process_chunk(current_chunk, chunk_index, output_dir)
                print(f"Processed chunk {chunk_index}: {output_file}")
                chunk_index += 1
                current_chunk = []
                
                # Print progress every 100 chunks
                if chunk_index % 100 == 0:
                    print(f"Processed {chunk_index} chunks...")
    
    # Process remaining lines
    if current_chunk:
        output_file = process_chunk(current_chunk, chunk_index, output_dir)
        print(f"Processed final chunk {chunk_index}: {output_file}")
    
    print(f"Conversion complete! Total chunks: {chunk_index + 1}")

if __name__ == "__main__":
    input_file = "/set1.train.txt.part2"
    output_dir = "parquet_chunks"
    chunk_size = 10000
    print(f"Starting conversion of {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size: {chunk_size} lines")
    
    split_and_convert(input_file, output_dir, chunk_size)
