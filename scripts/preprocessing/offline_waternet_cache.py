import os
import argparse
import pandas as pd
import cv2
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Suppress PyTorch Hub warnings
import warnings
warnings.filterwarnings("ignore")

# --- repo-root bootstrap: this script lives in scripts/<area>/; add the
# repo root (two levels up) to sys.path so `import bioreef` resolves no
# matter the cwd or how the script is invoked. ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
from bioreef.data.data_factory import WaterNetRestorer

def process_single_image(row, input_dir, output_dir, restorer):
    """
    Reads an image, applies WaterNet, and saves it. 
    Thread-safe logic due to Python's GIL + PyTorch backend handling.
    """
    filename = row['file_name']
    in_path = os.path.join(input_dir, filename)
    out_path = os.path.join(output_dir, filename)
    
    # Skip if already exists (and aggressively aggressively enforce the "Rolling Deletion" on the raw file)
    if os.path.exists(out_path):
        if os.path.exists(in_path):
            try:
                os.remove(in_path)
            except Exception:
                pass
        return True
        
    # Read the authentic image
    img = cv2.imread(in_path)
    if img is None:
        return False
        
    try:
        # Apply restoration via WaterNet wrapper loaded on GPU
        restored = restorer(img)
        # Save securely in lossless PNG format
        cv2.imwrite(out_path, restored)
        
        # Execute Rolling Deletion to completely freeze disk-space consumption
        try:
            os.remove(in_path)
        except Exception as e:
            pass
            
        return True
    except Exception as e:
        print(f"Failed to restore {filename}: {e}")
        return False

def run_cache_pipeline(csv_path, input_dir, output_dir, workers=16):
    print(f"Starting Offline WaterNet Extractor")
    print(f"Target CSV      : {csv_path}")
    print(f"Input Directory : {input_dir}")
    print(f"Output Directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the target CSV
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"Identified {total_samples} images for offline caching.")
    
    # 2. Initialize the WaterNet Restorer strictly onto GPU 0 for maximum throughput
    import logging
    logging.basicConfig(level=logging.INFO)
    print("Loading specialized WaterNet pre-trained weights...")
    restorer = WaterNetRestorer()
    # Force initialization
    restorer._load_model()
    
    if hasattr(restorer, '_model') and "Identity" in str(type(restorer._model)):
        print("\n[!] CRITICAL ERROR: WaterNet failed to load from PyTorch Hub!")
        print("The script silently fell back to an Identity (passthrough) matrix.")
        print("Please review the logger output above to see exactly what dependency the VM is missing.")
        return
        
    print("WaterNet loaded securely onto the GPU.")
    
    # 3. Fire the thread pool for extreme IO saturation utilizing the 128GB of RAM
    print(f"Booting {workers} I/O workers to saturate system memory constraints...")
    
    success_count = 0
    fail_count = 0
    
    # Notice we iterate using pure row dicts for thread safety
    rows = df.to_dict('records')
    
    # Using ThreadPoolExecutor because cv2.imread and PyTorch release the GIL
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_image, row, input_dir, output_dir, restorer) for row in rows]
        
        for future in tqdm(futures, desc="Restoring Dataset", unit="img"):
            if future.result():
                success_count += 1
            else:
                fail_count += 1
                
    print("\n--- Cache Summary ---")
    print(f"Successful Restorations : {success_count}/{total_samples}")
    print(f"Failed I/O Drops        : {fail_count}")
    print(f"Target Cache            : {output_dir}")
    if fail_count == 0:
        print("Dataset cleanly stabilized. DDP dataloadables are fully optimized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioReef - Massive Offline WaterNet Cache")
    parser.add_argument("--csv", type=str, default="data_oz/metadata/frame_metadata.csv", help="Path to the target metadata CSV")
    parser.add_argument("--input_dir", type=str, default="data_oz/frames", help="Raw images directory")
    parser.add_argument("--output_dir", type=str, default="/media/openuae/UUI/frames_waternet_3", help="Target cache directory")
    parser.add_argument("--workers", type=int, default=16, help="IO Threads. Requires massive system RAM.")
    
    import numpy as np # Needed for the dummy init
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"CRITICAL ERROR: Given CSV '{args.csv}' does not exist.")
        print("Please run `create_subset_csv.py` first if testing, or point this to `frame_metadata.csv`.")
    else:
        run_cache_pipeline(args.csv, args.input_dir, args.output_dir, args.workers)
