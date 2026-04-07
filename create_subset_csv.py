import pandas as pd
import numpy as np
import os

def create_subset(
    input_csv="data/metadata/frame_metadata.csv", 
    output_csv="data/metadata/subset_5k_metadata.csv", 
    top_n_species=50, 
    samples_per_species=100, 
    seed=42
):
    print(f"Reading full metadata from {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found in the current directory.")
        return

    df = pd.read_csv(input_csv)
    original_len = len(df)
    
    # Drop rows without a species label
    df = df.dropna(subset=['species'])
    print(f"Dropped {original_len - len(df)} rows with missing species labels.")
    
    # Find the top N most frequent species
    species_counts = df['species'].value_counts()
    top_species = species_counts.head(top_n_species).index.tolist()
    
    print(f"Selected top {top_n_species} species. (e.g., {top_species[:5]}...)")
    
    # Filter dataset to only include top species
    df_top = df[df['species'].isin(top_species)]
    
    # Sample up to `samples_per_species` from each
    # This prevents the most frequent species (e.g. 6k images) from dominating the 5k subset
    subset_dfs = []
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    for sp in top_species:
        sp_df = df_top[df_top['species'] == sp]
        if len(sp_df) > samples_per_species:
            sp_df = sp_df.sample(n=samples_per_species, random_state=seed)
        subset_dfs.append(sp_df)
        
    final_subset_df = pd.concat(subset_dfs).reset_index(drop=True)
    
    # Shuffle the final dataset
    final_subset_df = final_subset_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    print(f"\nFinal subset shape: {final_subset_df.shape}")
    print(f"Total unique species in subset: {final_subset_df['species'].nunique()}")
    
    final_subset_df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully saved 5k verification subset to: {output_csv}")
    print("This subset is perfectly balanced with up to 100 images per species, ideal for VM scaling tests.")

if __name__ == "__main__":
    create_subset()
