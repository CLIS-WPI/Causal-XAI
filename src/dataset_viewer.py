import h5py
import numpy as np

with h5py.File('C:\\Users\\snatanzi\\Desktop\\Causal-XAI\\results\\channel_data_step_280.h5', 'r') as f:
    print("Keys in the file:", list(f.keys()))
    
    print("\nExploring each group's contents:")
    for group_name in f.keys():
        print(f"\n{group_name}:")
        group = f[group_name]
        
        # Print group structure
        print("Group contents:", list(group.keys()))
        
        # Try to access actual datasets within the group
        for dataset_name in group.keys():
            print(f"\nDataset: {dataset_name}")
            try:
                dataset = group[dataset_name]
                print(f"Shape: {dataset.shape}")
                print(f"Type: {dataset.dtype}")
                print("First 5 values:")
                if len(dataset.shape) > 0:  # Check if it's not empty
                    print(dataset[0:5])
                else:
                    print("Empty dataset")
            except Exception as e:
                print(f"Error reading dataset: {e}")