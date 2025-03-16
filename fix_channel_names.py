"""
Helper script to check the positions of EOG channels in the standard_1020 list
"""
from dataset import standard_1020

# Define EOG channels from the dataset
eog_channels = [
    'EOGD', 'EOGDL', 'EOGDR', 'EOGDiagLU', 'EOGDiagR', 'EOGL', 'EOGLH',
    'EOGR', 'EOGRD', 'EOGRH', 'EOGRU', 'EOGU', 'EOGUL', 'EOGUR'
]

def main():
    # Print positions of EOG channels
    print("EOG channel positions in standard_1020:")
    print("=" * 50)
    print(f"{'Channel':<15} | {'Index':<5} | {'Within 64 range?'}")
    print("-" * 50)
    
    for ch in eog_channels:
        try:
            index = standard_1020.index(ch)
            within_range = "Yes" if index < 64 else "No"
            print(f"{ch:<15} | {index:<5} | {within_range}")
        except ValueError:
            print(f"{ch:<15} | Not found in standard_1020")
    
    print("\nTotal channels in standard_1020:", len(standard_1020))
    print("Indices beyond 64:", [i for i, ch in enumerate(standard_1020) if i >= 64])

if __name__ == "__main__":
    main()