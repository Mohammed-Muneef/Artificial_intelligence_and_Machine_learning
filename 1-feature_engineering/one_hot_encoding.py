import pandas as pd
import numpy as np

def explain_top_k_approach():
    print("=== One-Hot Encoding for High Cardinality Variables (Top-K Approach) ===\n")
    print("Theory:")
    print("When a categorical variable has many unique categories (high cardinality),")
    print("standard one-hot encoding can create too many columns (dimensions).")
    print("A common solution is to limit encoding to the top K most frequent labels,")
    print("grouping the less frequent labels into an implicit 'Other' category (all zeros).\n")

def one_hot_top_k():
    # 1. Simulate a dataset with high variety (High Cardinality)
    # Let's say we have a 'City' column with many different cities
    cities = ['New York', 'London', 'Paris', 'Tokyo', 'Dubai'] * 100  # Frequent ones
    cities += ['City_A', 'City_B', 'City_C', 'City_D', 'City_E'] * 5    # Less frequent
    cities += [f'Random_City_{i}' for i in range(50)]                    # Rare ones
    
    np.random.shuffle(cities)
    
    df = pd.DataFrame({'City': cities})
    
    print(f"Dataset created with {len(df)} rows.")
    print(f"Total unique cities: {df['City'].nunique()}")
    print("-" * 30)

    # 2. Decide on K (e.g., Top 10)
    K = 10
    
    # 3. Find the Top K most frequent labels
    top_k_labels = df['City'].value_counts().nlargest(K).index
    
    print(f"Top {K} most frequent cities:")
    print(list(top_k_labels))
    print("-" * 30)

    # 4. Perform One-Hot Encoding for ONLY these K labels
    # We iterate through the top list and create a binary column for each
    for label in top_k_labels:
        # Create column: City_New York, City_London, etc.
        # np.where check: If City == label then 1, else 0
        df[label] = np.where(df['City'] == label, 1, 0)

    print(f"\nResulting DataFrame (showing Top {K} columns):")
    # Display the first few rows with the original column and new dummy columns
    print(df[['City'] + list(top_k_labels)].head(15))
    
    print("\nObservation:")
    print("Rows with 'New York' have a 1 in the 'New York' column.")
    print("Rows with rare cities (e.g., 'Random_City_1') will have 0s in all top-k columns.")

if __name__ == "__main__":
    explain_top_k_approach()
    one_hot_top_k()
