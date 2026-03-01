import numpy as np
import itertools
import pandas as pd
from typing import List, Tuple

def generate_probability_list(binary_subset: List[int]) -> List[float]:
    """
    Generate a probability list for a given binary subset.
    Each '1' in the subset gets a probability from a Dirichlet distribution, '0's get 0.
    """
    indices_of_ones = [i for i, value in enumerate(binary_subset) if value == 1]
    if not indices_of_ones:
        return [0.0] * len(binary_subset)
    num_ones = len(indices_of_ones)
    probabilities_for_ones = np.random.dirichlet(np.ones(num_ones))
    probability_list = [0.0] * len(binary_subset)
    for i, index in enumerate(indices_of_ones):
        probability_list[index] = probabilities_for_ones[i]
    return probability_list

def generate_one_hot_batch(probabilities: List[float], num_samples: int) -> np.ndarray:
    """
    Generate a batch of one-hot encoded samples based on the given probabilities.
    """
    probabilities = np.array(probabilities)
    p_index = np.random.choice(len(probabilities), size=num_samples, p=probabilities)
    one_hot_batch = np.zeros((num_samples, len(probabilities)))
    one_hot_batch[np.arange(num_samples), p_index] = 1
    return one_hot_batch

def generate_data(offer_set: List[int], max_size: int, min_size: int, num_samples_per_subset: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data (X, Y) for all subsets of offer_set of sizes between min_size and max_size.
    """
    probability_lists = []
    binary_subsets = []

    for r in range(min_size, max_size + 1):
        for subset in itertools.combinations(offer_set, r):
            binary_subset = [1 if x in subset else 0 for x in offer_set]
            probability_list = generate_probability_list(binary_subset)
            probability_lists.append(probability_list)
            binary_subsets.append(binary_subset)

    X = [subset for subset in binary_subsets for _ in range(num_samples_per_subset)]
    X = np.array(X)

    Y = [generate_one_hot_batch(p, num_samples_per_subset) for p in probability_lists]
    Y = np.concatenate(Y, axis=0)    

    return X, Y

def save_dataset_to_csv(X: np.ndarray, Y: np.ndarray, offer_set: List[int], filename: str) -> None:
    """
    Save the dataset (X, Y) to a CSV file with appropriate column names.
    """
    dataset = np.hstack((X, Y))
    columns = [f"X{i}" for i in offer_set] + [f"Y{i}" for i in offer_set]
    df = pd.DataFrame(dataset, columns=columns)
    df.to_csv(filename, index=False)
    
    print(df.head())
    print(df.columns)

def main():
    np.random.seed(20)

    num_products = 20
    train_size = 80
    test_size = 20
    offer_set = list(range(num_products))
    max_assortment_size = 15
    min_assortment_size = 15

    X_train, Y_train = generate_data(
        offer_set=offer_set,
        max_size=max_assortment_size,
        min_size=min_assortment_size,
        num_samples_per_subset=train_size
    )

    X_test, Y_test = generate_data(
        offer_set=offer_set,
        max_size=max_assortment_size,
        min_size=min_assortment_size,
        num_samples_per_subset=test_size
    )

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    save_dataset_to_csv(X_train, Y_train, offer_set, "data/Synthetic_20-15-80_Train.csv")
    save_dataset_to_csv(X_test, Y_test, offer_set, "data/Synthetic_20-15-80_Test.csv")

if __name__ == "__main__":
    main()
