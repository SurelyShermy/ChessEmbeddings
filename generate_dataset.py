#pgn->aif->npy dictionary
#split by 70/20/10

# to do : do train test split.

import random
from chessboard_utils import generate_chessboard_pair, calculate_distance

def generate_dataset(num_samples):
    dataset = []
    for _ in range(num_samples):
        chessboard1, chessboard2 = generate_chessboard_pair()
        distance = calculate_distance(chessboard1, chessboard2)
        dataset.append((chessboard1, chessboard2, distance))
    return dataset

# Generate and save the train and validation datasets
train_dataset = generate_dataset(100000)
val_dataset = generate_dataset(20000)

with open('train_dataset.json', 'w') as file:
    json.dump(train_dataset, file)

with open('val_dataset.json', 'w') as file:
    json.dump(val_dataset, file)