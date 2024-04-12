#ssh into regan server to access aifs, and preprocess datafiles with necessary information
import os
import numpy as np

def prep_data(train_board_pairs, train_distances, val_board_pairs, val_distances):
    trainset = []
    valset = []
    tournaments = os.listdir("./game_dataset")
    for tournament in tournaments:
        tournament_path = os.path.join("./game_dataset", tournament)
        games = os.listdir(tournament_path)
        for game in games:
            game_path = os.path.join(tournament_path, game)
            game_data = np.load(game_path, allow_pickle=True).item()
            if game_data["data_set"] == "train":
                trainset.append(game_data)
            elif game_data["data_set"] == "test":
                valset.append(game_data)
    for game in trainset:
        for engine in game:
            for i in range(len(engine)-2):
                board1 = engine[i]["FEN"]
                board2 = engine[i+1]["FEN"]
                distance = engine[i]["Eval"] - engine[i+1]["Eval"]
                train_board_pairs.append((board1, board2))
                train_distances.append(distance)
    for game in valset:
        for engine in game:
            for i in range(len(engine)-2):
                board1 = engine[i]["FEN"]
                board2 = engine[i+1]["FEN"]
                distance = engine[i]["Eval"] - engine[i+1]["Eval"]
                val_board_pairs.append((board1, board2))
                val_distances.append(distance)
    return train_board_pairs, train_distances, val_board_pairs, val_distances