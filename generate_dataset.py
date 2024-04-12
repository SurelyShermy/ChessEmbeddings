#ssh into regan server to access aifs, and preprocess datafiles with necessary information
import os
import numpy as np

def prep_data(directory):
    trainset = []
    valset = []
    train_board_pairs = []
    train_distances = []
    val_board_pairs = []
    val_distances = []
    tournaments = os.listdir(directory)
    for tournament in tournaments:
        tournament_path = os.path.join(directory, tournament)
        games = os.listdir(tournament_path)
        for game in games:
            game_path = os.path.join(tournament_path, game)
            game_data = np.load(game_path, allow_pickle=True).item()
            if game_data["data_subset"] == "Train":
                trainset.append(game_data)
            elif game_data["data_subset"] == "Test":
                valset.append(game_data)
    for game in trainset:
        for engine in game:
            if len(engine) == 0:
                continue
            for moves in engine:
                try:
                    for i in range(len(moves)-2):
                        board1 = {}
                        board2 = {}
                        board1["FEN"] = moves[i]["FEN"]
                        board1["Turn"] = moves[i]["Turn"]
                        board2["FEN"] = moves[i+1]["FEN"]
                        board2["Turn"] = moves[i+1]["Turn"]
                        distance = moves[i]["Eval"] - moves[i+1]["Eval"]
                        train_board_pairs.append((board1, board2))
                        train_distances.append(distance)
                except:
                    print(f"Error in {engine} in valset")
                    pass
    for game in valset:
        for engine in game:
            if len(engine) == 0:
                continue
            for moves in engine:
                try:
                    for i in range(len(moves)-2):
                        board1 = {}
                        board2 = {}
                        board1["FEN"] = moves[i]["FEN"]
                        board1["Turn"] = moves[i]["Turn"]
                        board2["FEN"] = moves[i+1]["FEN"]
                        board2["Turn"] = moves[i+1]["Turn"]
                        distance = moves[i]["Eval"] - moves[i+1]["Eval"]
                        val_board_pairs.append((board1, board2))
                        val_distances.append(distance)
                except:
                    print(f"Error in {engine} in valset")
                    pass
    return train_board_pairs, train_distances, val_board_pairs, val_distances