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
        for engine in game["Engines"]:
            for i in range(len(game["Engines"][engine])-2):
                board1 = {}
                board2 = {}
                board1["FEN"] = game["Engines"][engine][i]["FEN"]
                board1["Turn"] =game["Engines"][engine][i]["Turn"]
                board2["FEN"] = game["Engines"][engine][i+1]["FEN"]
                board2["Turn"] = game["Engines"][engine][i+1]["Turn"]
                # print(board1["FEN"], board2["FEN"])
                distance = np.double(game["Engines"][engine][i]["Eval"]) - np.double(game["Engines"][engine][i+1]["Eval"])
                train_board_pairs.append((board1, board2))
                train_distances.append(distance)
                # except Exception as e:
                #     print(e)
                #     pass
    for game in valset:
        for engine in game["Engines"]:
            for i in range(len(game["Engines"][engine])-2):
                board1 = {}
                board2 = {}
                board1["FEN"] = game["Engines"][engine][i]["FEN"]
                board1["Turn"] =game["Engines"][engine][i]["Turn"]
                board2["FEN"] = game["Engines"][engine][i+1]["FEN"]
                board2["Turn"] = game["Engines"][engine][i+1]["Turn"]
                # print(board1["FEN"], board2["FEN"])
                distance = np.double(game["Engines"][engine][i]["Eval"]) - np.double(game["Engines"][engine][i+1]["Eval"])
                val_board_pairs.append((board1, board2))
                val_distances.append(distance)
                # except Exception as e:
                #     print(e)
                #     pass
    return train_board_pairs, train_distances, val_board_pairs, val_distances