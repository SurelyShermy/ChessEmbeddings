import re
import os
from Engines_utility_table import get_custom_utility
def parse_games_from_aif(filename, content):
    """
    Parses games from the AIF content and returns structured data.

    This example assumes games are separated by a distinct pattern or keyword,
    and each game contains structured data that can be extracted using regular expressions or string manipulation.
    """
    # all filenames follow the convention "<tournamentname>_<enginename>d<depth>.aif"
    splitfilename = filename.split("_")
    tournament_name = splitfilename[0]
    engine_name = splitfilename[1].split("d")[0]
    games_data = []
    games = content.split("; AIF v1.0 Kenneth Regan and Tamal Biswas")[1:]  # Skipping the first split before the first game
    destination = os.path.join("/shared/projects/regan/CSE250A8/CSE702_Embeddings/game_dataset/", tournament_name)
    if not os.path.exists(destination):
        os.mkdir(destination)
    total = len(games)
    train_limit = int(total * 0.6)
    test_limit = train_limit + int(total * 0.2)
    val1_limit = test_limit + int(total * 0.1)
    for index, game in enumerate(games):
        game_dictionary = {}
        pattern = r'\[GameID "(.*?)"\]'
        match = re.search(pattern, game)
        if match:
            gameid = match.group(1)
            gameid = gameid.replace(" ", "")
            gameid = gameid.replace("\n", "")
            gameid = gameid.replace("\r", "")
            gameid = gameid.replace("\t", "")
            gameid = gameid.replace("\\", "")
            gameid = re.sub(r'[;,\'/]', '_', gameid)
        if index < train_limit:
            data_subset = "Train"
        elif index < test_limit:
            data_subset = "Test"
        elif index < val1_limit:
            data_subset = "Validation1"
        else:
            data_subset = "Validation2"
        game_dictionary["data_subset"] = data_subset
        white_elo_pattern = r'\[WhiteElo "(.*?)"\]'
        black_elo_pattern = r'\[BlackElo "(.*?)"\]'
        whiteelo_match = re.search(white_elo_pattern, content)
        blackelo_match = re.search(black_elo_pattern, content)
        if whiteelo_match:
            whiteelo = whiteelo_match.group(1)
            game_dictionary["WhiteElo"] = whiteelo
        if blackelo_match:
            blackelo = blackelo_match.group(1)
            game_dictionary["BlackElo"] = blackelo
        game_dictionary[engine_name] = []
        moves = game.split("[EndMove]")
        engine_game_data = {}
        for move in moves:
            eval_pattern = r'\[Eval "(.*?)"\]'
            eval_match = re.search(eval_pattern, move)
            if eval_match:
                eval = eval_match.group(1)
                engine_game_data["Eval"] = eval
            turn_pattern = r'\[Turn "(.*?)"\]'
            turn_match = re.search(turn_pattern, move)
            if turn_match:
                turn = turn_match.group(1)
                engine_game_data["Turn"] = turn
            move_played_pattern = r'\[MovePlayed "(.*?)"\]'
            move_played_match = re.search(move_played_pattern, move)
            if move_played_match:
                move_played = move_played_match.group(1)
                engine_game_data["MovePlayed"] = move_played
            engine_move_pattern = r'\[EngineMove "(.*?)"\]'
            engine_move_match = re.search(engine_move_pattern, move)
            if engine_move_match:
                engine_move = engine_move_match.group(1)
                engine_game_data["EngineMove"] = engine_move
            fen_pattern = r'\[FEN "(.*?)"\]'
            fen_match = re.search(fen_pattern, move)
            if fen_match:
                fen = fen_match.group(1)
                engine_game_data["FEN"] = fen
            game_dictionary[engine_name].append(engine_game_data)
            engine_game_data = {}
        if(gameid+".npy" in os.listdir(destination)):
            game_dictionary = np.load(gameid+".npy", allow_pickle=True).item()
            game_dictionary[engine_name] = engine_game_data
            np.save(gameid+".npy", game_dictionary)
        else:
            file_path = os.path.join(destination, gameid + ".npy")
            print("Saving to:", file_path)  # Debug print to check the file path
            with open(file_path, "wb") as file:
                np.save(file, game_dictionary)
        game_dictionary = {}

    return

import chess

def extract_positions_from_game(game_data):
    """
    Extracts board positions from a single game's data.

    Parameters:
    - game_data: A dictionary containing the game's information, including a list of moves.

    Returns:
    - A list of tuples, where each tuple contains a move and the corresponding FEN string of the board after the move.
    """
    # Initialize a chess board with the starting position
    board = chess.Board()

    # Placeholder for the results
    positions_after_moves = []

    # Assuming 'moves' is a list of strings representing the moves in standard algebraic notation
    moves = game_data.get('moves', [])

    for move_san in moves:
        try:
            # Parse the move and apply it to the board
            move = board.parse_san(move_san)
            board.push(move)

            # Append the current move and board position (in FEN) to the results list
            positions_after_moves.append((move_san, board.fen()))
        except ValueError as e:
            # Handle invalid moves or notation errors
            print(f"Error processing move '{move_san}': {e}")
            break

    return positions_after_moves


import pandas as pd


def create_data_frame(games_data):
    """
    Converts games data into a pandas DataFrame.

    Parameters:
    - games_data: A list of dictionaries, where each dictionary contains data for a single game.

    Returns:
    - A pandas DataFrame where each row represents a game and each column represents a piece of data about that game.
    """
    # Ensure games_data is a list of dictionaries
    if not all(isinstance(game, dict) for game in games_data):
        raise ValueError("games_data must be a list of dictionaries.")

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(games_data)

    return df
# games_data = [
#     {
#         "title": "Game 1",
#         "player1": "Player A",
#         "player2": "Player B",
#         "result": "1-0",
#         "moves": ["e4", "e5", "Nf3", "Nc6"],
#         # Include other game details as needed
#     },
#     {
#         "title": "Game 2",
#         "player1": "Player C",
#         "player2": "Player D",
#         "result": "0-1",
#         "moves": ["d4", "d5", "c4", "e6"],
#         # Include other game details as needed
#     },
#     # Add more games as needed
# ]

# # Convert the structured games data into a DataFrame
# df_games = create_data_frame(games_data)

# # Display the resulting DataFrame
# print(df_games)

import numpy as np

def average_utility(games_data):
    row_averages = np.mean(games_data[:, 14:], axis=1)

def main():
    # Read the content of the AIF file
    with open("WijkTataBJan2024cat13_SF11d20-30pv64.aif", "r") as file:
        content = file.read()

    # Parse games from the AIF content
    parse_games_from_aif("WijkTataBJan2024cat13_SF11d20-30pv64.aif", content)
    return
if __name__ == "__main__":
    main()