import re


def parse_games_from_aif(content):
    """
    Parses games from the AIF content and returns structured data.

    This example assumes games are separated by a distinct pattern or keyword,
    and each game contains structured data that can be extracted using regular expressions or string manipulation.
    """

    # Placeholder list to store each game's data
    games_data = []

    # Example: Assuming each game starts with a unique identifier like "Game Start"
    # and ends with "Game End" (these would be replaced with actual identifiers from the AIF file)
    game_sections = content.split("Game Start")[1:]  # Skipping the first split before the first game

    for section in game_sections:
        game_info = section.split("Game End")[0]  # Get content up to "Game End"

        # Extract information using regular expressions or string methods
        # This is highly speculative and needs to be adjusted to match the actual content structure

        # Example: Extracting a title assuming it's labeled as "Title: <title>"
        title_match = re.search(r"Title: (.+)", game_info)
        title = title_match.group(1) if title_match else None

        # Continue extracting other details in a similar fashion...

        # Assuming extracting moves, which might be listed line by line or in another format
        moves = []
        for line in game_info.split("\n"):
            if "Move:" in line:  # Speculative example
                moves.append(line.replace("Move: ", "").strip())

        # Append structured game data to the list
        games_data.append({
            "title": title,
            "moves": moves,
            # Include other fields as extracted
        })
    games_data = [
        {
            "title": "Game 1",
            "player1": "Player A",
            "player2": "Player B",
            "result": "1-0",
            "moves": ["e4", "e5", "Nf3", "Nc6"],
            # Include other game details as needed
        },
        {
            "title": "Game 2",
            "player1": "Player C",
            "player2": "Player D",
            "result": "0-1",
            "moves": ["d4", "d5", "c4", "e6"],
            # Include other game details as needed
        },
        # Add more games as needed
    ]

    # Convert the structured games data into a DataFrame
    df_games = create_data_frame(games_data)

    # Display the resulting DataFrame
    print(df_games)


    return games_data


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
games_data = [
    {
        "title": "Game 1",
        "player1": "Player A",
        "player2": "Player B",
        "result": "1-0",
        "moves": ["e4", "e5", "Nf3", "Nc6"],
        # Include other game details as needed
    },
    {
        "title": "Game 2",
        "player1": "Player C",
        "player2": "Player D",
        "result": "0-1",
        "moves": ["d4", "d5", "c4", "e6"],
        # Include other game details as needed
    },
    # Add more games as needed
]

# Convert the structured games data into a DataFrame
df_games = create_data_frame(games_data)

# Display the resulting DataFrame
print(df_games)
