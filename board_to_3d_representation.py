# Take a board and turn it into a 3d representation
import numpy as np

def fen_to_onehot(fen):
    """
    Convert a FEN string to a one-hot encoded 8x8x6x2 matrix.
    
    Parameters:
    - fen: A string representing the board in FEN notation.
    
    Returns:
    - A one-hot encoded 8x8x6x2 numpy array.
    """
    # Define the pieces in FEN notation
    pieces = 'pnbrqkPNBRQK'
    
    # Initialize an empty one-hot encoded matrix
    onehot = np.zeros((8, 8, 6, 2))
    
    # Split the FEN string into rows
    rows = fen.split()[0].split('/')
    
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                # Empty squares
                col += int(char)
            else:
                piece_idx = pieces.index(char)
                piece_type = piece_idx % 6
                color = piece_idx // 6
                onehot[i, col, piece_type, color] = 1
                col += 1
    
    return onehot

# Example usage
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
onehot_matrix = fen_to_onehot(fen)

print(onehot_matrix.shape)  # Should be (8, 8, 6, 2)
