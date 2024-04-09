from functions import parse_games_from_aif
import os

def process_aif_files(directory):
# Iterate over all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file has a .aif extension
        if filename.endswith('.aif'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            
            # Read the content of the .aif file
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Call the parse_games_from_aif function with the filename and content
            parse_games_from_aif(filename, content)

if __name__ == "__main__":
    # Specify the directory containing the .aif files
    directory = '/shared/projects/regan/Chess/CSE702/AIF'
    process_aif_files(directory)