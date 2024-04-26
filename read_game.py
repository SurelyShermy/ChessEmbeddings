import numpy as np
import pprint

# Replace 'your_file.npy' with the path to your .npy file
#file_path = '../game_dataset/AA1375in2012Exp12/FAI-chT21213_FaroeIslands_2012.11.18_3.4_Johannesen_Hakun_Finnsson_EliW_1-0.npy'
#file_path = '/shared/projects/regan/CSE250A8/CSE702_Embeddings/game_dataset/WchRapid2018c/WchRapid_StPetersburg_2018.12.28_15_Wang_Hao_Cheparinov_Ivan_1_2-1_2.npy'
file_path = './games_dataset/AeroflotOpenAMar2024/AeroflotOpenA2024_MoscowRUS_2024.03.03_1.1_Grischuk_Alexander_Aleksandrov_Aleksej_1-0.npy'
#file_path = '/shared/projects/regan/CSE250A8/CSE702_Embeddings/game_dataset/ChessComTitledTuesdayG4BlitzB05Mar2024rd6/TitledTue5thMarLate_chess.comINT_2024.03.05_6_Spitzl_Vinzent_Wafa_Hamed_0-1.npy'
#file_path = '/shared/projects/regan/CSE250A8/CSE702_Embeddings/game_dataset/WchRapid2017c/WchRapid_Riadh_2017.12.28_15_Alanazy_Mohammed_Salama_Taher_0-1.npy'
#file_path = '/shared/projects/regan/CSE250A8/CSE702_Embeddings/game_dataset/FloripaOpenJan2024rd9/XFloripaChessFestival2024_FlorianopolisBRA_2024.01.27_9_Vieira_JoseRenatoBragaS._Silva_HenriqueBrasilBarrientos_1_2-1_2.npy'
# Load the dictionary from the .npy file
data = np.load(file_path, allow_pickle=True).item()
pp = pprint.PrettyPrinter(indent=4)

# Now, data is the dictionary loaded from the .npy file
pp.pprint(data)

# Example of accessing a value with a key (replace 'key' with your actual key)
# print(data['key'])

