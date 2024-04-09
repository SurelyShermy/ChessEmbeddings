import numpy as np

def get_custom_utility(content):

    # Find the start and end of the utility values matrix
    start_index = content.find('--------------------------------------------------------------------------------------------------------------------------')
    end_index = content.find('==========================================================================================================================')

    # Extract the utility values matrix
    utility_matrix = content[start_index:end_index].strip().split('\n')[3:]

    # Extract the moves
    moves = [line.split()[0] for line in utility_matrix]

    # Extract the utility values
    utility_values = [line.split()[1:] for line in utility_matrix]

    # Find the maximum number of columns in the utility values matrix
    max_columns = max(len(row) for row in utility_values)

    # Pad the rows with missing values using a default value (e.g., 0)
    default_value = 0
    padded_utility_values = [row + [default_value] * (max_columns - len(row)) for row in utility_values]

    # Convert utility values to NumPy array
    utility_array = np.array(padded_utility_values, dtype=int)

    # # Print the extracted data
    # print("Moves:")
    # print(moves)
    # print("\nUtility Values Matrix:")
    # print(utility_array)

    # print(len(utility_array),len(moves))

    # Calculate the average utility value for each move from depth 10 onwards
    average_utilities = np.mean(utility_array[:,10:], axis=1)
    # print("\nAverage Utility Values:")
    # print(average_utilities)
    # print(len(average_utilities))

    # # Calculate the variance of the utility values as a function of depth
    # variances = np.var(utility_array, axis=0)
    # print("\nVariances of Utility Values:")
    # print(variances)
    # #plot this variance
    # import matplotlib.pyplot as plt
    # plt.plot(variances)
    # plt.xlabel('Depth')
    # plt.ylabel('Variance')
    # plt.title('Variance of Utility Values as a Function of Depth')
    # plt.show()


    # find the move which corresponds the max of average_utilities
    max_utility = np.argmax(average_utilities)
    # best_move = moves[max_utility_index]
    return max_utility
    # print("\nBest Move:", best_move)
    # print("\nAverage Utility Value of Best Move:", average_utilities[max_utility_index])
