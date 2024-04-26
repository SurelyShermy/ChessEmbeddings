import numpy as np
import matplotlib.pyplot as plt

def get_custom_utility(content):

    # Find the start and end of the utility values matrix
    start_index = content.find('--------------------------------------------------------------------------------------------------------------------')
    end_index = content.find('================================================================================================================')

    # Extract the utility values matrix
    utility_matrix = content[start_index:end_index].strip().split('\n')[3:]

    # Extract the moves
    moves = [line.split()[0] for line in utility_matrix]

    # Extract the utility values
    matrix = []
    int_evals = []
    for line in utility_matrix:
        parts = line.split()
        evaluations = parts[1:]  # Skip the move name
        #log5+abs(util)
        for e in evaluations:
            e = e.replace('+', '')
            if e.endswith('X'):
                e = e.replace('X', '')
                int_evals.append(np.log(int(e)*10))
            elif e.endswith('x'):
                print(e)
                e = e.replace('x', '')
                print(e)
                int_evals.append(-np.log(5+int(e)*10))
            elif "-M" in e:
                int_evals.append(-np.log(5+10000))
            elif "M" in e:
                int_evals.append(np.log(10000+5))
            else:
                int_evals.append(np.sign(int(e))*np.log(5+np.abs(int(e))))
        matrix.append(int_evals)
        int_evals = []

    # Convert to a numpy array
    matrix = np.array(matrix)
    # Calculate mean and variance for each column
    means = np.mean(matrix, axis=0)

    variances = np.var(matrix, axis=0)
    # Create a range for the x-axis based on the number of columns
    x_axis = range(1, means.shape[0]+1)

    # Plotting
    plt.figure(figsize=(14, 7))

    # Plot means
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(x_axis, means, marker='o', linestyle='-', color='b')
    plt.title('Mean of Evaluations by Column')
    plt.xlabel('Column Index (Depth)')
    plt.ylabel('Mean Evaluation')
    plt.grid(True)

    # Plot variances
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(x_axis, variances, marker='o', linestyle='-', color='r')
    plt.title('Variance of Evaluations by Column')
    plt.xlabel('Column Index (Depth)')
    plt.ylabel('Variance Evaluation')
    plt.grid(True)

    # Show the plot
    plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area.
    plt.show()
    return
    # print("\nBest Move:", best_move)
    # print("\nAverage Utility Value of Best Move:", average_utilities[max_utility_index])
def main():
    # Read the content of the utility table file
    with open('variance_mean_extraction_example.aif', 'r') as file:
        content = file.read()

    # Get the custom utility values
    get_custom_utility(content)
    return
if __name__ == "__main__":
    main()