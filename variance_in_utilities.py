import numpy as np
import matplotlib.pyplot as plt

# Replace 'path/to/your/file.txt' with the actual path to your file
file_path = 'eval_file_2.txt'

# Step 1: Read the file
with open(file_path, 'r') as file:
    numbers = np.array([np.abs(int(line.strip())) for line in file])

upper_limit=250

mean = np.mean(numbers)
std = np.std(numbers)
threshold=3

numbers = numbers[np.abs(numbers) < upper_limit]
numbers = numbers[np.abs(numbers) != 0 ]
#is_not_outlier = (np.abs(numbers - mean) < threshold * std)
#cleaned_numbers = numbers[is_not_outlier]
#cleaned_numbers = cleaned_numbers[np.abs(cleaned_numbers) < upper_limit]
#cleaned_numbers = cleaned_numbers[np.abs(cleaned_numbers) !=0]
#mean = np.mean(cleaned_numbers)
#std = np.std(cleaned_numbers)
#print(mean, std)

plt.hist(numbers, bins=300)
plt.title('Cleaned Distribution')
plt.xlabel('Number')
plt.ylabel('Frequency')
plt.show()
# Calculate the consecutive differences
consecutive_diff = np.diff(numbers)

plt.hist(consecutive_diff, bins=300)
plt.title('diff Distribution')
plt.xlabel('Number')
plt.ylabel('Frequency')
plt.show()