# Open the text file for reading
with open('D:/Users/ddd/AIC/AIC22/test/S02/c009/gt/gt.txt', 'r') as file:
    lines = file.readlines()
# 006 [1694, 1755, 1907, 4224, 4822, 4842, 4875, 4897, 5030, 5032, 5033]
# 007 [527, 2249, 3278, 3282, 4120, 4167]
# 008 [884, 888, 1662, 2684, 2685, 2689, 3251]
# 009 []
# Initialize variables
prev_num = None
jump_positions = []

# Iterate through each line in the file
for idx, line in enumerate(lines):
    # Extract the first number from the line
    num = int(line.split(',')[0])  # Extracting the first number from each line

    # Check for jump in number sequence
    if prev_num is not None and num - prev_num > 1:
        jump_positions.append(idx)

    prev_num = num

# Print the positions where the number sequence jumps
print("Jump positions:", jump_positions)

