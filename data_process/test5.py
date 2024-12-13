with open('output4.txt', 'r') as file:
    lines = file.readlines()

with open('output5.txt', 'w') as output_file:
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            third_num = int(parts[2]) + 1
            parts[2] = str(third_num)
            updated_line = ' '.join(parts)
            output_file.write(updated_line + '\n')
        else:
            output_file.write(line)
