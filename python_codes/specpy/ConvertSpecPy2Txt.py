import sys

# Ensure there are enough arguments passed
if len(sys.argv) < 3:
    print("Usage: python ConvertSpecPy2Txt.py input_filename multiplication_factor")
    sys.exit(1)

input_filename = sys.argv[1]
multiplication_factor = float(sys.argv[2])
outfname_digit = int(multiplication_factor)
output_filename_tail = f"_{outfname_digit}Bq_converted.mac"
output_filename = input_filename.replace('.txt', output_filename_tail)

# read the input file
with open(input_filename, 'r') as f:
    lines = f.readlines()

# find the starting line of the data
for i, line in enumerate(lines):
    if line.strip() == "# Header End: 17 lines":
        start_index = i + 1
        break

# read the data and calculate the sum of column[1] for normalization
data = []
sum_val = 0
for line in lines[start_index:]:
    columns = line.split(';')
    value = float(columns[1])
    sum_val += value
    data.append([float(columns[0]) * 0.001, value])  # keV to MeV (GATE)

# Normalize column[1] by dividing with the summed value and multiply with multiplication_factor
for line in data:
    line[1] = (line[1] / sum_val) * multiplication_factor

# write the new file
with open(output_filename, 'w') as f:
    for line in data:
        f.write(f"/gate/source/xraygun/gps/histpoint  {line[0]:.6f}\t{line[1]:.6f}\n")

