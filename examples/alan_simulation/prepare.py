import numpy as onp


with open("alanine_solvated_raw.lmpdat", "r") as f:
    raw_lines = f.readlines()

# Correct to atomic nummbers
lookup = {1: "6", 2: "1", 3: "7", 4: "8"}

lines = []
header = True
masses = True
for line in raw_lines:
    if line.startswith("Masses"):
        header = False
        lines.append(line)
        continue

    if line.startswith("Atoms"):
        masses = False
        lines.append(line)
        continue

    if line.startswith("Bonds"): break

    if header:
        lines.append(line)
        continue

    entries = line.split()

    if len(entries) < 1:
        continue

    print(entries)

    if masses: 
        entries[0] = lookup[int(entries[0])] 
    else:
        entries.pop(1)
        entries.pop(2)
        entries[1] = lookup[int(entries[1])]

    lines.append(" ".join(entries))

lines = [f"{l}\n" for l in lines]

with open("alanine_solvated.lmpdat", "w") as f:
    f.writelines(lines)

