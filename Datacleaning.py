# %%
import re
import sys
import glob
import os.path

# %%
# Cleaning Dyslexia data VIRKER

list_of_files = glob.glob('Data/D_data/*.txt')

for file_name in list_of_files:
    print(file_name)  # Dette kan kommenteres ud hvis vi har lyst

    # This needs to be done *inside the loop*
    f = open(file_name, 'r', encoding='utf8', errors='ignore')
    lst = []
    for line in f:
        line.strip()
        line = re.sub(
            r"\(\D*\d?\d{4}(?:, s.? [0-9]+.?.?[0-9].?)?(([;])\D*\d{4})*\)|\(([a-zA-Z]+\d\D*\d{4}\))", "", f.read())
        line = re.sub(r'”[^"]+”', "", line)
        line = re.sub(r'"[^"]+"', "", line)
        lst.append(line)
    f.close()

    f = open(os.path.join('Data/D_data/Final_D_data',
                          os.path.basename(file_name)), 'w')

    for line in lst:
        f.write(line)
    f.close()

# %%
# Cleaning Non-Dyslexia data VIRKER

list_of_files = glob.glob('Data/ND_data/*.txt')

for file_name in list_of_files:
    print(file_name)  # Dette kan kommenteres ud hvis vi har lyst

    # This needs to be done *inside the loop*
    f = open(file_name, 'r', encoding='utf8', errors='ignore')
    lst = []
    for line in f:
        line.strip()
        line = re.sub(
            r"\(\D*\d?\d{4}(?:, s.? [0-9]+.?.?[0-9].?)?(([;])\D*\d{4})*\)|\(([a-zA-Z]+\d\D*\d{4}\))", "", f.read())
        line = re.sub(r'”[^"]+”', "", line)
        line = re.sub(r'"[^"]+"', "", line)
        lst.append(line)
    f.close()

    f = open(os.path.join('Data/ND_data/Final_ND_data',
                          os.path.basename(file_name)), 'w')

    for line in lst:
        f.write(line)
    f.close()
# %%
