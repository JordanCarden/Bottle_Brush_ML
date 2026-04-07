import numpy as np
import pandas as pd

# Read the data

header = []
f1 = open("single_polymer.pdb", "r")
with open("single_polymer.pdb", "r") as file:
    lines = file.readlines()

x = None
for line in lines: 
    if line.startswith('ATOM'):
        atom_number = int(line[6:11].strip())
        x = atom_number
nn=x
mm=nn+1
for i in range(0):
    header.append(f1.readline())
print(header)


a = []  # a list from 0 to 5000, number of lines need to read
datax = []
datay = []
dataz = []
for i in range(1, mm):
    a.append(i)
for n, line in enumerate(f1):
    if n in a:
        t = line.strip()
        x = float(t[29:37])
        y = float(t[38:45])
        z = float(t[46:53])
        datax.append(x)
        datay.append(y)
        dataz.append(z)

datax=np.array(datax)
datay=np.array(datay)
dataz=np.array(dataz)
print(datax)
print(datay)
print(dataz)

object_coords = np.hstack((datax.reshape(nn,1), datay.reshape(nn,1), dataz.reshape(nn,1)))
# Parameters for duplication
num_duplicates_y = 5
num_duplicates_x = 4
angle = 0
stepside_y = 45
stepside_x = 45

# Calculate total duplicates
total_duplicates = num_duplicates_x * num_duplicates_y

# Create an empty array for storing duplicated coordinates
duplicated_coords = np.empty((total_duplicates, object_coords.shape[0], object_coords.shape[1]))

# Loop through each duplication step in both x and y directions
count = 0
for i in range(num_duplicates_x):
    for j in range(num_duplicates_y):
        angle_rad = angle * np.pi / 180
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])

        # Apply rotation and translation
        translated_coords = np.dot(object_coords, rotation_matrix) + [stepside_x * i, stepside_y * j, 0]
        duplicated_coords[count] = translated_coords
        count += 1

# Reshape duplicated_coords for saving
reshaped_coords = duplicated_coords.reshape(-1, 3)

# Save the duplicated coordinates to a text file
np.savetxt('20_coordinate.txt', reshaped_coords, fmt='%7.3f %7.3f %7.3f')
