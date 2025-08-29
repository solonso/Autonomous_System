#!/usr/bin/env python3
from PIL import Image
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python3 check_map.py <map_file>")
    sys.exit(1)

map_file = sys.argv[1]
img = Image.open(map_file).convert("L")
gridmap = np.array(img.getdata()).reshape(img.size[0], img.size[1]) / 255
gridmap[gridmap > 0.5] = 1
gridmap[gridmap <= 0.5] = 0
gridmap = (gridmap * -1) + 1

print(f"Map shape: {gridmap.shape}")
print(f"Free spaces: {np.sum(gridmap == 0)}")
print(f"Obstacles: {np.sum(gridmap == 1)}")

free_positions = np.where(gridmap == 0)
print("Sample free positions:")
for i in range(min(10, len(free_positions[0]))):
    print(f"  ({free_positions[0][i]}, {free_positions[1][i]})")

# Check if specific positions are free
test_positions = [(10, 10), (50, 50), (5, 5), (20, 20)]
print("\nChecking test positions:")
for pos in test_positions:
    if 0 <= pos[0] < gridmap.shape[0] and 0 <= pos[1] < gridmap.shape[1]:
        status = "FREE" if gridmap[pos[0], pos[1]] == 0 else "OBSTACLE"
        print(f"  {pos}: {status}")
    else:
        print(f"  {pos}: OUT OF BOUNDS")
