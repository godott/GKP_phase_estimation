import os 
import re

prob = 0.0

for my_file in os.listdir("."):

    match = re.compile(".*(prob[0-9.]+).*").search(my_file)

    print(match.group(1))
