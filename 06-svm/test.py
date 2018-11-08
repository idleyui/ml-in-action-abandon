
import random


l = [0] * 1000 + [1] * 1000
random.shuffle(l)

for i in range(len(l)):
    l[i] = 1 if l[i] == 0 else -1

print(len(l))
