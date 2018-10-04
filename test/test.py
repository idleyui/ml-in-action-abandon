from collections import Counter
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

# dic
# loop dic
# https://stackoverflow.com/questions/3294889/iterating-over-dictionaries-using-for-loops
d = {'x': 1, 'y': 2, 'z': 3}
for key, value in d.items():
    print(key, ',', value)

# count list to dic
# https://stackoverflow.com/questions/3496518/python-using-a-dictionary-to-count-the-items-in-a-list
l = [1, 2, 1, 2, 3, 3, 3]
del (l[0])
print(Counter(l))
print(Counter(l).most_common(1)[0])
print(l)
print(list(set(sorted(l))))

label = 'hight'
my_tree = {'label': label, 'left': {}, 'right': {}, 'bound': 0}
my_tree['left'] = {'label': label, 'left': {}, 'right': {}, 'bound': 0}
my_tree['right'] = {'label': label, 'left': {}, 'right': {}, 'bound': 0}
print(my_tree)

a = 2.23243
print("%.2f" % a)
