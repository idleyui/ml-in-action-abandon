# python 3.6.5
f = open('table.txt', 'w')
for i in range(9):
    for j in range(i + 1):
        f.write("%d*%d=%d\t" % (i + 1, j + 1, (i + 1) * (j + 1)))
    f.write('\n')
f.close()
