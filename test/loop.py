def add(x, y):
    b = x + y
    c = b
    return c


if __name__ == '__main__':
    ls = []
    for i in range(100):
        ls.append(add(i, i))
        ls.append(i)
        ls.append(i)
    print(ls)
