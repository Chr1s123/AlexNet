def rd_csv(filename):
    with open(filename, 'r') as f:
        str = f.readlines()[1:]
    a = [l.rstrip().split(',') for l in str]
    return dict(((name, label) for name, label in a))
