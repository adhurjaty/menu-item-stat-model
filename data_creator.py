import numpy as np

class OrderMatrix(object):
    columns = []
    records = {}

    def __init__(self, columns):
        self.columns = columns

    def __str__(self):
        s = '\t'*2 + '\t'.join(name for name in self.columns) + '\n'
        for name, row in self.get_ordered_rows():
            tabs = '\t' + ('\t' if len(name) < 8 else '')
            vals = '\t'.join(map(str, row))
            s += f'{name}{tabs}{vals}\n'
        return s

    def add_record(self, name, values):
        self.records[name] = list(map(float, values))

    def get_ordered_rows(self):
        return [(key, self.records[key]) for key in sorted(self.records.keys())]

    def to_matrix(self):
        mat = [row[:] for _, row in self.get_ordered_rows()]
        return np.array(mat)


def generate_usages(food_matrix):
    pass
    

if __name__ == '__main__':
    filename = 'item_matrix.csv'
    with open(filename, 'r') as f:
        contents = f.read().strip().split('\n')
    ingredients = contents.pop(0).split(',')[1:]

    matrix = OrderMatrix(ingredients)

    for line in contents:
        row = line.split(',')
        matrix.add_record(row[0], row[1:])

    result = matrix.to_matrix()
    print(result)