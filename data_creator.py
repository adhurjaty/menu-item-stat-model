import numpy as np
import random

class OrderMatrix(object):
    columns = []
    records = {}

    @staticmethod
    def from_mat(matrix, columns, ordered_row_names):
        om = OrderMatrix(columns)
        for i, row in enumerate(matrix):
            om.add_record(ordered_row_names[i], row)
        return om

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

    def get_ordered_names(self):
        return sorted(self.records.keys())

    def to_matrix(self):
        mat = [row[:] for _, row in self.get_ordered_rows()]
        return np.array(mat)


def random_usage(mean):
    return np.random.normal(mean, mean / 5)


def random_ingredients(ingredients):
    amount = random.randint(0, 10)
    return np.array([amount] + list(map(lambda el: random_usage(el) * amount, ingredients)))


def generate_usages(food_matrix):
    in_mat = food_matrix.to_matrix()
    out_mat = np.apply_along_axis(random_ingredients, 1, in_mat)
    order_amounts = out_mat[:, 0]
    ingredient_amounts = out_mat[:, 1:]
    return (order_amounts, OrderMatrix.from_mat(ingredient_amounts, food_matrix.columns, 
        food_matrix.get_ordered_names()))

def create_food_matrix(filename):
    with open(filename, 'r') as f:
        contents = f.read().strip().split('\n')
    ingredients = contents.pop(0).split(',')[1:]

    matrix = OrderMatrix(ingredients)

    for line in contents:
        row = line.split(',')
        matrix.add_record(row[0], row[1:])

    return matrix

if __name__ == '__main__':
    matrix = create_food_matrix('item_matrix.csv')    
    usages = generate_usages(matrix)
    print(usages)