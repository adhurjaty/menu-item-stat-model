# Usage Predictor

Statistical model to get average ingredient usages for variable ingredient menu items. 

## Problem:

The customer needs to estimate the cost of ingredients for each menu item. The customer's menu items are like that of Chipotle or Subway - the restaurant patrons can choose variable amounts of most ingredients.

## Proposed solution:

Create a statistical model mapping menu item to ingredients with proportions. Model should be linear where
```
mW = g
```
where m is menu items (1 x n_menu_items)
W is weights (n_menu_items x n_ingredients)
g is ingredients (1 x n_ingredients)

Note I am omitting a bias term. Bias would correspond to waste or some fixed usage per ingredient, so may add that in the future - want a simple model initially.

## Training the model:

Model input: menu item sales

Model output: ingredient usages

Cost function: RMSE

Optimization algorithm: Gradient descent


## Data generation:

data_creator.py generates food usage data. Gives a set of examples of daily item ordering to daily ingredient usage.

regression.py contains the Tensorflow model and trains it.


