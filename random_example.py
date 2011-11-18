from  clustering import *
from random import *

def RandomPoints(n, x_mean, y_mean, s):
  points = []
  for i in range(n):
    x = normalvariate(x_mean, 0.2)
    y = normalvariate(y_mean, 0.2)
    points.append(Point(s + i, "", Vector((x, y))))
  return points

def RandomUniformPoints(n, s):
  points = []
  for i in range(n):
    x = random()
    y = random()
    points.append(Point(s + i, "", Vector((x, y))))
  return points
