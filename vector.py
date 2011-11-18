from itertools import *

class Vector:
  def __init__(self, point_array):
    self.data = point_array

  def __getitem__(self, ind): return self.data[ind]

  def difference(self, p2):
    return Vector([d1 - d2 for d1, d2 in izip(self.data, p2.data)])

  def divide(self, div):
    return Vector([d / div for d in self.data])

  def l2(self):
    return sum([d1* d1 for d1  in self.data])

  def l1(self):
    return sum([abs(d1) for d1  in self.data])

  def __repr__(self):
    return str(self.data)

  @staticmethod
  def average(vectors):
    n = float(len(vectors))
    dim = len(vectors[0].data)
    return Vector([ sum([ v.data[d] for v in vectors]) / n for d in range(dim)])

class Metric:
  def __init__(self):
    pass
  
  def distance(self, point1, point2):
    pass

class EuclideanDistance(Metric):
  def distance(self, point1, point2):
    return point1.location.difference(point2.location).l2()

class L1Distance(Metric):
  def distance(self, point1, point2):
    return point1.location.difference(point2.location).l1()
