import matplotlib
import matplotlib.pyplot as plt
from itertools import izip
from vector import *

class Point:
  def __init__(self, id, name, location):
    self.id = id
    self.name = name
    self.location = location

  def __hash__(self):
    return self.id

  def __repr__(self):
    return str(self.location)
  
class ClusteringAnswer:
  def __init__(self, clustering_problem, assignments):
    self.clustering_problem = clustering_problem
    self.assignments = assignments
    self.city_assignment = dict([(city.id, facility)
                                 for city, facility in assignments.iteritems()])
    self.facility_assignment = {} 
    for city, facility in assignments.iteritems():
      self.facility_assignment.setdefault(facility, [])
      self.facility_assignment[facility].append(city)                                 

    self.open_facilities = set([facility
                                for _, facility in assignments.iteritems()])

  def CityAssignment(self, city):
    return self.city_assignment[city.id]

  def __repr__(self):
    s = ""
    for f, cs in self.facility_assignment.iteritems():
      s += str(f) + "\n"
      for c in cs:
        score = self.clustering_problem.ScoreCityAssignment(c, f)
        s += "\t" + str(c) + " " + str(score) + "\n" 
    return s
      

  def ScoreAnswer(self):
    "Implements facility location objective"
    cp = self.clustering_problem
    return sum([cp.ScoreCityAssignment(city, facility)
                for city, facility in self.assignments.iteritems()]) +\
           sum([cp.ScoreOpenFacility(facility)
                for facility in self.open_facilities])
  @staticmethod
  def Draw(answers):

    for i,s in enumerate(answers):
      plt.subplot(2, 2, i + 1)
      colors = ["red", "yellow", "green", "blue", "pink", "purple", "orange", "brown", "white"]
      plt.xticks([])
      plt.yticks([])
      plt.xlim([0, 1.0])
      plt.ylim([0, 1.0])
      plt.xlabel("%.3f"%s.ScoreAnswer())
      for c in s.clustering_problem.cities:
        x = [c.location.data[0]]
        y = [c.location.data[1]]
        facility = s.CityAssignment(c)
        n = facility.id % len(colors)
        plt.scatter(x, y, c=colors[n])

      for f in s.clustering_problem.facilities:
        x = [f.location.data[0]]
        y = [f.location.data[1]]
        if f in s.open_facilities:
          n = f.id % len(colors)
          color = colors[n]
        else:
          color = "black"
        plt.scatter(x, y, c=color, marker='s', s = 100)
      
    plt.show()
    
      

class ClusteringProblem:
  """
  A clustering problem consists of cities, facilities to connect to a
  metric cost structure for connection and costs for opening
  facilities.
  """
  def __init__(self, cities, facilities, metric, facility_cost):
    self.facilities = facilities
    self.cities = cities
    self._metric = metric
    self._facility_cost = facility_cost 
    

  def ScoreCityAssignment(self, city, facility):
    return self._metric.distance(city, facility)

  def ScoreOpenFacility(self, facility):
    return self._facility_cost[facility.id]
  
