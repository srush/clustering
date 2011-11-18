from vector import *
from random import *
INF = 1e20
ADVANCE = 0
STAY = 1

    
class DataPoint:
  def __init__(self, time_step, point):
    self.time_step = time_step
    self.point = point

  def __repr__(self):
    return "%s %s"%(self.time_step, self.point)
class TimeSeries:
  def __init__(self, data_points):
    self.data_points = data_points
    
  def __len__(self): return len(self.data_points)
  def __getitem__(self, ind): return self.data_points[ind]
    
class Phoneme:
  def __init__(self, name, id):
    self.name = name
    self.id = id
  def __key__(self): return self.id

  def __repr__(self):
    return "%s %d"%(self.name,self.id) 

class Prediction:
  def __init__(self, phoneme_set, centers):
    self.set = phoneme_set
    self.centers = centers

  def __repr__(self):
    return "".join([str(phoneme) + " " + str(vec) for phoneme, vec in self.centers.iteritems()])

  def for_phoneme(self, phoneme):
    return self.centers[phoneme]

  @staticmethod
  def random(phoneme_set, dim):
    phoneme_centers = {}
    for phoneme in phoneme_set:
      phoneme_centers[phoneme] = Vector([random() * 0.001 for i in range(dim)])
    return Prediction(phoneme_set, phoneme_centers)

  @staticmethod
  def maximize_likelihood(phoneme_set, classification):
    phoneme_centers = {}
    for phoneme, points in classification.iteritems():
      phoneme_centers[phoneme] = Vector.average(list([p.point for p in points]))
    return Prediction(phoneme_set, phoneme_centers)

class Sentence:
  def __init__(self, phoneme_instances):
    self.instances = phoneme_instances
    
  def __getitem__(self, ind): return self.instances[ind]
  def __len__(self): return len(self.instances)
  
class ViterbiAlign:
  def __init__(self, phoneme_set):
    self.phoneme_set = phoneme_set
  
  def cost(self, phoneme, point):
    return phoneme.point.difference(point).l2()
        
  def run(self, prediction, sentence, timeseries):
    m = len(sentence)
    n = len(timeseries)
    pi = {}
    M = {}
    for i in range(m):
      M[i, 0] = INF
    for j in range(n):
      M[-1, j] = INF
    M[0, 0] = 0.0
        
    for j in range(1, n):
      for i in range(0, m):
        a = M[i, j-1]
        b = M[i-1, j- 1]
        if a < b:
          M[i,j] =  a
          pi[i,j] = STAY
        else:
          M[i,j] = b
          pi[i,j] = ADVANCE

        # Generate point i from phoneme instance at j
        center = prediction.for_phoneme(sentence[i])
        gen_cost = self.cost(timeseries[j], center)
        M[i, j] += gen_cost
    
    # Find where each of the phoneme instances was placed
    phoneme_points = {}
    last = (m -1, n -1)
    cur = last
    map = {}
    while True:
      (i,j) = cur
      map[j] = i
      phoneme = sentence[i]
      phoneme_points.setdefault(phoneme, set())
      phoneme_points[phoneme].add(timeseries[j])

      if cur == (0,0): break
      if pi[cur] == STAY:
        cur = (i, j - 1)
      else:
        cur = (i - 1, j - 1)
        
    return M[last], map, phoneme_points
  
class SemiMarkov:
  def __init__(self, phoneme_set):
    self.phoneme_set = phoneme_set

  def set_penalty(self, penalties):
    self.penalty = penalties
  
  def initialize(self, state, start):
    ""
    return self.merge(state, start, start, None)

  def score(self, cluster):
    ""
    return cluster[1]

  def merge(self, state, start_time, end_time, cluster):
    ""
    min_score = INF 
    for a in self.possible_points:
      local_score = 0.0
      for t in range(start_time, end_time + 1):
        local_score += a.point.difference(self.time_series[t].point).l1() + self.penalty.get((state, a.time_step),0.0)
      if local_score < min_score:
        min_score = local_score
        min_point = a
    return (min_point, min_score)

  def run(self, sentence, timeseries):
    # Everything can be a center
    self.possible_points = [timeseries[i] for i in range(len(timeseries)) ]
    self.time_series = timeseries
    
    m = len(sentence)
    n = len(timeseries)
    C = {}
    
    # The when moving to state i at time j what was the best previous step?
    pi = {}

    # The best score moving to state i at time j.
    M = {}
    
    for i in range(m):
      M[i, 0] = INF
    for j in range(n):
      M[0, j] = INF
    M[0, 0] = 0.0
    pi[0,0] = -1
    for j in range(0, n + 1):
      for i in range(0, m + 1):
        # Initialize new cluster
        if i != m and j != n:
          C[i, j, j] = self.initialize(i,j)
          for j_last in range(j):
            C[i, j_last, j] = self.merge(i,j_last,j, C[i, j_last, j-1])
        if j == n:
          for j_last in range(j):
            if j_last < i: continue
            C[i, j_last, j] = C[i, j_last, j - 1]
        if i != 0 and j != 0 and i <= j:
          M[i,j] = INF
          for j_last in range(j):
            if j_last < i-1: continue
            score = M[i-1, j_last] + self.score(C[i-1, j_last, j-1])
            if score < M[i,j]:
              # The best way to start i at j is if i-1 starts at j_last
              M[i, j] = score

              pi[i, j] = j_last

          assert( (i,j) in pi ) 

    # Find where each of the phoneme instances where placed
    phoneme_points = {}
    last = (m, n )
    cur = last
    map = {}
    centers = {}
    while True:
      (i,j) = cur
      phoneme = sentence[i - 1]
      phoneme_points.setdefault(phoneme, set())
      first = pi[cur]
      for t in range(first, j):
        map[j] = i - 1
        phoneme_points[phoneme].add(timeseries[t])
      centers[i - 1] = C[i - 1, first, j][0]
      if i - 1  == 0: break
      cur = (i - 1, first)

    return M[last], centers, map, phoneme_points
              
def run_subgradient(phonemes, sent, data):
  sm = SemiMarkov(phonemes)

  # Multipliers all start at 0
  penalties = {} 

  for round_num in range(1, 1000):
    sm.set_penalty(penalties)
    score, centers, m, points = sm.run(sent, data)
    # Update
    update = 1.0 / float(round_num)
    for i in range(len(sent)):
      # Lower my score.
      penalties.setdefault((i, centers[i].time_step), 0.0)
      penalties[i, centers[i].time_step] += update
      
      # Raise the score of other centers for the same phoneme.
      count = 0
      for i_other in range(len(sent)):
        if sent[i].id == sent[i_other].id:
          count += 1
      for i_other in range(len(sent)):
        if sent[i].id == sent[i_other].id:
          penalties.setdefault((i, centers[i_other].time_step), 0.0)
          penalties[i, centers[i_other].time_step] -= update / float(count)
    print score, penalties

