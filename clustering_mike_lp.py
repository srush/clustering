from gurobipy import *
from clustering import *


class ClusteringMikeLPBuilder:
  def __init__(self):
    pass
  
  def build(self, stats):
    SPLIT_EPS = 100
    # Create a new model
    m = Model("mip1")

    # Minimize
    m.ModelSense = -1

    # Create variables
    alpha = {}
    beta = {}
    gamma = {}

    for w1 in stats.words:
      for c in stats.classes:
        alpha[c, w1] =m.addVar(0.0, 1.0,
                               0.0,
                               GRB.CONTINUOUS,
                               "alpha_{"+ str(c) + ", " + str(w1) + "}")
        beta[c, w1] = m.addVar(0.0, 1.0,
                               0.0,
                               GRB.CONTINUOUS,
                               "beta_{"+ str(c) + ", " + str(w1) + "}")
    
    for w1, w2 in stats.f_pair.iterkeys():
      counts =  stats.f_pair[w1,w2]
      for c in stats.classes:
        gamma[(w1, w2, c)] = m.addVar(0.0, 1.0,
                                      counts,
                                      GRB.CONTINUOUS,
                                      "gamma_{"+ str(w1) + "," + str(w2) + "," + str(c) + "}")
    
    # Integrate new variables
    m.update()

    # Alpha and Beta are distributions
    for w in stats.words:
      s = LinExpr()   
      for c in stats.classes:
        s.addTerms(1.0, alpha[c, w])
      m.addConstr(s, GRB.EQUAL, 1.0, "alpha "+str(w))

      s = LinExpr()   
      for c in stats.classes:
        s.addTerms(1.0, beta[c, w])
      m.addConstr(s, GRB.EQUAL, 1.0, "beta "+str(w))

    # Balance
    instance_split = int(stats.num_word_instances / float(len(stats.classes)))
    print instance_split
    for c in stats.classes:
      s = LinExpr()   
      for w in stats.words:
        s.addTerms(stats.f_one.get(w,0.0), alpha[c, w])
      m.addConstr(s, GRB.GREATER_EQUAL,
                  instance_split - SPLIT_EPS,
                  "balance alpha "+str(c))

      m.addConstr(s, GRB.LESS_EQUAL,
                  instance_split + SPLIT_EPS,
                  "balance alpha "+str(c))

      s = LinExpr()   
      for w in stats.words:
        s.addTerms(stats.f_two.get(w, 0.0), beta[c, w])
      m.addConstr(s, GRB.GREATER_EQUAL,
                  instance_split - SPLIT_EPS,
                  "balance beta "+str(c))

      m.addConstr(s, GRB.LESS_EQUAL,
                  instance_split + SPLIT_EPS,
                  "balance beta "+str(c))


    # Gamma linearization constraints.
    for (w1, w2, c), g in gamma.iteritems():
      m.addConstr(LinExpr([1.0, -1.0], [g, alpha[c, w1]]),
                  GRB.LESS_EQUAL, 0.0, "gamma lin alpha")

      m.addConstr(LinExpr([1.0, -1.0], [g, beta[c, w2]]),
                  GRB.LESS_EQUAL, 0.0, "gamma lin beta")

      m.addConstr(LinExpr([1.0, -1.0, -1.0], [g, alpha[c, w1], beta[c, w2]]),
                  GRB.GREATER_EQUAL, -1.0, "gamma lin both")

      
    m.optimize()
    print m
    assignments = {}
    for w in stats.words:
      for c in stats.classes:
        if alpha[c, w].X == 1.0:
          assignments.setdefault(c, [])
          assignments[c].append(w)
    for c in stats.classes:
      print c 
      for w in assignments.get(c, []):
        print "\t", w     
    m.write("model.lp")
    
