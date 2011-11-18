from gurobipy import *
from clustering import *


class ClusteringLPBuilder:
  def __init__(self):
    pass
  
  def build(self, clustering_problem):
    cp = clustering_problem
    # Create a new model
    m = Model("mip1")

    # Minimize
    m.ModelSense = 1

    # Create variables
    # Connections
    x = {}
    # Facilities
    y = {}
    for f in cp.facilities:
      for c in cp.cities:
        x[(f.id, c.id)] = m.addVar(0.0, 1.0,
                                       cp.ScoreCityAssignment(c, f),
                                       GRB.BINARY,
                                       "x_{"+ str(f.id) + "," + str(c.id) + "}")

      y[f.id] = m.addVar(0.0, 1.0, cp.ScoreOpenFacility(f),
                         GRB.BINARY, "y_"+str(f.id))
    
    # Integrate new variables
    m.update()
    
    # Add constraint: x_{i,j} - y_{i,j} <= 0
    for c in cp.cities:
      s = LinExpr()
      for f in cp.facilities:
        s.addTerms(-1.0, x[f.id, c.id])
      s.addConstant(1.0)
      m.addConstr(s, GRB.EQUAL, 0.0, "City "+str(c.id))

    # Add constraint: x_{i,j} - y_{i,j} <= 0
    for f in cp.facilities:
      for c in cp.cities:
        m.addConstr(LinExpr([1.0, -1.0], [x[f.id, c.id], y[f.id]]),
                            GRB.LESS_EQUAL, 0.0, "")
    
    m.optimize()

    assignments = {}
    for f in cp.facilities:
      for c in cp.cities:
        if x[f.id, c.id].X == 1.0:
          assignments[c] = f
    return ClusteringAnswer(cp, assignments)
#     for v in m.getVars():
#       print v.VarName, v.X, v.Obj
      
#     print 'Obj:', m.ObjVal
    
