from  clustering_lp import *
from random_example import *
from  clustering_approx import *

cities = RandomPoints(1000, 0.9, 0.9, 0) + \
         RandomPoints(1000, 0.1, 0.1, 1000) + \
         RandomPoints(1000, 0.9, 0.1, 2000) + \
         RandomPoints(1000, 0.1, 0.9, 3000)
facilities = RandomUniformPoints(100, 0)

cp = ClusteringProblem(cities, facilities, L1Distance(), [1.0] * (100))

cp_approx = ClusteringApprox()
answer2 = cp_approx.build(cp)
print "Score (Approx):", answer2.ScoreAnswer()

cp_lp = ClusteringLPBuilder()
answer1 = cp_lp.build(cp)
print "Score (ILP):", answer1.ScoreAnswer()

ClusteringAnswer.Draw([answer1, answer2])

