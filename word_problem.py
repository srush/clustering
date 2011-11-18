from clustering_lp import *
from clustering_approx import *
from words import *
import sys

inp = sys.argv[1]
collection = ReadBrown(open(inp))
cities = [Point(i,"", w) for i, w in enumerate(collection.GetMostCommon(1000))]
facilities = [Point(i,"", w) for i, w in enumerate(collection.GetMostCommon(100))]

cp = ClusteringProblem(cities, facilities, JacardDistance(), [1.0] * (100))

cp_approx = ClusteringApprox()
answer2 = cp_approx.build(cp)
print "Score (Approx):", answer2.ScoreAnswer()
cp_lp = ClusteringLPBuilder()
answer1 = cp_lp.build(cp)
print "Score (ILP):", answer1.ScoreAnswer()
print str(answer2)
# ClusteringAnswer.Draw([answer1, answer2])

