from clustering_mike_lp import *
from words import *
import sys

inp = sys.argv[1]
collection = ReadBrownWordStats(open(inp), 64)

cp_lp = ClusteringMikeLPBuilder()
cp_lp.build(collection)
# ClusteringAnswer.Draw([answer1, answer2])

