from  clustering_approx import *




cities = [ Point(i, "", point)
           for i, point in enumerate([(0,0), (20,1), (1, 7), (100, 101)])]
facilities = [ Point(i, "", point)
               for i, point in enumerate([(2,2), (100, 100)])]

cp = ClusteringProblem(cities, facilities, EuclideanDistance(), [1.0,1.0])


cp_approx = ClusteringApprox()

answer = cp_approx.build(cp)
print answer.ScoreAnswer()


