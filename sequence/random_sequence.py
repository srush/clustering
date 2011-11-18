import sys
sys.path.append(".")

from sequence import *
from vector import *
phonemes = [Phoneme("", i)  for i in range(10)]

def point(x_hat, y_hat): 
  x = normalvariate(x_hat, 0.2)
  y = normalvariate(y_hat, 0.2)
  return Vector([x,y])

true_centers = [ point(0,0) for i in range(len(phonemes))] 

sent_build = []
data_build = []
time = 1
for i in range(50):
  p = choice(phonemes)
  center = true_centers[p.id]
  duration = int(normalvariate(10, .5))
  sent_build.append(p)
  for d in range(duration):
    next = point(center[0], center[1])
    data_build.append(DataPoint(time, next))
    time += 1
sent = Sentence(sent_build)

data = TimeSeries(data_build)


prediction = Prediction.random(phonemes, 2)

va = ViterbiAlign(phonemes)

for i in range(10):
  score, m, points = va.run(prediction, sent, data)
  prediction = Prediction.maximize_likelihood(phonemes, points)
  print score, prediction
#draw_points(points)
