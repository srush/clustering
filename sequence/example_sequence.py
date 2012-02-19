import sys
import numpy as np
import random
sys.path.append(".")
random.seed(2)
from sequence import *
from vector import *
phonemes = [Phoneme("a", 1), Phoneme("b", 2), Phoneme("c", 3)]
data = TimeSeries([DataPoint(1, np.array([0.0, 1.0])),
                   DataPoint(2, np.array([1.0, 1.0])),
                   DataPoint(3, np.array([0.0, 0.0])),
                   DataPoint(4, np.array([0.0, 0.1]))])

sent = Sentence([phonemes[0], phonemes[1], phonemes[2]])

prediction = Prediction.random(phonemes, 2)

print prediction
va = ViterbiAlign(phonemes)

for i in range(50):
  score, m, points = va.run(prediction, sent, data)
  prediction = Prediction.maximize_likelihood(phonemes, points)
  print score, points, prediction
#draw_points(points)
