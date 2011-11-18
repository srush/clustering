import sys
sys.path.append(".")

from sequence import *
from vector import *
phonemes = [Phoneme("a", 1), Phoneme("b", 2), Phoneme("c", 3)]
data = TimeSeries([DataPoint(1, Vector([0.0, 1.0])),
                   DataPoint(2, Vector([1.0, 1.0])),
                   DataPoint(3, Vector([0.0, 0.0])),
                   DataPoint(4, Vector([0.0, 0.1]))])

sent = Sentence([phonemes[0], phonemes[1], phonemes[2]])

prediction = Prediction.random(phonemes, 2)

print prediction
va = ViterbiAlign(phonemes)

for i in range(10):
  _, m, points = va.run(prediction, sent, data)
  prediction = Prediction.maximize_likelihood(phonemes, points)

#draw_points(points)
