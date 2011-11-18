import sys
sys.path.append(".")
from sequence import *
from vector import *

phonemes = [Phoneme("a", 1), Phoneme("b", 2), Phoneme("c", 3)]
data = TimeSeries([DataPoint(0, Vector([0.0, 1.0])),
                   DataPoint(1, Vector([0.0, 0.1])),
                   DataPoint(2, Vector([1.0, 1.0])),
                   DataPoint(3, Vector([0.0, 0.0]))
                   ])

sent = Sentence([phonemes[0], phonemes[1], phonemes[2], phonemes[1]])

run_subgradient(phonemes, sent, data)
#draw_points(points)
