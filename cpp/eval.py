import sys
print "start"
#from nltk.corpus import timit
sys.path.append(".")
sys.path.append("sequence")
sys.path.append("build")
import speech_pb2 as speech
import math
import numpy as np
import numpy.linalg
#import matplotlib.pyplot as plt
#import mdp
print "start"

def dist(vec1, vec2):
  return pow(numpy.linalg.norm(vec1 - vec2), 2)

def to_array(vec):
  return np.array([d for d in vec.dim])

class PhonemeInfo:
  def __init__(self, phoneme, proposed_center, gold_center, centers):
    self.phoneme = phoneme
    self.gold_total = 0
    self.gold_distance = 0
    self.gold_count = 0
    self.proposed_center = proposed_center
    self.total = [0] * len(proposed_center)
    self.distance = [0] * len(proposed_center)
    self.count = [0] * len(proposed_center)
    self.gold_center = gold_center

    self.best_possible = min([dist(center, gold_center)  
                              for center in centers])

  def AddInstance(self, mode, size, distance):
    self.total[mode] += size
    self.distance[mode] += distance
    self.count[mode] += 1

  def AddGoldInstance(self, size, distance):
    self.gold_total += size
    self.gold_distance += distance
    self.gold_count += 1

  def __cmp__(a, b):
    return cmp(b.gold_total, a.gold_total)

  def __repr__(self):
    return "PHO: %6s %3d %3d %5.1f %7.3f %7.3f\n"%("/"+self.phoneme+"/", 
                                      self.gold_count,
                                      self.gold_total,
                                      float(self.gold_total)/ self.gold_count,
                                      self.gold_distance / float(self.gold_total + 1),
                                              self.best_possible) + \
  "\n".join(["PHO: %6s %3d %3d %5.1f %7.3f %7.3f"%("", self.count[m], 
                                   self.total[m], 
                                   float(self.total[m])/ self.count[m],
                                   self.distance[m] / float(self.total[m] + 1), dist(c, self.gold_center)) for m,c in enumerate(self.proposed_center) if self.count[m] != 0])

class SpeechSolution:
  def __init__(self, phoneme_set, utterances, centers, alignment):
    self.phoneme_set = phoneme_set
    self.utterances = utterances 
    self.alignment = alignment
    self.centers = [to_array(center) for center in centers.centers]

  def EstimateCorrectCenters(self):
    self.gold_centers = {}
    phoneme_type_sum = {}
    phoneme_type_count = {}
    print "estimating"
    for u, utterance in enumerate(self.utterances.utterances):
      sequence_vectors = map(to_array, utterance.sequence)
      for i, phone in enumerate(utterance.phones):
        gold_start = utterance.correct_division[i]
        gold_end = utterance.correct_division[i +1]
        for s in range(gold_start, gold_end):
          phoneme_type_count.setdefault(phone, 0)
          phoneme_type_sum.setdefault(phone, 0)
          phoneme_type_count[phone] += 1
          phoneme_type_sum[phone] += sequence_vectors[s]
    for type in phoneme_type_sum:
      self.gold_centers[type] = phoneme_type_sum[type] / float(phoneme_type_count[type])
    print "done estimating"
        

  def DrawPlot(self, gold):

    address = {}
    count = 0
    seqs = []    
    for u, (utterance, alignment) in enumerate(zip(self.utterances.utterances, 
                                                   self.alignment.alignments)):
    
      for s, seq in enumerate(utterance.sequence):
        seqs.append(to_array(seq))
        address[u,s] = count
        count += 1
    all_seq = np.array(seqs)
      
    pcan = mdp.nodes.PCANode(output_dim=2)
    pcar = pcan.execute(all_seq)
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    plot_count = 0
    for u, (utterance, alignment) in enumerate(zip(self.utterances.utterances, 
                                                   self.alignment.alignments)[:]):
      for i, phone in enumerate(utterance.phones):
        if not gold:
          proposed_start = alignment.proposed_segmentation[i]
          proposed_end = alignment.proposed_segmentation[i + 1]

        else:
          proposed_start = utterance.correct_division[i]
          proposed_end = utterance.correct_division[i + 1]

        type = phone / float(61);
        try:
          start = address[u, proposed_start]
          end = address[u, proposed_end]
        except KeyError:
          pass
        ax.plot(pcar[start:end+1,0], pcar[start:end+1,1], 'o',color= plt.cm.hsv(type))
        #ax.plot(sum(pcar[start:end+1,0])/ len(pcar[start:end+1,0]), sum(pcar[start:end+1,1])/ len(pcar[start:end+1,1]), 'o',color= plt.cm.hsv(type))

    plt.show()
    
  def ShowAlignment(self):
    total_diff = 0
    total_distance = 0
    total_gold_distance = 0
    hist = {}
    phonemes = {}
    phoneme_centers = []
    if not self.alignment.phoneme_multi_centers:
      phoneme_centers =[ [to_array(center)] for center in self.alignment.phoneme_centers ]
    else:
      for multi in self.alignment.phoneme_multi_centers:
        phoneme_centers.append(map(to_array, multi.centers))
    print 
    self.EstimateCorrectCenters()
    for i, phoneme in enumerate(self.phoneme_set.phonemes):
      if i in self.gold_centers.keys():
        phonemes[i] = PhonemeInfo(phoneme.name, 
                                  phoneme_centers[i], 
                                  self.gold_centers[i],
                                  self.centers)

    for utterance, alignment in zip(self.utterances.utterances, 
                                    self.alignment.alignments):
      diff = 0
      sent_distance = 0
      gold_sent_distance = 0
      print "SENT:", utterance.sentence
      print "PHONEMES:",
      for phone in utterance.phones:
        print  "/" + self.phoneme_set.phonemes[phone].name + "/",
      print 
      print
      sequence_vectors = map(to_array, utterance.sequence)
      for i, phone in enumerate(utterance.phones):
        distance = 0.0
        gold_distance = 0.0
        gold_proposed_distance = 0.0
        selected_center = 0
        if alignment.selected_centers:
          selected_center = alignment.selected_centers[i]
        center = phoneme_centers[phone][selected_center]

        proposed_start = alignment.proposed_segmentation[i]
        proposed_end = alignment.proposed_segmentation[i + 1]
        for s in range(proposed_start, proposed_end):
          distance += dist(sequence_vectors[s], center)

        phonemes[phone].AddInstance(selected_center, abs(proposed_end - proposed_start), distance)
        gold_start = utterance.correct_division[i]
        gold_end = utterance.correct_division[i +1]
        gold_center = self.gold_centers[phone]
        for s in range(gold_start, gold_end):
          gold_distance += dist(sequence_vectors[s], gold_center)
          gold_proposed_distance += dist(sequence_vectors[s], center)


        gold_sum = 0 
        for s in range(gold_start, gold_end):
          gold_sum += sequence_vectors[s]

        phonemes[phone].AddGoldInstance(abs(gold_end - gold_start), gold_distance)
        if gold_end - gold_start > 0:
          gold_average = gold_distance / float(gold_end - gold_start)
          gold_centroid = gold_sum / float(gold_end - gold_start)
          # centroid_sum = 0
          # for s in range(gold_start, gold_end):
          #   centroid_sum += dist(to_array(utterance.sequence[s]), gold_centroid)
          # centroid_dist = centroid_sum / float(gold_end - gold_start)
        else:
          gold_average = 0.0 
          centroid_dist = 0.0

        proposed_average = distance / float(proposed_end - proposed_start)
        symbol = ""
        if abs(proposed_start - gold_start) > 6:
          symbol = "X"
        if abs(proposed_start - proposed_end) <= 3:
          symbol += "S"
        print "INS: %3d %5s %4d %4d %7.3f %7.3f %4d %4d %7.3f %3d %3d %3d %s"\
        %(i, "/" + self.phoneme_set.phonemes[phone].name + "/", 
          gold_start, 
          gold_end, 
          gold_average, 
          gold_proposed_distance / float(gold_end - gold_start),
          proposed_start, 
          proposed_end, 
          proposed_average, 
          abs(proposed_start - proposed_end), 
          abs(proposed_start - gold_start), 
          selected_center, 
          symbol)
        local_diff = abs(proposed_start - gold_start)
        diff += abs(proposed_start - gold_start)
        hist.setdefault(local_diff, 0)
        hist[local_diff] += 1 

        sent_distance += distance
        gold_sent_distance += gold_distance
      print "SentL1Dist: ", diff
      print "SentObj: ", sent_distance
      print "GoldObj: ", gold_sent_distance
      total_diff += diff
      total_gold_distance += gold_sent_distance
      total_distance += sent_distance

    for info in sorted(phonemes.values()):
      print str(info)

    print "L1Dist: ", total_diff
    print "TotalObj: ", total_distance
    print "TotalGoldObj: ", total_gold_distance
    
    hist_sum = 0.0
    for k in hist:
      hist_sum += hist[k]
    total = 0.0
    for k in hist:
      total += hist[k]
      if k <= 4:
        print k, total / float(hist_sum)
      

  @staticmethod
  def load(phoneme_file, utterance_file, center_file, alignment_file):
    print "parsing"
    phoneme_set = speech.PhonemeSet()
    utterance_set = speech.UtteranceSet()
    utterance_alignments = speech.SpeechSolution()
    centers = speech.CenterSet()
    phoneme_set.ParseFromString(open(phoneme_file,'r').read())
    utterance_set.ParseFromString(open(utterance_file, 'r').read())
    centers.ParseFromString(open(center_file, 'r').read())
    utterance_alignments.ParseFromString(open(alignment_file, 'r').read())
    print "done parsing"
    return SpeechSolution(phoneme_set, utterance_set, centers, utterance_alignments)



speech_solution = SpeechSolution.load(sys.argv[1],
                                      sys.argv[2],
                                      sys.argv[3],
                                      sys.argv[4])
#                                      "/tmp/kmeans_solution40")

speech_solution.ShowAlignment()
#speech_solution.DrawPlot(True)

print 
