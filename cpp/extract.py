import sys
sys.path.append(".")
sys.path.append("sequence")
sys.path.append("build")
from nltk.corpus.reader import timit
import nltk.data 
from yaafelib import *
import speech_pb2 as speech
import math
import numpy as np
import scipy.cluster.vq as vq
import numpy.linalg
import gflags
FLAGS = gflags.FLAGS

gflags.DEFINE_string('corpus_path', "/home/alexanderrush/Projects/clustering/corpus/TIMITNLTK", 'Path to TIMIT-like corpus')

gflags.DEFINE_string('output_name', "", 'Output name of this speech problem.')

gflags.MarkFlagAsRequired('output_name')



problem_name = sys.argv[1]

timit = timit.TimitCorpusReader(nltk.data.FileSystemPathPointer(CORPUS_PATH))
#print timit.utterances()


fp = FeaturePlan(sample_rate=16000)
fp.addFeature('mfcc: MFCC blockSize=160 stepSize=80')
fp.addFeature('mfcc_d1: MFCC blockSize=160 stepSize=80 > Derivate DOrder=1')
fp.addFeature('mfcc_d2: MFCC blockSize=160 stepSize=80 > Derivate DOrder=2')

# fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')
# fp.addFeature('mfcc_d1: MFCC blockSize=512 stepSize=256 > Derivate DOrder=1')
# fp.addFeature('mfcc_d2: MFCC blockSize=512 stepSize=256 > Derivate DOrder=2')

df = fp.getDataFlow()
df.display()

# configure an Engine
engine = Engine()
engine.load(df)

# Compute the set of all phoneme types.
phoneme_map = {}
phoneme_set = speech.PhonemeSet()
for i, p in enumerate(set(timit.phones())):
  phoneme = phoneme_set.phonemes.add()
  phoneme.id = i
  phoneme.name = p
  phoneme_map[p] = i
  print i, p 

# All the male utterances of a region.
brugnara = set([l.strip() for l in open("corpus")])
#f.startswith("dr1-f")
utterance_names = [f for f in timit.utterances() if f.split("-")[1].split("/")[0] in brugnara and "sa1" not in f and "sa2" not in f]
print len(utterance_names)
utterance_set = speech.UtteranceSet()
all_features = []
#for utterance_file in utterance_names: 
for utterance_file in utterance_names:
#for utterance_file in [ u for u in utterance_names if u == "dr8-mbcg0/sx57"]: 

  # extract features from an audio file using AudioFileProcessor
  afp = AudioFileProcessor()
  afp.processFile(engine, timit.abspath(utterance_file + ".wav"))
  phone_times = timit.phone_times(utterance_file)
  print phone_times
  last = float(phone_times[-1][2])
  features = engine.readAllOutputs()
  
  #print timit.sents(utterance_file)

  utterance = utterance_set.utterances.add()
  for phone in timit.phones(utterance_file):
    if phone == "q": continue
    utterance.phones.append(phoneme_map[phone])
  #print features["mfcc"]

  final = len(features["mfcc"])
  #print " ".join([  str(p)  for (p, s, e) in phone_times])
  #print " ".join([  str(int(round(s/last * final)))  for (p, s, e) in phone_times])
  ends = []
  drop = 0 
  drop_set = set()
  for (p, s, e) in phone_times:
    def to_position(i): return int(round(i/last * final))
    if p == "q": 
      for i in range(to_position(s), to_position(e)):
        drop_set.add(i)
        drop += 1
        print "dropping ",i
      continue
    ends.append(int(round(s/last * final)) - drop)
  ends.append(final - drop)
  print drop_set
  print ends
  for e in ends:
    utterance.correct_division.append(e)

  for i, (f1,f2,f3) in enumerate(zip(features["mfcc"], features["mfcc_d1"], features["mfcc_d2"])):
    if i in drop_set: continue
    feats = np.append(f1, [f2, f3])
    assert(len(feats) == 39)
    all_features.append(np.array(feats))
    sequence = utterance.sequence.add()
    utterance.feature_dimensions = len(feats)
    for f in feats:
      sequence.dim.append(f)
  assert(len(utterance.sequence) == final - drop)
  for end in ends:
    assert(end <= final - drop)
  utterance.sentence = " ".join(timit.words(utterance_file))



# Construct possible centers.
print len(all_features)
print "Feature Length ", len(all_features[0])


#hac = linkage(np.array(all_features))
#print hac
#tree = to_tree(hac)
#print tree
feat_address = {}
def construct(tree, address):
  if tree == None: return
  feat_address[tree.get_id()] = address
  construct(tree.get_right(), [1
] + address)
  construct(tree.get_left(), [0] + address)
#construct(tree, [])


# centers = [] # all_features
# print "pruning", len(all_features)
# for i, c in enumerate(all_features):
#   keep = True
#   print i
#   for c2 in centers:
#     if math.pow(numpy.linalg.norm(c - c2),2) < 1.0:
#       keep = False
#       break
#   if keep:
#     centers.append(c)

centers, _ = vq.kmeans2(np.array(all_features), 500)
center_set = speech.CenterSet()

for i, center in enumerate(centers):
  #print i, center, feat_address[i]
  vec = center_set.centers.add()
  for dim in center:
    vec.dim.append(dim)
  segment = center_set.segments.add()
  #for bit in feat_address[i]:
  #  segment.address.append(bit)

f = open("problems/%s_pho"%problem_name, "wb")
f.write(phoneme_set.SerializeToString())
f.close()
f2 = open("problems/%s_utt"%problem_name, "wb")
f2.write(utterance_set.SerializeToString())
f2.close()
f3 = open("problems/%s_cent"%problem_name, "wb")
f3.write(center_set.SerializeToString())
f3.close()



#phoneme_map = dict([(phoneme.name, phoneme) for phoneme in phoneme_types] )

# phoneme_types = [Phoneme(p,i) for i, p in enumerate(set(timit.phones()))]

# #print phoneme_map
# sent_phones = [phoneme_map[phone] for phone in timit.phones("dr8-mbcg0/sx57")]
# data_points = TimeSeries([DataPoint(i+1, f) for i, f in enumerate(feats["mfcc"])])

# prediction = Prediction.random(phoneme_types, 13)
# va = ViterbiAlign(phoneme_types)


