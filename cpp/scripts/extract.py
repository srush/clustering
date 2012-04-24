import sys, os
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
gflags.DEFINE_string('data_set', "brugnara", 'Speech data set to use.')
gflags.DEFINE_integer('clusters', 500, 'Number of clusters to use.')
gflags.MarkFlagAsRequired('output_name')

def load_timit(path):
  return timit.TimitCorpusReader(nltk.data.FileSystemPathPointer(path))

class FeatureExtractor:
  def __init__(self):
    fp = FeaturePlan(sample_rate=16000)
    fp.addFeature('mfcc: MFCC blockSize=160 stepSize=80')
    fp.addFeature('mfcc_d1: MFCC blockSize=160 stepSize=80 > Derivate DOrder=1')
    fp.addFeature('mfcc_d2: MFCC blockSize=160 stepSize=80 > Derivate DOrder=2')
    # fp.addFeature('mfcc: MFCC blockSize=512 stepSize=512')
    # fp.addFeature('mfcc_d1: MFCC blockSize=512 stepSize=512 > Derivate DOrder=1')
    # fp.addFeature('mfcc_d2: MFCC blockSize=512 stepSize=512 > Derivate DOrder=2')

    df = fp.getDataFlow()

    self.engine = Engine()
    self.engine.load(df)

  def extract_features(self, wav_path):
    afp = AudioFileProcessor()
    afp.processFile(self.engine, wav_path)
    features = self.engine.readAllOutputs()

    # Collapse the features into vectors.
    transposed_features = [] 
    first = True 
    for f in features.itervalues(): 
      for i, time_step in enumerate(f):
        if first:
          transposed_features.append(np.array(time_step))
        else:
          transposed_features[i] = np.append(transposed_features[i],
                                             time_step)
      first = False
    return transposed_features



class SpeechProblem:
  def __init__(self, corpus, output_name):
    # Set of allowed centers.
    self.center_set = speech.CenterSet()

    # Set of used phonemes.
    self.phoneme_set = speech.PhonemeSet()

    # Set of utterances.
    self.utterance_set = speech.UtteranceSet()

    self.phoneme_map = {}
    # Name of this speech problem.
    self.output_name = output_name

    self.corpus = corpus
    self.all_features = []

  def add_utterance(self, file_name, features):
    # Information from TIMIT. 
    phone_times = self.corpus.phone_times(file_name)
    phones = self.corpus.phones(file_name)

    # Protobuf information.
    utterance = self.utterance_set.utterances.add()
    utterance.sentence_file = file_name
    utterance.sentence = " ".join(self.corpus.words(file_name))

    assert(self.phoneme_map)
    for phone in phones:
      # Remove phoneme q hack.
      if phone == "q": continue      
      utterance.phones.append(self.phoneme_map[phone])

    last = float(phone_times[-1][2])
    final = len(features)
    segments = []

    # Dropped frames. 
    drop = 0 
    drop_set = set()
    # Convert timit markers to wave frames. 
    def pos(i): return int(round(i/last * final))
    for (p, s, e) in phone_times:
      if p == "q": 
        for i in range(pos(s), pos(e)):
          drop_set.add(i)
          drop += 1
      else:
        segments.append(pos(s) - drop)
    segments.append(final - drop)

    # Write the correct segments.
    for time in segments:
      utterance.correct_division.append(time)
    
    # Write the features at each timestep.
    for i, time_step in enumerate(features):
      if i in drop_set: continue
      self.all_features.append(np.array(time_step))
      sequence = utterance.sequence.add()
      utterance.feature_dimensions = len(time_step)
      for dim in time_step:
        sequence.dim.append(dim)

    # Checks
    assert(len(utterance.sequence) == final - drop)
    for end in segments:
      assert(end <= final - drop)

  def extract_phonemes(self):
    for i, p in enumerate(set(self.corpus.phones())):
      phoneme = self.phoneme_set.phonemes.add()
      phoneme.id = i
      phoneme.name = p
      self.phoneme_map[p] = i

  # Given all available feature vectors, extract possible centers.
  def extract_centers(self):
    centers, _ = vq.kmeans2(np.array(self.all_features), FLAGS.clusters)
    for i, center in enumerate(centers):
      vec = self.center_set.centers.add()
      for dim in center:
        vec.dim.append(dim)
      segment = self.center_set.segments.add()

  def write(self):
    try:
      os.mkdir("../data/problems/%s/"%self.output_name)
    except:
      pass
    f = open("../data/problems/%s/pho"%self.output_name, "wb")
    f.write(self.phoneme_set.SerializeToString())
    f.close()

    f2 = open("../data/problems/%s/utt"%self.output_name, "wb")
    f2.write(self.utterance_set.SerializeToString())
    f2.close()

    f3 = open("../data/problems/%s/cent"%self.output_name, "wb")
    f3.write(self.center_set.SerializeToString())
    f3.close()

def load_brugnara_files(timit):
  brugnara = set([l.strip() for l in open("corpus")])
  return [f for f in timit.utterances() 
          if f.split("-")[1].split("/")[0] in brugnara 
          and "sa1" not in f and "sa2" not in f]

def main(argv):
  argv = FLAGS(argv) 
  timit = load_timit(FLAGS.corpus_path)
  extractor = FeatureExtractor()

  utterance_names = []
  if FLAGS.data_set == "brugnara":
    utterance_names = load_brugnara_files(timit)
  elif FLAGS.data_set == "ten":
    utterance_names = load_brugnara_files(timit)[:10]
  elif FLAGS.data_set == "one":
    utterance_names = load_brugnara_files(timit)[:1]

  speech_problem = SpeechProblem(timit, FLAGS.output_name)
  speech_problem.extract_phonemes()
  for utterance_file in utterance_names:
    features = extractor.extract_features(timit.abspath(utterance_file + ".wav"))
    speech_problem.add_utterance(utterance_file, features)
  speech_problem.extract_centers()
  speech_problem.write()

if __name__ == '__main__':
  main(sys.argv)
