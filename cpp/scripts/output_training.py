import sys, os

from itertools import *
from nltk.corpus.reader import timit
import nltk.data 
from yaafelib import *
import numpy as np
from numpy.linalg import *
import scipy.cluster.vq as vq
import gflags
import random

random.seed(1)

FLAGS = gflags.FLAGS

inf = float("inf")

gflags.DEFINE_string('corpus_path', 
                     "/home/alexanderrush/Projects/clustering/corpus/", 
                     'Path to TIMIT-like corpus')
gflags.DEFINE_integer('vq_size', 256, 'Number of clusters to use in vq.')
gflags.DEFINE_bool('is_test', False, 'Is this test data.')
gflags.DEFINE_bool('is_dev', False, 'Is this dev data.')
gflags.DEFINE_string('output_prefix', 'timit', "")
gflags.DEFINE_bool('shrink_data', False, 'Use a small data set.')
gflags.DEFINE_bool('nca', False, 'Use nca.')

def load_timit(path):
  return timit.TimitCorpusReader(nltk.data.FileSystemPathPointer(path))

def load_brugnara_files(timit):
  brugnara = set([l.strip() for l in open("corpus")])
  return [f for f in timit.utterances() 
          if f.split("-")[1].split("/")[0] in brugnara 
          and "sa1" not in f and "sa2" not in f]

def load_training_files(timit):
  #brugnara = set([l.strip() for l in open("corpus")])
  return [f for f in timit.utterances() 
          if  "sa1" not in f and "sa2" not in f]

def load_core_test_files(timit):
  core = set([l.strip().lower() for l in open("core_test")])
  return [f for f in timit.utterances() 
          if f.split("-")[1].split("/")[0] in core 
          and "sa1" not in f and "sa2" not in f]

def load_core_dev_files(timit):
  core = set([l.strip().lower() for l in open("core_test")])
  return [f for f in timit.utterances() 
          if f.split("-")[1].split("/")[0] not in core 
          and "sa1" not in f and "sa2" not in f][:200]

class FeatureExtractor:
  def __init__(self):
    fp = FeaturePlan(sample_rate=16000)
    fp.addFeature('mfcc: MFCC blockSize=400 stepSize=160')
    fp.addFeature('mfcc_d1: MFCC blockSize=400 stepSize=160 > Derivate DOrder=1')
    fp.addFeature('mfcc_d2: MFCC blockSize=400 stepSize=160 > Derivate DOrder=2')
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


class VQ:
  def __init__(self, num_code_words):
    self.num_code_words = num_code_words
  
  def write_points(self, file, points, labels):
    nca_test = open(file, "w")
    for feat,label in izip(points, labels):
      print >>nca_test, str(label) + "," + ",".join(map(str, feat))
    nca_test.close()

  def run_nca(self):
    matrix = []
    os.system("../../nca/nca /tmp/nca_test 0 10000 0.01 > /tmp/nca_matrix")
    matrix_file = open("/tmp/nca_matrix")
    for l in matrix_file:
      if l.strip() == "START": continue
      if l.strip() == "END": break
      matrix.append(map(float, l.strip().split()))

    matrix = np.matrix(matrix)
    matrix_file.close()
    print matrix
    return matrix

    
  def make_code_book(self, all_features, labels):
    white_features = all_features #vq.whiten(all_features)
    print >>sys.stderr, "KMEANS"
    if len(white_features) > 50000:
      #inds = range(len(white_features))
      # kmeans_features_inds = random.sample(inds, 50000) 
      # kmeans_features = [white_features[ind] for ind in kmeans_features_inds]
      # kmeans_labels = [labels[ind] for ind in kmeans_features_inds]
      kmeans_features = random.sample(white_features, 50000)  #[white_features[ind] for ind in kmeans_features_inds]
    else:
      kmeans_features = white_features
      #kmeans_labels = labels

    if FLAGS.nca:
      self.write_points("/tmp/nca_test", kmeans_features[:2000], kmeans_labels[:2000])
      matrix = self.run_nca()
      kmeans_features = [ np.array(np.dot(matrix, feat))[0] for feat in kmeans_features ]
      white_features =  [ np.array(np.dot(matrix, feat))[0] for feat in white_features ]

    self.code_book, _ = vq.kmeans2(np.array(kmeans_features), 
                                   self.num_code_words)

    #print >>sys.stderr, "Done kmeans"
    vqs, _ = vq.vq(np.array(white_features), 
                   self.code_book)
    return vqs

  
  def assign_code_words(self, features):
    vqs, _ = vq.vq(np.array(features), 
                   self.code_book)
    #print vqs
    return vqs
    # TODO: be smart one nearest neighbors
  
def construct_gold(corpus, file, vq_features):
  phone_times = corpus.phone_times(file)
  phones = corpus.phones(file)
  utterance_data = []

  # Last time of the wave.
  last = float(phone_times[-1][2])

  # Last position in the features.
  final_feature_step = len(vq_features)

  def pos(i): return int(round(i/last * final_feature_step))

  for (p, s, e) in phone_times:
    #print p, s, e, pos(s),pos(e)
    
    frame_start = pos(s)
    frame_end = pos(e)
    for frame in range(frame_start, frame_end):
      vq_code_word = vq_features[frame]
      utterance_data.append((p, vq_code_word))

  return utterance_data

def main(argv):
  argv = FLAGS(argv) 
  data_sets = []
  timit_dev = load_timit(FLAGS.corpus_path + "TIMITNLTKTEST")
  if FLAGS.shrink_data:
    files = load_core_dev_files(timit_dev)[:10]
  else:
    files = load_core_dev_files(timit_dev)
  data_sets.append((timit_dev, files, "dev"))
  timit_train = load_timit(FLAGS.corpus_path + "TIMITNLTK39")
  if FLAGS.shrink_data:
    files = load_training_files(timit_train)[:10]
  else:
    files = load_training_files(timit_train)
  data_sets.append((timit_train, files, "train"))
  timit_test = load_timit(FLAGS.corpus_path + "TIMITNLTKTEST")
  if FLAGS.shrink_data:
    files = load_core_test_files(timit_test)[:10]
  else:
    files = load_core_test_files(timit_test)
  data_sets.append((timit_test, files, "test"))
    

  extractor = FeatureExtractor()
  all_features = []
  #all_states = []
  utterance_features = []
  feature_count = 0
  for timit, utterance_names, _ in data_sets:
    for utterance_file in utterance_names:
      features = extractor.extract_features(timit.abspath(utterance_file + ".wav"))

      #utterance_data = construct_gold(timit, utterance_file, features)
      #states = [state for state,_ in utterance_data]
      
      global_indices = []
      #for feature, state in features: # izip(features, states):
      for feature in features: # izip(features, states):
        all_features.append(feature)
        #all_states.append(state)
        global_indices.append(feature_count)
        feature_count += 1
      utterance_features.append(global_indices)

  vq = VQ(FLAGS.vq_size)
  vqs = vq.make_code_book(all_features, [])
  
  utterance_ind = 0
  for timit, utterance_names, suffix in data_sets:
    file_name = FLAGS.output_prefix + "_" + str(FLAGS.vq_size) + "_" + suffix
    if FLAGS.shrink_data:
      file_name += "_shrink"
    out_file = open(file_name, 'w')
    all_utterances = []
    for utterance_file in utterance_names:
      feature_inds = utterance_features[utterance_ind]
      utterance_ind += 1
      vq_features = [vqs[ind] for ind in feature_inds]

      utterance_data = construct_gold(timit, utterance_file, vq_features)
      all_utterances += utterance_data
      
      print >>out_file, " ".join(["%s/%s"%(p,code) for (p, code) in utterance_data])

  if False:
    correct_steps = 0
    total_steps = 0

    phoneme_histogram = {}
    vq_histogram = {}
    for p, code in all_utterances:
      phoneme_histogram.setdefault(p, {})
      phoneme_histogram[p].setdefault(code, 0)
      phoneme_histogram[p][code] += 1

      vq_histogram.setdefault(code, {})
      vq_histogram[code].setdefault(p, 0)
      vq_histogram[code][p] += 1
      total_steps += 1

    for p,groups in phoneme_histogram.iteritems():
      print p
      pairs = groups.items()
      pairs.sort(key=lambda a: a[1])
      pairs.reverse()
      total = sum([num for _,num in pairs])
      print "\t",
      for code, nums in pairs:
        if nums / float(total) < 0.01: continue
        print "%3s:%3.2f "%(code, nums / float(total)),
      print 

    for code in range(FLAGS.vq_size):
      if code not in vq_histogram: continue
      groups = vq_histogram[code]
      print code,
      pairs = groups.items()
      pairs.sort(key=lambda a: a[1])
      pairs.reverse()
      total = sum([num for _,num in pairs])
      correct_steps += pairs[0][1]
      print "", total,
      print "\t",
      print
      for p, nums in pairs:
        if nums / float(total) < 0.01: continue
        print "%3s:%3.2f "%(p, nums/ float(total)),
      print

    print total_steps, correct_steps, correct_steps / float(total_steps)
    print len(phoneme_histogram.keys())
    
if __name__ == '__main__':
  main(sys.argv)
