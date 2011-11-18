
class JacardDistance:
  def distance(self, point1, point2):
    def set_metric(s1, s2):
      return 1.0 - (len(s1 & s2) / float(len(s1 | s2)))
    return set_metric(set(point1.location.last_words.keys()), set(point2.location.last_words.keys())) + \
           set_metric(set(point1.location.next_words.keys()), set(point2.location.next_words.keys()))
           

class TypeCollection:
  def __init__(self):
    self.word_types = {}

  def GetType(self, typ):
    if typ not in self.word_types:
      self.word_types[typ] = WordType(typ)
    return self.word_types[typ]

  def GetMostCommon(self, n):
    word_types = self.word_types.values()
    word_types.sort(key = lambda x: x.seen)
    return word_types[:n]
              
class WordType:
  def __init__(self, word):
    self.word = word
    self.id = hash(word)
    self.next_words = {}
    self.last_words = {}
    self.seen = 0

  def __hash__(self):
    return self.word

  def __repr__(self):
    return self.word

  def Increment(self):
    self.seen += 1
    
  def AddInstance(self, last_word, next_word):
    self.last_words.setdefault(last_word, 0)
    self.next_words.setdefault(next_word, 0)
    self.last_words[last_word] += 1
    self.next_words[next_word] += 1


  
def ReadBrown(handle):
  collection = TypeCollection()
  for l in handle:
    words = [w.split("/")[0] for w in l.split()]
    last = len(words) - 1
    for i, word in enumerate(words):
      word_type = collection.GetType(word)
      word_type.Increment()
      if i == 0:
        last_word = "*START*"
      else:
        last_word = words[i - 1]
      if i == last:
        next_word = "*END*"
      else:
        next_word = words[i + 1]
      word_type.AddInstance(last_word, next_word)
  return collection
      

class WordStats:
  def __init__(self, num_classes):
    self.classes = range(num_classes)
    self.f_pair = {}
    self.f_one = {}
    self.f_two = {}
    self.words = set()
    self.num_word_instances = 0
    
  def AddPair(self, w1, w2):
    self.words.add(w1)
    self.words.add(w2)
    self.f_pair.setdefault((w1, w2), 0)
    self.f_pair[w1,w2] += 1
    self.f_one.setdefault(w1, 0)
    self.f_one[w1] += 1
    self.f_two.setdefault(w2, 0)
    self.f_two[w2] += 1
    self.num_word_instances += 1
    
def ReadBrownWordStats(handle, num_classes):
  collection = WordStats(num_classes)
  for l in handle:
    words = [w.split("/")[0] for w in l.split()]
    last = len(words) - 1
    for i, word in enumerate(words):
      if i == 0: continue
      collection.AddPair(words[i-1], words[i])
  return collection
