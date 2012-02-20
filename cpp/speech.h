#ifndef SPEECH_H
#define SPEECH_H

#include <vector>
#include <string>

#include "data_point.h"
#include "build/speech.pb.h"
using namespace std;

DataPoint *DataPointFromProtobuf(const speech::Vector &vector);

class PhonemeType {
 public:
 PhonemeType(int id, string rep): id_(id), phoneme_(rep){}
  
  string phoneme() const {
    return phoneme_;
  }
  
  int id() const { return id_; }

  static PhonemeType *FromProtobuf(const speech::Phoneme &phoneme);

 private:
  // The id of the phoneme type.
  int id_;
  
  // The string representation of the phoneme.
  string phoneme_; 
};


class PhonemeSet {
 public:
  // The number of phonemes.
  int phonemes_size() const {
    return phonemes_.size();
  }

  // Get a phoneme.
  const PhonemeType &phoneme(int phoneme_id) const {
    return *phonemes_[phoneme_id];
  }
  
  // Read the phoneme set from the protobuf.
  static PhonemeSet *FromProtobuf(const speech::PhonemeSet &phoneme);
 private:
  // Number of phonemes.
  int phonemes_size_;

  // The set of phonemes.
  vector<PhonemeType *> phonemes_;
};

class Utterance {
 public:
 Utterance(const PhonemeSet &phoneme_set): phoneme_set_(phoneme_set) {}  

  int phones_size() const {
    return phones_.size();
  }
  
  const PhonemeType &phones(int phone_id) const {
    return phoneme_set_.phoneme(phones_[phone_id]);
  } 


  int sequence_size() const {
    return sequence_.size();
  }
  
  const DataPoint &sequence(int sequence_id) const {
    assert(sequence_id < sequence_size());
    return *sequence_[sequence_id];
  } 

  // The longest length possible for phones.
  const int max_phone_length() const {
    return 20;
  }

  const PhonemeSet &phoneme_set() const { return phoneme_set_; }

  int num_features() const { return num_features_ ; } 

  int num_phonemes() const {
    return phoneme_set_.phonemes_size();
  } 
  
  // Read the utterance from a protobuf.
  static Utterance *FromProtobuf(const PhonemeSet &phoneme_set,
                                 const speech::Utterance &utterance);

  // Score how close the current alignment is the correct alignment.
  int ScoreAlignment(const vector<int> &alignment) const;

 private:
  // The set of possble phonemes.
  const PhonemeSet &phoneme_set_;

  // The sentence in human readable form.
  string sentence_;
  
  // The number of features in each time step.
  int num_features_;

  // The phones in the sentece.
  vector<int> phones_;

  // The data time steps.
  vector<DataPoint *> sequence_;

  // The gold split of the phonemes.
  vector<int> correct_divisions_;
};

#endif
