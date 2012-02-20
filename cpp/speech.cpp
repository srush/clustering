#include "speech.h"

DataPoint *DataPointFromProtobuf(const speech::Vector &vector) {
  int size = vector.dim_size();
  DataPoint *data_point = new DataPoint(size, size);
  for (int i = 0; i < size; ++i) {
    (*data_point)[i] = vector.dim(i);
  }
  return data_point;
}

PhonemeType *PhonemeType::FromProtobuf(const speech::Phoneme &phoneme) {
  return new PhonemeType(phoneme.id(), phoneme.name());
}

PhonemeSet *PhonemeSet::FromProtobuf(const speech::PhonemeSet &phoneme_set_buf) {
  PhonemeSet *phoneme_set = new PhonemeSet();
  for (int i = 0; i < phoneme_set_buf.phonemes_size(); ++i) {
    PhonemeType *type = PhonemeType::FromProtobuf(phoneme_set_buf.phonemes(i));
    phoneme_set->phonemes_.push_back(type);
  }
  return phoneme_set;
}

int Utterance::ScoreAlignment(const vector<int> &alignment) const {
  //assert(alignment.size() == correct_divisions_.size());
  int score = 0;
  for (uint i = 0; i < alignment.size(); ++i) {
    score += abs(alignment[i] - correct_divisions_[i]);
    //cerr << alignment[i] << " " << correct_divisions_[i] << endl;
  }
  return score;
} 

Utterance *Utterance::FromProtobuf(const PhonemeSet &phoneme_set,
                                   const speech::Utterance &utterance_buf) {
  Utterance *utterance = new Utterance(phoneme_set);

  // Phones
  for (int i = 0; i < utterance_buf.phones_size(); ++i) {
    utterance->phones_.push_back(utterance_buf.phones(i));
  }

  // Data points. 
  for (int i = 0; i < utterance_buf.sequence_size(); ++i) {
    DataPoint *point = DataPointFromProtobuf(utterance_buf.sequence(i));
    utterance->sequence_.push_back(point);
  }
  utterance->num_features_ = utterance_buf.feature_dimensions();

  utterance->sentence_ = utterance_buf.sentence();
  
  // Gold divisions. 
  utterance->correct_divisions_.resize(utterance_buf.correct_division_size());
  for (int i = 0; i < utterance_buf.correct_division_size(); ++i) {
    utterance->correct_divisions_[i] = utterance_buf.correct_division(i);
  }

  return utterance;
}

