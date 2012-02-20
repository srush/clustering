#ifndef SPEECH_ALIGN_H
#define SPEECH_ALIGN_H


class AlignmentProblem {
 public:
 AlignmentProblem(const Utterance &utterance, 
                  const TimeSequence &time_sequence, 
                  const PhonemeSet &phoneme_set): 
  utterance_(utterance), time_sequence_(utterance), phoneme_set_(utterance)
  {}
  
  const Utterance &utterance() {
    return utterance_;
  }

  const TimeSequence &time_sequence() {
    return time_sequence_;
  }

  const PhonemeSet &phoneme_set() {
    return phoneme_set_;
  }

 private:
  // The sequence of phonemes.
  const Utterance &utterance_;

  // The sequence of speech vectors.
  const TimeSequence &time_sequence_;

  // The global set of language phoneme types.
  const PhonemeSet &phoneme_set_;
};


class AlignmentSolution {
 public:
  AlignmentSolution(const AlignmentProblem &problem,
                    vector<int> change_points);
  
  void Print();

 private:
  // The problem that this is a solution for.
  const AlignmentProblem &problem_;

  // The start of the phone.
  vector<int> phone_start_; 

  // The end of the phone.
  vector<int> phone_end_; 
};


// An alignment solver takes a AlignmentProblem 
// and proposes an alignment solution.
class AlignmentSolver {
  AlignmentSolver() {}
  virtual AlignmentSolver() {}

  virtual AlignmentSolution SolveAlignment(const AlignmentProblem &problem);
};

#endif
