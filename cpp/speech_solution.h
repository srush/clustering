#ifndef SPEECH_SOLUTION_H
#define SPEECH_SOLUTION_H

#include <vector>
#include "cluster_problem.h"
using namespace std;

class SpeechAlignment {
 public:
  SpeechAlignment() {}
  SpeechAlignment(int problem, const vector<int> &alignment);
  SpeechAlignment(const SpeechAlignment &alignment);

  void StateAlign(int state, int *s, int *e) const {
    *s = alignment_[state];
    *e = alignment_[state + 1] - 1;
  }

  int HiddenAlign(int state) const {
    return hidden_align_[state];
  }

  vector<int> *mutable_alignment() {
    return &alignment_;
  }

  void set_hidden_alignment(int state, int hidden) {
    hidden_align_[state] = hidden;
  }

  vector<int> *mutable_hidden_alignment() {
    return &hidden_align_;
  }
  
  int alignment_size() const {
    return alignment_.size() - 1;
  }

 private:
  // The problem number.
  int problem_;
  
  // The alignment of states to start and end positions.
  vector<int> alignment_;

  // The alignment of states to hidden. Assumed to match type_to_hidden.
  vector<int> hidden_align_;
};

class SpeechSolution {
 public:
 SpeechSolution(const ClusterSet &cluster_set) : 
  speech_alignments_(cluster_set.problems_size()),
    type_to_hidden_(cluster_set.num_types()),
    cluster_set_(cluster_set)
    {}
  SpeechSolution(const SpeechSolution &solution);

  const SpeechAlignment &alignment(int problem) const {
    return speech_alignments_[problem];
  }

  vector<vector<int> > AlignmentAssignments() {
    int num_problems = speech_alignments_.size();
    vector<vector<int> > results(num_problems);
    for (int u = 0; u < num_problems; ++u) {
      const SpeechAlignment &align = alignment(u);
      results[u].resize(align.alignment_size());
      for (int i = 0; i < align.alignment_size(); ++i) {
        results[u][i] = align.HiddenAlign(i);
      }
    }
    return results;
  }

  vector<vector<int> > ClusterAssignments() {
    int num_problems = speech_alignments_.size();
    vector<vector<int> > results(num_problems);
    for (int u = 0; u < num_problems; ++u) {
      const SpeechAlignment &align = alignment(u);
      results[u].resize(align.alignment_size());
      for (int i = 0; i < align.alignment_size(); ++i) {
        int type = cluster_set_.problem(u).MapState(i); 
        results[u][i] = TypeToHidden(type);
      }
    }    
    return results;
  } 

  int TypeToHidden(int type) const {
    return type_to_hidden_[type];
  }

  SpeechAlignment *mutable_alignment(int problem) {
    return &speech_alignments_[problem];
  }

  vector<int> *mutable_types() { return &type_to_hidden_; } 
  
  void set_type_to_hidden(int type, int hidden) { 
    type_to_hidden_[type] = hidden; 
  } 

 private:
  // Alignments for each of the sentences.
  vector<SpeechAlignment> speech_alignments_;

  // Mapping from types to hidden choices.
  vector<int> type_to_hidden_; 

  const ClusterSet &cluster_set_;
};

#endif
