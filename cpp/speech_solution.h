#ifndef SPEECH_SOLUTION_H
#define SPEECH_SOLUTION_H

#include <vector>
#include "build/speech.pb.h"
#include "cluster_problem.h"
#include "speech_problem.h"
using namespace std;

class SpeechProblemSet;

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

  void ToProtobuf(speech::UtteranceAlignment &align) const {
    for (uint i = 0; i < alignment_.size(); ++i) {
      align.add_proposed_segmentation(alignment_[i]);
    }
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
    use_special_(false),
    type_to_special_(cluster_set.num_types()),
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

  // Score the current speech solution. 
  int ScoreSolution() const {
    int total = 0;
    for (int u = 0; u < cluster_set_.problems_size(); ++u) {
      const SpeechAlignment &align = alignment(u);
      const ClusterProblem &problem = cluster_set_.problem(u);
      for (int i = 0; i < problem.num_states; ++i) {
        int s, e;
        align.StateAlign(i, &s, &e);
        int s2, e2;
        problem.GoldStateAlign(i, &s2, &e2);
        total += abs(s2 - s);
      }
    }
    return total;
  }

  void ToProtobuf(speech::SpeechSolution &solution, 
                  const SpeechProblemSet &speech_problem) const;

  int TypeToHidden(int type) const {
    return type_to_hidden_[type];
  }

  SpeechAlignment *mutable_alignment(int problem) {
    return &speech_alignments_[problem];
  }

  vector<int> *mutable_types() { return &type_to_hidden_; } 
  
  void set_type_to_hidden(int type, int hidden) { 
    type_to_hidden_[type] = hidden; 
    use_special_ = false;
  } 

  void set_type_to_special(int type, DataPoint point) {
    type_to_special_[type] = point;
    use_special_ = true;
  }

 private:
  // Alignments for each of the sentences.
  vector<SpeechAlignment> speech_alignments_;

  // Mapping from types to hidden choices.
  vector<int> type_to_hidden_;
  
  bool use_special_;
  vector<DataPoint> type_to_special_;

  const ClusterSet &cluster_set_;
};

#endif
