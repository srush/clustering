#ifndef SEMIMARKOV_H
#define SEMIMARKOV_H

#include <vector>
#include "assert.h"
#include "span_chart.h"
using namespace std;
#define INF 1e20

class SemiMarkov {
 public:
 SemiMarkov(int states, int timesteps, int width_limit) 
   : num_states_(states),
    num_timesteps_(timesteps),
    width_limit_(width_limit),
    initialized_(false),
    forward_(false),
    backward_(false) { assert(num_states_< num_timesteps_); }

  ~SemiMarkov () {
    delete scores_;
    delete pruned_;
  }
  
  // Initialize the chart.
  void Initialize();

  // Run the semi-markov Viterbi forward algorithm.
  void ViterbiForward();

  // Run the semi-markov Viterbi backward algorithm.
  void ViterbiBackward();

  // Retrieve the best forward path and score. Requires that viterbi
  // forward was run.
  double GetBestPath(vector<int> *path) const;

  // Retieve the best path from the backward chart.
  double GetBestBackPath(vector<int> *path) const;

  double GetForwardScore(int state, int timestep) const;

  double GetBackwardScore(int state, int timestep) const;

  // Set the score associated with a span (start-end inclusive).
  void set_score(int start, int end, int state, double score) {
    assert(start <= end);
    scores_->get(start, end - start)[state] = score;
  }

  void set_pruned(int start, int end) {
    assert(end - start < width_limit_);
    pruned_->set(start, end - start, true);
  }
  
  // Getters for states and actions.
  int num_states() { return num_states_; }
  int num_timesteps() { return num_timesteps_; }

  // Score a state ranging from start to end inclusive. 
  double score(int start, int end, int state) const {
    assert(start <= end);
    return scores_->get(start, end - start)[state];
  }

  bool pruned(int start, int end) const { 
    return pruned_->get_const(start, end - start);
  }
 private:

  // Number of rows of the trellis.
  int num_states_;

  // Number of columns of the trellis.
  int num_timesteps_;

  // Limit the number timesteps per state.
  int width_limit_;

  // Is the memory allocated and initialized.
  bool initialized_;

  // Have we run the forward viterbi pass.
  bool forward_;

  // Have we run the backward viterbi pass.
  bool backward_;

  // Charts for viterbi forward. best_score_[time][state]
  vector<vector<double> > best_score_;
  vector<vector<int> > best_action_;

  // Charts for viterbi backwards. best_back_score_[time][state]
  vector<vector<double> > best_back_score_;
  vector<vector<int> > best_back_action_;

  SpanChart<vector<double> > *scores_;
  SpanChart<bool> *pruned_;
};

#endif
