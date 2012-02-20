#ifndef VITERBI_H
#define VITERBI_H

#include <vector>
using namespace std;

class Viterbi {
 public:
 Viterbi(int num_states, int num_timesteps): 
  num_states_(num_states), num_timesteps_(num_timesteps) {}

  // Initialize the viterbi chart.
  void Initialize();

  // Run the Viterbi algorithm.
  double Run(vector<int> *path);

  // Score a state ranging from start to end inclusive. 
  double score(int time, int state) const {
    return scores_[time][state];
  }

  // Set the score associated with a span (start-end inclusive).
  void set_score(int time, int state, double score) {
    scores_[time][state] = score;
  }

 private:
  // Number of rows of the trellis.
  int num_states_;

  // Number of columns of the trellis.
  int num_timesteps_;

  // Charts for viterbi
  vector<vector<double> > best_score_;
  vector<vector<int> > best_action_;

  vector< vector<double> > scores_;
};

#endif
