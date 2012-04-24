#ifndef VITERBI_H
#define VITERBI_H

#include <vector>
#include <assert.h>
using namespace std;

double LogAdd(double a, double b); 

class Viterbi {
 public:
 Viterbi(int num_states, int num_timesteps, int num_centers, int min_width): 
  num_states_(num_states), 
    num_timesteps_(num_timesteps),
    num_centers_(num_centers), 
    min_width_(min_width), 
    use_sum_(false) {}

  // Do marginals instead of min-marginals.
  void set_use_sum() {
    use_sum_ = true;
  }

  // Initialize the viterbi chart.
  void Initialize();

  // Run the Viterbi algorithm.
  void ForwardScores();
  void BackwardScores();
  
  // Compute the min-marginals of the chart. States x centers.
  void MinMarginals(vector<vector<double> > *min_marginals);
  
  // Compute the probabilistic marginals of the chart. 
  void Marginals(vector<vector<vector<double> > > *marginals);

  // Get the best path through the chart. 
  double GetBestPath(vector<int> *path, vector<int> *centers);

  // Get the best score through the chart. 
  double GetBestScore() {
    return forward_scores_[num_timesteps_][num_states_][0];
  }

  // Score a state ranging from start to end inclusive. 
  double score(int time, int center) const;

  // Set the score associated with a span (start-end inclusive).
  void set_score(int time, int center, double score) {
    scores_[time][center] = score;
  }

  double lambda(int state, int center) const {
    assert(state < num_states_);
    assert(center < num_centers_);
    return lambda_[state][center];
  }

  void set_lambda(int state, int center, double score) {
    lambda_[state][center] = score;
  }


  double transition_score(int time, int state, int center) const {
    assert(time < num_timesteps_);
    assert(state < num_states_);
    assert(center < num_centers_);
    return transition_score_[time][state][center];
  }

  void set_transition_score(int time, int state, int center, double score) {
    if (use_sum_) {
      assert(score >= 0.0);
    }
    transition_score_[time][state][center] = score;
  }


  int num_centers() { return num_centers_; } 
  int num_states() { return num_states_; } 
  int num_timesteps() { return num_timesteps_; } 

 private:
  // Reset the chart without resizing.
  void ResetChart();

  // Number of rows of the trellis.
  int num_states_;

  // Number of columns of the trellis.
  int num_timesteps_;

  // Number of centers available for each state.
  int num_centers_;

  // The minimum time steps to stay in each state.
  int min_width_;

  // Charts for viterbi
  vector<vector<vector<double > > > forward_scores_;
  vector<vector<vector<double > > > backward_scores_;
  
  // The center of the last state.
  vector<vector<vector<int> > > back_pointer_;

  // The parameters of the viterbi model.
  vector< vector<double> > scores_;
  vector< vector<double> > lambda_;
  vector< vector< vector<double> > > transition_score_;

  bool use_sum_;
};

#endif
