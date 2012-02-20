#ifndef CLUSTER_RELAX_H
#define CLUSTER_RELAX_H

#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <vector>
#include "cluster_problem.h"
#include "cluster_subgrad.h"
#include "distances.h"
#include "semimarkov.h"
#include "span_chart.h"
using namespace std;

// Precompute the best "hidden" state for each state.
class HiddenHolder { 
 public:
 HiddenHolder(const ClusterProblem &cp) : cp_(cp) {
    best_score_ = new SpanChart<vector<double> >(cp_.num_steps, cp.width_limit);
    best_hidden_ = new SpanChart<vector<int> >(cp_.num_steps, cp.width_limit);
    center_costs_ = new SpanChart<vector<double> >(cp_.num_steps, cp.width_limit);
    hidden_costs_.resize(cp.num_states); 
    for (int s = 0; s < cp_.num_steps; ++s) {
      for (int o = 0; o < cp_.width_limit; ++o) {
        best_score_->get(s, o).resize(cp.num_states);
        best_hidden_->get(s, o).resize(cp.num_states);
        center_costs_->get(s, o).resize(cp.num_hidden);
      }
    }
    for (int i = 0; i < cp_.num_states; ++i) {
      hidden_costs_[i].resize(cp.num_hidden);
    }
  }

  void Update(int state, int hidden, int score) {
    hidden_costs_[state][hidden] += score;
  }

  void ComputeBestHidden() {
    for (int s = 0; s < cp_.num_steps; ++s) {
      for (int o = 0; o < cp_.width_limit; ++o) {
        const vector<double> &center_cost = center_costs_->get(s, o); 
        for (int i = 0; i < cp_.num_states; ++i) {
          const vector<double> &hidden_cost = hidden_costs_[i];
          double *best = &best_score_->get(s, o)[i]; 
          for (int h = 0; h < cp_.num_hidden; ++h) {
            double cost = center_cost[h] + hidden_cost[i]; 
            if (cost < *best) {
              best_hidden_->get(s, o)[i] = h;
              *best = cost;
            } 
          }
        }
      }
    }
  }

  int get_center(int start, int end, int state) {
    return best_hidden_->get(start, end)[state];
  }

 private:
  const ClusterProblem &cp_;

  // The cost from a span to a center, does not change between iterations.
  SpanChart<vector <double> > *center_costs_;

  // The cost of a state choosing a center. Assumed dynamic.
  vector<vector <double> > hidden_costs_;

  // The current best hidden for a span and a state. 
  SpanChart<vector<int> > *best_hidden_;
  
  // The current best score for a span and a state. 
  SpanChart<vector<double > > *best_score_;  
};

class HMMSolver {
 public:
 HMMSolver(const ClusterProblem &cp, 
           const DistanceHolder &distances) : 
  cp_(cp), distances_(distances), hidden_holder_(cp), 
    semi_markov_(cp.num_states, cp.num_steps, cp.width_limit)
  {}

  double Solve() {
    hidden_holder_.ComputeBestHidden();
    double score = semi_markov_.Run(&semi_markov_path_);

    state_to_center_.resize(cp_.num_states);
    for (uint i = 0; i < semi_markov_path_.size() - 1; ++i) {
      int start = semi_markov_path_[i];
      int end = semi_markov_path_[i + 1];
      state_to_center_[i] = hidden_holder_.get_center(start, end, i);
    }
    return score;
  }

  void Update(int state, int hidden, int score) {
    hidden_holder_.Update(state, hidden, score);
  }

  ClusterSubgrad *ToSubgrad() const;

 private:
  const ClusterProblem &cp_;
  
  // The distances from the spans to the centers.
  const DistanceHolder &distances_;

  // Computes the best center for each span.
  HiddenHolder hidden_holder_;

  // Computes the best center for each span.
  SemiMarkov semi_markov_;

  // The path through the hmm.
  vector<int> semi_markov_path_;

  // Current mapping.
  vector<int> state_to_center_;
};

#endif
