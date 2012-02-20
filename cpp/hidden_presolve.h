#ifndef HIDDEN_PRESOLVE_H
#define HIDDEN_PRESOLVE_H

#include "cluster_problem.h"
#include "distances.h"
#include "span_chart.h"

// Precompute the best "hidden" state for each state.
class HiddenHolder { 
 public:
  HiddenHolder(const ClusterProblem &cp, 
               const DistanceHolder &distances);

  void set_reparameterization(const vector<vector<double> > *reparameterization);

  //double Update(int state, int hidden, double score) {
  //assert (state < cp_.num_states);
  //assert (hidden < cp_.num_hidden);
    /* if (hidden_costs_[state][hidden] + score < 0.0) { */
    /*   double actual = hidden_costs_[state][hidden]; */
    /*   hidden_costs_[state][hidden] = 0.0; */
    /*   return actual; */
    /* } else { */
    /*   hidden_costs_[state][hidden] += score; */
    /*   return score; */
    /* } */
    //hidden_costs_[state][hidden] += score;
    //return score;
  //}

  void ComputeBestHidden();

  int get_center(int start, int offset, int state) {
    assert (state < cp_.num_states);
    assert (start < cp_.num_steps);
    assert (offset < cp_.width_limit);
    return best_hidden_->get(start, offset)[state];
  }

  double get_score(int start, int offset, int state) {
    assert (state < cp_.num_states);
    assert (start < cp_.num_steps);
    assert (offset < cp_.width_limit);
    return best_score_->get(start, offset)[state];
  }

  double get_primal_score(int start, int offset, int state) {
    assert (state < cp_.num_states);
    assert (start < cp_.num_steps);
    assert (offset < cp_.width_limit);
    return primal_score_->get(start, offset)[state];
  }

  bool pruned(int s, int o) {
    return used_[s][o].size() == 0;
  }

  double Rescore(int s, int o, int state, int hidden) const {
    //cerr << cp_.MapState(state) << " " << hidden << " " << (*reparameterization_)[state][hidden] << endl;
    return
      distances_.get_distance(s, o, hidden) + (*reparameterization_)[state][hidden];
  }

  double PrimalRescore(int s, int o, int state, int hidden) const {
    return distances_.get_distance(s, o, hidden);
  }

  double hidden_costs(int state, int hidden) {
    return (*reparameterization_)[state][hidden];
  }
 private:
  const ClusterProblem &cp_;


  // The current best hidden for a span and a state. 
  SpanChart<vector<int> > *best_hidden_;
  
  // The current best score for a span and a state. 
  SpanChart<vector<double > > *best_score_;  

  // The current primal score (just distance) for a span and a state. 
  SpanChart<vector<double > > *primal_score_;  

  // The cost from a span to a center, does not change between iterations.
  const DistanceHolder &distances_;

  // Which hidden states are allowed for a split.
  vector<vector<vector<int> > > used_;

  // The cost of a state choosing a center. Assumed dynamic.
  //vector<vector <double> > hidden_costs_;

  // The reparameterization.
  const vector<vector<double> > *reparameterization_;
};

#endif
