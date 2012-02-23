#ifndef HMM_SOLVER_H
#define HMM_SOLVER_H

#include "cluster_problem.h"
//#include "cluster_subgrad.h"
#include "hidden_presolve.h"
#include "distances.h"
#include "semimarkov.h"
#include "span_chart.h"
#include "speech_solution.h"

class HMMSolver {
 public:
 HMMSolver(const ClusterProblem &cp, 
           const DistanceHolder &distances
           ) : 
  cp_(cp), hidden_holder_(cp, distances), distances_(distances) {}

  // Return the min score through the path and 
  // an alignment of timesteps to states.  
  double Solve(SpeechAlignment *alignment);

  // Compute the max-marginals for this problem.
  double MaxMarginals(vector<vector<double> > *mu);

  void set_reparameterization(const vector<vector<double> > *reparameterization) {
    hidden_holder_.set_reparameterization(reparameterization); 
  }

  /* double Update(int state, int hidden, double score) { */
  /*   return hidden_holder_.Update(state, hidden, score); */
  /* } */

  /* void Update(vector<vector<double> > reparameterization ) { */
    
  //}
  
  // Convert the current estimate to a subgradient.
  /* void ToSubgrad(int problem,  */
  /*                const BallHolder &ball_holder, */
  /*                ClusterSubgrad *subgrad) const; */
  
  double Check(int type, int hidden) {
    double check = 0.0;
    for (int i = 0; i < cp_.num_states; ++i) {
      if (type == cp_.MapState(i)) {
        check += hidden_holder_.hidden_costs(i, hidden);
      }
    }
    return check;
  }

  /* double CheckScore(const vector<int> &path) { */
  /*   for (uint i = 0; i < path.size() - 1; ++i) { */
  /*     int start = path[i]; */
  /*     int end = path[i + 1] -1; */
  /*     state_to_center_[i] = hidden_holder_.get_center(start,  */
  /*                                                     end - start,  */
  /*                                                     i);     */
  /*     primal += hidden_holder_.get_primal_score(start, end - start, i); */
  /*   } */
  /*   return primal; */
  /* } */

  // Rescore this problem based on a solution.
  double Rescore(int problem, const SpeechSolution &solution) const;
  double PrimalRescore(int problem, const SpeechSolution &solution) const;

 private:
  SemiMarkov *InitializeSemiMarkov();

  const ClusterProblem &cp_;

  // Computes the best center for each span.
  HiddenHolder hidden_holder_;
  
  // The distances from the spans to the centers.
  const DistanceHolder &distances_;

  // Computes the best center for each span.
  //SemiMarkov semi_markov_;

  // The path through the hmm.
  vector<int> semi_markov_path_;

  // Current mapping.
  vector<int> state_to_center_;
};

#endif
