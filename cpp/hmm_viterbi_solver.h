#ifndef HMM_VITERBI_SOLVER_H
#define HMM_VITERBI_SOLVER_H

#include "cluster_problem.h"
#include "distances.h"
#include "viterbi.h"
#include "span_chart.h"
#include "speech_solution.h"

class HMMViterbiSolver {
 public:
 HMMViterbiSolver(const ClusterProblem &cp, 
                  const ThinDistanceHolder &distances) : 
  cp_(cp), distances_(distances) {
  }

  // Return the min score through the path and 
  // an alignment of timesteps to states.  
  double Solve(SpeechAlignment *alignment);

  // Compute the max-marginals for this problem.
  double MaxMarginals(vector<vector<double> > *mu, 
                      SpeechAlignment *alignment);

  void set_reparameterization(const vector<vector<double> > *reparameterization) {
    reparameterization_ = reparameterization;
  }
  
  double Check(int type, int hidden) {
    double check = 0.0;
    assert(false);
    /* for (int i = 0; i < cp_.num_states; ++i) { */
    /*   if (type == cp_.MapState(i)) { */
    /*     check += hidden_holder_.hidden_costs(i, hidden); */
    /*   } */
    /* } */
    return check;
  }

  // Rescore this problem based on a solution.
  double Rescore(int problem, const SpeechSolution &solution) const;
  double PrimalRescore(int problem, const SpeechSolution &solution) const;

  double ScoreSegment(int start, int offset, int state, int center) const;
  double PrimalScoreSegment(int start, int offset, int state, int center) const;


 private:
  Viterbi *InitializeViterbi();

  const ClusterProblem &cp_;
  
  // The distances from the spans to the centers.
  const ThinDistanceHolder &distances_;

  // The path through the hmm.
  vector<int> semi_markov_path_;

  // Current mapping.
  vector<int> state_to_center_;

  const vector<vector<double> > *reparameterization_;

  Viterbi *viterbi_;
};

#endif
