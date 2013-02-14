#ifndef HMM_ASTAR_SOLVER_H
#define HMM_ASTAR_SOLVER_H

#include "cluster_problem.h"
#include "distances.h"
#include "viterbi.h"
#include "span_chart.h"
#include "speech_solution.h"
#include "astar_memory.h"
#include "hidden_solver.h"

class HMMAStarSolver {
 public:
 HMMAStarSolver(const ClusterProblem &cp, 
                const ThinDistanceHolder &distances,
                int problem) 
   : cp_(cp), distances_(distances), conflicts_(cp.num_types(), false), enforced_constraints_(cp.num_types(), false), enforced_(0), round_(0) , problem_(problem) {}

  // Return the min score through the path and 
  // an alignment of timesteps to states.  
  double Solve(SpeechAlignment *alignment, bool exact, const HiddenSolver &solver, 
               const Reparameterization &delta_hmm,
               const Reparameterization &delta_hidden,
               double upper_bound);

  void set_reparameterization(const vector<vector<double> > *reparameterization) {
    reparameterization_ = reparameterization;
  }
  
 private:
  Search<HMMState, Expander> *InitializeAStar(bool exact);

  const ClusterProblem &cp_;
  
  // The distances from the spans to the centers.
  const ThinDistanceHolder &distances_;

  // The path through the hmm.
  vector<int> semi_markov_path_;

  // Current mapping.
  vector<int> state_to_center_;

  const vector<vector<double> > *reparameterization_;

  vector<int> conflicts_;
  vector<bool> enforced_constraints_;
  int enforced_;
  int round_;
  int problem_;
};

#endif
