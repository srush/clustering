#ifndef HMM_BEAM_SEARCH_SOLVER_H
#define HMM_BEAM_SEARCH_SOLVER_H

#include "cluster_problem.h"
#include "distances.h"
#include "viterbi.h"
#include "span_chart.h"
#include "speech_solution.h"
#include "astar_memory.h"
#include "hidden_solver.h"

class HMMBeamSearchSolver {
 public:
 HMMBeamSearchSolver(const ClusterSet &cs, 
                     const vector<ThinDistanceHolder *> &distances) 
   : cs_(cs), 
    distances_(distances), round_(0) {}

  // Return the min score through the path and 
  // an alignment of timesteps to states.  
  double Solve(SpeechSolution *solution, 
               bool exact, const HiddenSolver &solver, 
               const Reparameterization &delta_hmm,
               const Reparameterization &delta_hidden,
               double upper_bound);

  void set_reparameterization(const Reparameterization *reparameterization) {
    reparameterization_ = reparameterization;
  }
  
 private:

  const ClusterSet &cs_;
  
  // The distances from the spans to the centers.
  const vector<ThinDistanceHolder *> &distances_;

  // The path through the hmm.
  vector<int> semi_markov_path_;

  // Current mapping.
  vector<int> state_to_center_;

  const Reparameterization *reparameterization_;

  int round_;
  int problem_;
};

#endif
