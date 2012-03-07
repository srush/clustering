#ifndef HOP_SOLVER_H
#define HOP_SOLVER_H

#include "cluster_problem.h"
#include "speech_solution.h"

// Compute the 
class HOPSolver {
 public:
 HOPSolver(int type, int num_choices) : type_(type), num_choices_(num_choices) {} 

  // Find the best hidden for each type. Works dynamically, so takes
  // into account updates.
  double Solve(); 
  
  void set_reparameterization(vector<vector<double> > *reparameterization) {
    reparameterization_ = reparameterization;
  }

  // Compute max-marginals.
  double MaxMarginals(vector<vector<double> > *mu);
  
  double Rescore(const SpeechSolution &solution) const;

  double Check(int type, int hidden);

  int get_mode(int mode) { return best_hidden_[mode]; } 
                      
 private:
  // The number of possible choices.
  int type_;
  int num_choices_;

  const vector<vector<double > > *reparameterization_;

  vector<double>  best_score_;
  vector<int>  best_hidden_;
};

#endif
