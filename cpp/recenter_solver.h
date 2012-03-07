#ifndef RECENTER_SOLVER_H
#define RECENTER_SOLVER_H

#include "cluster_problem.h"
#include "speech_solution.h"

// Compute the 
class RecenterSolver {
 public:
 RecenterSolver(int type, int num_choices) : type_(type), num_choices_(num_choices) {} 

  // Find the best hidden for each type. Works dynamically, so takes
  // into account updates.
  double Solve(); 
  
  void set_reparameterization(vector<double> *segment_param,
                              vector<vector<double> > *center_param) {
    segment_param_ = segment_param;
    center_param_ = center_param;
  }

  // Compute max-marginals.
  double MaxMarginals(vector<double> *segment_mu,
                      vector<vector<double> > *center_mu);
  
  double Rescore(const SpeechSolution &solution) const;

  double Check(int type, int hidden);
                      
  bool IsChosen(int center, int use_sequence) const;
  double AllBut(int center, int center2, int num_needed) const;

  int SegmentCenter() const {
    return best_hidden_[1][0];
  } 
 private:
  int type_;

  const vector<vector<double> > *center_param_;
  const vector<double> *segment_param_;

  vector<double>  best_score_[2];
  vector<int>  best_hidden_[2];

  int num_choices_;
  int best_center_;
  int best_backup_center_;
  double all_off_;

};

#endif
