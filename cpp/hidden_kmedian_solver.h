#ifndef KMEDIANS_SOLVER_H
#define KMEDIANS_SOLVER_H

#include "cluster_problem.h"
#include "speech_solution.h"
#include "kmedians.h"
// Compute the 
class KMediansSolver {
 public:
 KMediansSolver(int type, const SpeechProblemSet &sp, const ClusterSet &cp) : 
  type_(type), sp_(sp), cp_(cp){ } 

  // Find the best hidden for each type. Works dynamically, so takes
  // into account updates.
  double Solve(); 
  
  void set_reparameterization(vector<vector<vector<double> > > *repar) {
    repar_ = repar;
  }

  // Compute max-marginals.
  double MaxMarginals(vector<double> *segment_mu);
  
  double Rescore(const SpeechSolution &solution) const;

  double Check(int type, int hidden);
                      
  int get_mode(int mode) {
    return centers_[mode];
  }

 private:
  kmedians *InitializeKMedians();

  int type_;
  const SpeechProblemSet &sp_;
  const ClusterSet &cp_;

  const vector<vector<vector<double> > > *repar_;

  vector<int> centers_;
};

#endif
