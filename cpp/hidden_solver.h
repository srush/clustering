#ifndef HIDDEN_SOLVER_H
#define HIDDEN_SOLVER_H

#include "cluster_problem.h"
//#include "cluster_subgrad.h"
#include "distances.h"
#include "speech_solution.h"

// Compute the second part of the decomposition.
class HiddenSolver {
 public:
  HiddenSolver(const ClusterSet &cs); 

               //int num_types, int num_hidden);

  // Find the best hidden for each type. Works dynamically, so takes
  // into account updates.
  double Solve(); 
  
  /* void Update(int type, int hidden, double score) { */

  /* } */
  void set_reparameterization(vector<vector<vector<double> > > *reparameterization) {
    reparameterization_ = reparameterization;

    for (int type = 0; type < cs_.num_types(); ++type) {
      for (int hidden = 0; hidden < cs_.num_hidden(type); ++hidden) {
        hidden_costs_[type][hidden] = 0.0;
      }
    }
    for (int problem = 0; problem < cs_.problems_size(); ++problem) {
      for (int state = 0; state < cs_.problem(problem).num_states; ++state) {
        int type = cs_.problem(problem).MapState(state);
        for (int hidden = 0; hidden < cs_.num_hidden(type); ++hidden) {
          hidden_costs_[type][hidden] += 
            (*reparameterization)[problem][state][hidden];
        }
      }
    }
  }
  /* bool is_eliminated(int type, int hidden) { */
  /*   return eliminated_[type].find(hidden); */
  /* } */

  // Compute max-marginals.
  double MaxMarginals(vector<vector<vector<double> > > *mu);
  
      /* void ToSubgrad(const ClusterSet &cs,  */
      /*            const BallHolder &ball_holder, */
      /*            ClusterSubgrad *subgrad) const; */

  double Rescore(const SpeechSolution &solution) const;

  double Check(int type, int hidden);
                      
  int TypeToHidden(int type, int mode) {
    return best_hidden_[mode][type];
  }
                      
 private:
  
  // Has the type received an update since the last solve.
  vector<bool> type_dirty_;

  // The cost of a *type* choosing a center. Assumed dynamic.
  vector<vector <double> > hidden_costs_;

  // The current best hidden for a type score. 
  vector<vector<double> >  best_score_; 
  

  // The current best hidden for a type assignment. 
  vector<vector<int> > best_hidden_;

  const ClusterSet &cs_;
  const vector<vector<vector<double > > > *reparameterization_;

  int num_modes_;
};

#endif
