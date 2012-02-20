#ifndef SPEECH_SUBGRAD_H
#define SPEECH_SUBGRAD_H

#include "cluster_subgrad.h"
#include "hidden_solver.h"
#include "hmm_solver.h"
#include "subgrad.h"
#include "speech_problem.h"

// Setup for the speech subgradient solver.
class SpeechSubgradient : public SubgradProblem {
 public:
  SpeechSubgradient(const SpeechProblemSet &problems);

  void Solve(const SubgradInfo &info, 
             SubgradResult *result);
  
  // Update the weights with vector.
  void Update(const DataPoint &data_point, double alpha);

  const vector<DataPoint> &centers() {
    return best_centers_;
  }

  void set_ball_holder(const BallHolder *ball_holder) {
    ball_holder_ = ball_holder;
  }

  // MPLP stuff.
  double MPLPAlignRound(int problem_num);
  double MPLPClusterRound();
  void MPLPRound(int round);
 private:

  double DualProposal(SpeechSolution *solution) const;
  double Primal(vector<DataPoint > *centroids);
  void SetReparameterization();
  void SetReparameterization2();

  double HiddenDualProposal(SpeechSolution *solution);
  double HiddenDualUnaryProposal(vector<vector<int> > *vars);
  vector<vector<vector<double> > > *MPLPDiff(const vector<vector<int> > &a, 
                                             const vector<vector<int> > &b) const;

  void MPLPAugment(vector<vector<vector<double> > > *weights, 
                   const vector<vector<vector<double> > > &augment,
                   double rate);

  double MPLPSubgradient(double rate);
  
  // The information of the underlying speech problem
  const SpeechProblemSet &problems_;
  const ClusterSet &cluster_problems_;

  // The decomposition of the hmm part of the problem.
  vector<HMMSolver *> hmm_solvers_;

  // The decomposition of the hmm part of the problem.
  HiddenSolver *hidden_solver_;

  // Precomputed terms for solvers.
  vector<DistanceHolder *> distance_holders_;

  // The best mean value seen so far.
  double best_means_;
  vector<DataPoint> best_centers_;

  // The number of types.
  int num_features_;
  const BallHolder *ball_holder_;

  // Subgrad reparameterizations.
  vector<vector<vector<double> > > *hmm_reparameterization_;
  vector<vector<vector<double> > > *hidden_reparameterization_;

  // For mplp.
  vector<vector<vector<double> > > *hmm_reparameterization2_;
  vector<vector<vector<double> > > *hidden_reparameterization2_;

  // 
  vector<vector<vector<double> > > *delta_hmm_;
  vector<vector<vector<double> > > *delta_hidden_;

  // Best primal value seen so far.
  double best_primal_value_;
};

#endif
