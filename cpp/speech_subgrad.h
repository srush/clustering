#ifndef SPEECH_SUBGRAD_H
#define SPEECH_SUBGRAD_H

//#include "cluster_subgrad.h"
#include "hidden_solver.h"
#include "hmm_solver.h"
#include "hop_solver.h"
#include "recenter_solver.h"
#include "hmm_viterbi_solver.h"
#include "hmm_astar_solver.h"
#include "subgrad.h"
#include "speech_problem.h"
#include "hidden_kmedian_solver.h"


// Setup for the speech subgradient solver.
class SpeechSubgradient {//: public SubgradProblem {
 public:
  SpeechSubgradient(const SpeechProblemSet &problems);

  const vector<vector<DataPoint> > &centers() {
    return best_centers_;
  }

  // MPLP stuff.
  double MPLPAlignRound(int problem_num, SpeechSolution *solution);
  double MPLPClusterRound();
  double MPLPCountRound();

  void MPLPRound(int round);

 private:
  int num_hidden(const StateLocation &loc) {
    return cluster_problems_.num_hidden(loc.type);
  }

  double DualProposal(SpeechSolution *solution) const;
  double Primal(SpeechSolution *solution, int round, vector<DataPoint > *centroids);
  void SetMPLPUpdateParams();
  void SetNaturalParams();

  double HiddenDualProposal(SpeechSolution *solution);
  double HiddenDualUnaryProposal(vector<vector<int> > *vars);
  Reparameterization *MPLPDiff(const vector<vector<int> > &a, 
                               const vector<vector<int> > &b) const;

  void MPLPAugment(Reparameterization *weights, 
                   const Reparameterization &augment,
                   double rate);

  double MPLPSubgradient(double rate, SpeechSolution *solution);
  void LocalSearch(SpeechSolution *dual_solution);

  // Run one round of coordinate descent.
  void MPLPDescentRound(SpeechSolution *solution);  

  // Compute the complete dual value. 
  double ComputeCompleteDual(SpeechSolution *solution);


  void MPLPRunSubgrad(int round);
  double ComputeDualSegment(SpeechSolution *solution);
  double ComputeDualRecenter(SpeechSolution *solution);
  void CheckKMedians(int type);
  void CheckAlignRound(int u);
  void CheckRecenter(int problem, int i);
  void CheckCountRound();
  double MPLPRecenterRound(const StateLocation &loc);
  double MPLPKMediansRound(int type);

  // The information of the underlying speech problem
  const SpeechProblemSet &problems_;
  const ClusterSet &cluster_problems_;

  // The decomposition of the hmm part of the problem.
  vector<HMMViterbiSolver *> hmm_solvers_;
  vector<HMMAStarSolver *> hmm_astar_solvers_;

  // The decomposition of the hmm part of the problem.
  HiddenSolver *hidden_solver_;

  vector<KMediansSolver * > kmedian_solvers_;

  // Precomputed terms for solvers.
  vector<ThinDistanceHolder *> distance_holders_;

  // The best centers seen so far.
  vector<vector<DataPoint> > best_centers_;

  // The number of types.
  int num_features_;

  // Subgrad reparameterizations.
  Reparameterization *hmm_reparameterization_;
  Reparameterization *hidden_reparameterization_;

  Reparameterization *hmm_reparameterization2_;
  Reparameterization *hidden_reparameterization2_;

  Reparameterization *delta_hmm_;
  Reparameterization *delta_hidden_;

  // Best primal value seen so far.
  double best_primal_value_;
};

#endif
