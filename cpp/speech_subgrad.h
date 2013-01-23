#ifndef SPEECH_SUBGRAD_H
#define SPEECH_SUBGRAD_H

//#include "cluster_subgrad.h"
#include "hidden_solver.h"
#include "hmm_solver.h"
#include "hop_solver.h"
#include "recenter_solver.h"
#include "hmm_viterbi_solver.h"
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
  //double MPLPClusterRound();
  double MPLPCountRound();

  void MPLPRound(int round);

 private:
  double DualProposal(SpeechSolution *solution) const;
  double Primal(SpeechSolution *solution, int round, vector<DataPoint > *centroids);
  void SetReparameterization();
  void SetReparameterization2();

  double HiddenDualProposal(SpeechSolution *solution);
  double HiddenDualUnaryProposal(vector<vector<int> > *vars);
  Reparameterization *MPLPDiff(const vector<vector<int> > &a, 
                               const vector<vector<int> > &b) const;

  void MPLPAugment(Reparameterization *weights, 
                   const Reparameterization &augment,
                   double rate);

  double MPLPSubgradient(double rate);

  // Run one round of coordinate descent.
  void DescentRound(SpeechSolution *solution);  
  double ComputeCompleteDual(SpeechSolution *solution);
  double ComputeDualSegment(SpeechSolution *solution);
  double ComputeDualRecenter(SpeechSolution *solution);
  void CheckKMedians(int type);
  void CheckAlignRound(int u);
  void CheckRecenter(int problem, int i);
  void CheckCountRound();
  double MPLPRecenterRound(int problem_num, int state);
  double MPLPKMediansRound(int type);

  // The information of the underlying speech problem
  const SpeechProblemSet &problems_;
  const ClusterSet &cluster_problems_;

  // The decomposition of the hmm part of the problem.
  vector<HMMViterbiSolver *> hmm_solvers_;

  // The decomposition of the hmm part of the problem.
  HiddenSolver *hidden_solver_;

  vector<KMediansSolver * > kmedian_solvers_;

  // Hop solver.
  vector<HOPSolver *> hop_solvers_;

  // Recenter solvers.
  vector<vector<RecenterSolver *> > recenter_solvers_;


  // Precomputed terms for solvers.
  vector<ThinDistanceHolder *> distance_holders_;

  // The best mean value seen so far.
  double best_means_;
  vector<vector<DataPoint> > best_centers_;

  // The number of types.
  int num_features_;

  int total_subgrad_;

  // Subgrad reparameterizations.
  Reparameterization *hmm_reparameterization_;
  Reparameterization *hidden_reparameterization_;

  Reparameterization *hmm_reparameterization2_;
  Reparameterization *hidden_reparameterization2_;

  Reparameterization *delta_hmm_;
  Reparameterization *delta_hidden_;

  /* vector<vector<vector<double> > > *hmm_reparameterization_; */
  /* vector<vector<vector<double> > > *hidden_reparameterization_; */

  // For mplp.
  /* vector<vector<vector<double> > > *hmm_reparameterization2_; */
  /* vector<vector<vector<double> > > *hidden_reparameterization2_; */

  // 
  /* vector<vector<vector<double> > > *delta_hmm_; */
  /* vector<vector<vector<double> > > *delta_hidden_; */

  // For 
  vector<vector<vector<vector<double> > > > *recenter_reparameterization_;
  vector<vector<vector<double> > > *hop_reparameterization_;

  vector<vector<vector<vector<double> > > > *recenter_reparameterization2_;
  vector<vector<vector<double> > > *hop_reparameterization2_;

  // 
  vector<vector<vector<vector<double> > > > *delta_recenter_;
  vector<vector<vector<double> > > *delta_hop_;

  // Best primal value seen so far.
  double best_primal_value_;
};

#endif
