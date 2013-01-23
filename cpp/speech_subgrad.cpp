#include "speech_subgrad.h"
#include "speech_kmeans.h"
#include <time.h>
#include <iostream>
#include <fstream>
using namespace std;
#define RECENTER 0
#define CHECK 0
SpeechSubgradient::SpeechSubgradient(const SpeechProblemSet &problems)
  : problems_(problems), 
    cluster_problems_(problems.MakeClusterSet()),
    hmm_solvers_(problems.utterance_size()),
    distance_holders_(problems.utterance_size()) {
  for (int index = 0; index < problems.utterance_size(); ++index) {
    const ClusterProblem &problem = cluster_problems_.problem(index);
    distance_holders_[index] = problems.MakeDistances(index);
    hmm_solvers_[index] = 
      new HMMViterbiSolver(problem, *distance_holders_[index]);
  }

  hidden_solver_ = new HiddenSolver(cluster_problems_);

  // MPLP variables
  hidden_reparameterization_ = cluster_problems_.CreateReparameterization();
  hmm_reparameterization_ = cluster_problems_.CreateReparameterization();
  hidden_reparameterization2_ = cluster_problems_.CreateReparameterization();
  hmm_reparameterization2_ = cluster_problems_.CreateReparameterization();
  delta_hmm_ = cluster_problems_.CreateReparameterization();
  delta_hidden_ = cluster_problems_.CreateReparameterization();  
  best_primal_value_ = INF;
}

void SpeechSubgradient::SetMPLPUpdateParams() {
  for (uint problem = 0; problem < hmm_solvers_.size(); ++problem) {
    hmm_solvers_[problem]->set_reparameterization(hmm_reparameterization_->problem(problem));
  }
  hidden_solver_->set_reparameterization(hidden_reparameterization_);
}

void SpeechSubgradient::SetNaturalParams() {
  for (uint problem = 0; problem < hmm_solvers_.size(); ++problem) {
    hmm_solvers_[problem]->set_reparameterization(hmm_reparameterization2_->problem(problem));
  }
  hidden_solver_->set_reparameterization(hidden_reparameterization2_);
}

double SpeechSubgradient::Primal(SpeechSolution *dual_proposal, 
                                 int round, 
                                 vector<DataPoint > *centroids) {

  double max_medians = problems_.MaximizeMedians(*dual_proposal, centroids);
  for (int type = 0; type < problems_.num_types(); ++type) {
    dual_proposal->set_type_to_special(type, 0, (*centroids)[type]);
  }
  
  double dual = 0.0;
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    dual += hmm_solvers_[u]->Rescore(u, *dual_proposal);
  }
  dual += hidden_solver_->Rescore(*dual_proposal);
  cerr << "SCORE: rescore is: " << dual << endl;
  return max_medians;
}

double SpeechSubgradient::DualProposal(SpeechSolution *solution) const {
  double dual = 0.0;
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    SpeechAlignment *alignment = solution->mutable_alignment(u);
    dual += hmm_solvers_[u]->Solve(alignment);
  }
  return dual;
}

double SpeechSubgradient::HiddenDualProposal(SpeechSolution *solution) {
  double dual = 0.0;
  for (int type = 0; type < problems_.num_types(); ++type) {
    dual = kmedian_solvers_[type]->Solve();  
    for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
      solution->set_type_to_hidden(type, mode, 
                                   kmedian_solvers_[type]->get_mode(mode));
    }
  }
  return dual;
}

double SpeechSubgradient::HiddenDualUnaryProposal(vector<vector<int> > *vars ) {
  double dual = 0.0;
  vars->resize(cluster_problems_.problems_size());
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    const ClusterProblem &problem = cluster_problems_.problem(u);
    (*vars)[u].resize(problem.num_states);
    for (int i = 0; i < problem.num_states; ++i) {
      int type = problem.MapState(i);
      StateLocation loc(u, i, type);
      double best_hidden = INF;
      for (int hidden = 0; 
           hidden < cluster_problems_.num_hidden(type); 
           ++hidden) {
        double trial = 
          delta_hmm_->get(loc, hidden) + delta_hidden_->get(loc, hidden);
        if (trial < best_hidden) {
          best_hidden = trial;
          (*vars)[u][i] = hidden;
        }
      }
      dual += best_hidden;
    }
  }
  return dual;
}

Reparameterization *SpeechSubgradient::MPLPDiff(const vector<vector<int> > &a, 
                                                const vector<vector<int> > &b) const {
  Reparameterization *diff  = 
    cluster_problems_.CreateReparameterization();
  
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    for (int i = 0; i < cluster_problems_.problem(u).num_states; ++i) {
      diff->data[u][i][a[u][i]] += 1.0;
      diff->data[u][i][b[u][i]] -= 1.0;
    }
  }
  return diff;
}

void SpeechSubgradient::MPLPAugment(Reparameterization *weights, 
                                    const Reparameterization &augment,
                                    double rate) {
  for (int loc_index = 0; loc_index < cluster_problems_.locations(); ++loc_index) {
    const StateLocation &loc = cluster_problems_.location(loc_index);
    for (int h = 0; h < cluster_problems_.num_hidden(loc.type); ++h) {
      weights->augment(loc, h, rate * augment.get(loc, h));
    }
  }
}

double SpeechSubgradient::MPLPSubgradient(double rate) {
  // Compute the dual proposal.
  SetNaturalParams();
  SpeechSolution solution(cluster_problems_);
  double dual = 0.0;
  dual += DualProposal(&solution);
  dual += HiddenDualProposal(&solution);
  vector<vector<int> > hmm = solution.AlignmentAssignments();
  vector<vector<int> > cluster;
  cluster = solution.ClusterAssignments();
  
  vector<vector<int> > unary;
  dual += HiddenDualUnaryProposal(&unary);
  Reparameterization *hmm_diff = MPLPDiff(hmm, unary);
  Reparameterization *cluster_diff = MPLPDiff(cluster, unary);

  MPLPAugment(hmm_reparameterization2_, *hmm_diff, rate);
  MPLPAugment(hidden_reparameterization2_, *cluster_diff, rate);
  int locations = cluster_problems_.locations();
  for (int loc_index = 0; loc_index < locations; ++loc_index) {
    const StateLocation &loc = cluster_problems_.location(loc_index);
    for (int h = 0; h < cluster_problems_.num_hidden(loc.type); ++h) {
      delta_hmm_->set(loc, h, -hmm_reparameterization2_->get(loc, h));
      delta_hidden_->set(loc, h, -hidden_reparameterization2_->get(loc, h));
      hmm_reparameterization_->set(loc, h, delta_hidden_->get(loc, h));
      hidden_reparameterization_->set(loc, h, delta_hmm_->get(loc, h));
    }
  }

  SetMPLPUpdateParams();
  delete hmm_diff;
  delete cluster_diff;
  return dual;
}

void SpeechSubgradient::CheckAlignRound(int u) {
  SetNaturalParams();
  SpeechAlignment alignment;
  double temp = hmm_solvers_[u]->Solve(&alignment);
  cerr << "TEST: check test: " << temp << endl; 
  assert(fabs(temp) < 1e-4);
  SetMPLPUpdateParams();
}

double SpeechSubgradient::MPLPAlignRound(int problem_num, 
                                         SpeechSolution *dual_solution) {
  int u = problem_num;
  const ClusterProblem &problem = cluster_problems_.problem(u);  
  int states = problem.num_states;


  // Find the max-marginal values.
  vector<vector<double> > max_marginals;
  SpeechAlignment *alignment = dual_solution->mutable_alignment(problem_num);
  double score = hmm_solvers_[u]->MaxMarginals(&max_marginals, alignment);

  // Reparameterize the distribution based on max-marginals.
  for (int i = 0; i < states; ++i) {
    int type = problem.MapState(i);
    StateLocation loc(u, i, type);
    for (int hidden = 0; hidden < num_hidden(loc); ++hidden) {
      delta_hmm_->set(loc, hidden, -delta_hidden_->get(loc, hidden) + 
                      (1.0 / states) * max_marginals[loc.state][hidden]);
    }
  }

  // Propagate the parameters.
  for (int i = 0; i < states; ++i) {
    int type = problem.MapState(i);
    StateLocation loc(u, i, type);
    for (int hidden = 0; hidden < num_hidden(loc); ++hidden) {
      hidden_reparameterization_->set(loc, hidden, 
                                      delta_hmm_->get(loc, hidden));
      hmm_reparameterization2_->set(loc, hidden, 
                                    -delta_hmm_->get(loc, hidden));
    }
  }
  if (CHECK) CheckAlignRound(u);
  return score;
}


// void SpeechSubgradient::CheckKMedians(int type) {
//   SetNaturalParams();
//   double temp = kmedian_solvers_[type]->Solve();
//   cerr << "TEST: " << temp << endl;
//   assert(fabs(temp) < 1e-4);
//   SetMPLPUpdateParams();
// }

double SpeechSubgradient::MPLPClusterRound() {
  // Resize the max marginal set. 
  Reparameterization *max_marginals  = cluster_problems_.CreateReparameterization();
  double score = hidden_solver_->MaxMarginals(max_marginals);
  int total_states = cluster_problems_.locations();

  // Reparameterize the distribution based on max-marginals.
  for (int index = 0; index < total_states; ++index) {
    StateLocation loc(cluster_problems_.location(index));
    for (int hidden = 0; hidden < num_hidden(loc); ++hidden) {
      delta_hidden_->set(loc, hidden, -delta_hmm_->get(loc, hidden) +
                         (1.0 / total_states) * max_marginals->get(loc, hidden));
    }
  }
  for (int index = 0; index < total_states; ++index) {
    StateLocation loc(cluster_problems_.location(index));
    for (int hidden = 0; hidden < num_hidden(loc); ++hidden) {
      hmm_reparameterization_->set(loc, hidden, delta_hidden_->get(loc, hidden));
      hidden_reparameterization2_->set(loc, hidden, -delta_hidden_->get(loc, hidden));
    }
  }

  if (CHECK) {
    SetNaturalParams();
    double temp = hidden_solver_->Solve();
    cerr << "temp test: " << temp << endl; 
    assert(fabs(temp) < 1e-4);
    SetMPLPUpdateParams();
  }
  delete max_marginals;
  return score;
}

void SpeechSubgradient::MPLPDescentRound(SpeechSolution *dual_solution) {
  double score;
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    // Run alignment (Viterbi) solver.
    clock_t start = clock();
    score = MPLPAlignRound(u, dual_solution);
    cerr << "TIME: Align: " << score << " " << clock() - start  << endl;
  }
  MPLPClusterRound();
}

double SpeechSubgradient::ComputeCompleteDual(SpeechSolution *solution) {
  double dual_value = 0.0;
  dual_value += ComputeDualSegment(solution);  
  return dual_value;
}

double SpeechSubgradient::ComputeDualSegment(SpeechSolution *solution) {
  double dual_value = 0.0;
  SpeechSolution unary_solution(cluster_problems_);  
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    const ClusterProblem &problem = cluster_problems_.problem(u);  
    SpeechAlignment *align = unary_solution.mutable_alignment(u);
    vector<int> *state_hidden = align->mutable_hidden_alignment();
    state_hidden->resize(problem.num_states);
    for (int i = 0; i < problem.num_states; ++i) {
      double best_hidden = INF;
      int type = problem.MapState(i);
      for (int hidden = 0; hidden < cluster_problems_.num_hidden(type); ++hidden) {
        double trial = 
          delta_hmm_->get(u, i, hidden) + delta_hidden_->get(u, i, hidden);
        if (trial < best_hidden) {
          best_hidden = trial;
          (*state_hidden)[i] = hidden;
        }
      }
      dual_value += best_hidden;
    }
  }
  return dual_value;
}

// Runs a round of MPLP. 
void SpeechSubgradient::MPLPRound(int round) {

  // Run a round of coordinate descent. 
  SetMPLPUpdateParams();
  SpeechSolution *dual_solution = new SpeechSolution(cluster_problems_);
  MPLPDescentRound(dual_solution);

  // Compute the current dual value. 
  SetNaturalParams();
  double dual_value = ComputeCompleteDual(dual_solution);

  // Compute the primal solution. 
  vector<DataPoint> centroids;
  Primal(dual_solution, round, &centroids);

  // Use the centroids to compute the best primal solution.
  vector<vector<DataPoint> >centers; 
  centers.resize(cluster_problems_.num_modes());
  for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
    centers[mode].resize(cluster_problems_.num_types());
    for (int type = 0; type < cluster_problems_.num_types(); ++type) {
      centers[mode][type] = dual_solution->TypeToSpecial(type, 0);
    }
  }
  double primal_value = 0.0;
  if (round % 3 == 0) { 
    SpeechKMeans kmeans(problems_);
    kmeans.SetCenters(centers);
    kmeans.set_use_medians(true);
    primal_value = kmeans.Run(2);

    if (primal_value < best_primal_value_) {
      best_primal_value_ = primal_value;
      best_centers_ = centers;
    } 
  }

  // Log the dual and primal values.
  cerr << "SCORE: Final primal value " 
       << best_primal_value_ << " " 
       << primal_value << endl;
  cerr << "SCORE: Final Dual value: " << dual_value << endl;

  // Reset the parameterization.
  SetMPLPUpdateParams();
  delete dual_solution;
}
