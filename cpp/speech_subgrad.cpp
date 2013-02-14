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
    hmm_astar_solvers_(problems.utterance_size()),
    distance_holders_(problems.utterance_size()) {
  for (int index = 0; index < problems.utterance_size(); ++index) {
    const ClusterProblem &problem = cluster_problems_.problem(index);
    distance_holders_[index] = problems.MakeDistances(index);
    hmm_solvers_[index] = 
      new HMMViterbiSolver(problem, *distance_holders_[index]);
    hmm_astar_solvers_[index] = 
      new HMMAStarSolver(problem, *distance_holders_[index], index);

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
  best_dual_value_ = -INF;
}

void SpeechSubgradient::SetMPLPUpdateParams() {
  for (uint problem = 0; problem < hmm_solvers_.size(); ++problem) {
    hmm_solvers_[problem]->set_reparameterization(hmm_reparameterization_->problem(problem));
    hmm_astar_solvers_[problem]->set_reparameterization(hmm_reparameterization_->problem(problem));
  }
  hidden_solver_->set_reparameterization(hidden_reparameterization_);
}

void SpeechSubgradient::SetNaturalParams() {
  for (uint problem = 0; problem < hmm_solvers_.size(); ++problem) {
    hmm_solvers_[problem]->set_reparameterization(hmm_reparameterization2_->problem(problem));
    hmm_astar_solvers_[problem]->set_reparameterization(hmm_reparameterization2_->problem(problem));
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
    dual += hmm_astar_solvers_[u]->Solve(alignment, true, *hidden_solver_, *delta_hmm_, *delta_hidden_, 
                                         best_primal_value_);
  }
  return dual;
}

// double SpeechSubgradient::HiddenDualProposal(SpeechSolution *solution) {
//   double dual = 0.0;
//   for (int type = 0; type < problems_.num_types(); ++type) {
//     dual = kmedian_solvers_[type]->Solve();  
//     for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
//       solution->set_type_to_hidden(type, mode, 
//                                    kmedian_solvers_[type]->get_mode(mode));
//     }
//   }
//   return dual;
// }

double SpeechSubgradient::HiddenDualProposal(SpeechSolution *solution) {
  vector<int> default_centers;
  double dual = hidden_solver_->Solve(default_centers);  
  for (int type = 0; type < problems_.num_types(); ++type) {
    for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
      solution->set_type_to_hidden(type, mode, 
                                   hidden_solver_->TypeToHidden(type, mode));
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
  Reparameterization *diff  = cluster_problems_.CreateReparameterization();
  for (int loc_index = 0; loc_index < cluster_problems_.locations(); ++loc_index) {
    const StateLocation &loc = cluster_problems_.location(loc_index);
    diff->augment(loc, a[loc.problem][loc.state], 1.0);
    diff->augment(loc, b[loc.problem][loc.state], -1.0);
  }
  return diff;
}

void SpeechSubgradient::MPLPAugment(Reparameterization *weights, 
                                    const Reparameterization &augment,
                                    double rate) {
  for (int loc_index = 0; loc_index < cluster_problems_.locations(); ++loc_index) {
    const StateLocation &loc = cluster_problems_.location(loc_index);
    for (int h = 0; h < num_hidden(loc); ++h) {
      weights->augment(loc, h, rate * augment.get(loc, h));
    }
  }
}

double SpeechSubgradient::MPLPSubgradient(double rate, 
                                          SpeechSolution *solution) {
  // Compute the dual proposal.
  SetNaturalParams();
  //SpeechSolution solution(cluster_problems_);
  double dual = 0.0;
  dual += DualProposal(solution);
  dual += HiddenDualProposal(solution);
  vector<vector<int> > hmm = solution->AlignmentAssignments();
  vector<vector<int> > cluster;
  cluster = solution->ClusterAssignments();
  
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
      hidden_reparameterization_->set(loc, hidden, delta_hmm_->get(loc, hidden));
      hmm_reparameterization2_->set(loc, hidden, -delta_hmm_->get(loc, hidden));
    }
  }
  if (CHECK) CheckAlignRound(u);
  return score;
}


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
    vector<int> default_centers;
    double temp = hidden_solver_->Solve(default_centers);
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
    cerr << "TIME: Align round: " << u << " " << score << " " << clock() - start  << endl;
  }
  clock_t start = clock();
  MPLPClusterRound();
  cerr << "TIME: Cluster round: " << clock() - start  << endl;
}

double SpeechSubgradient::ComputeCompleteDual(SpeechSolution *solution) {
  double dual_value = 0.0;
  dual_value += ComputeDualSegment(solution);  
  // SpeechSolution unary_solution(cluster_problems_);  
  // for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
  //   SpeechAlignment *align = unary_solution.mutable_alignment(u);
  //   dual_value += hmm_solvers_[u]->Solve(align);
  //  }
  //dual_value += hidden_solver_->Solve();
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
      StateLocation loc(u, i, type);
      for (int hidden = 0; hidden < num_hidden(loc); ++hidden) {
        double trial = 
          delta_hmm_->get(loc, hidden) + delta_hidden_->get(loc, hidden);
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

void SpeechSubgradient::MPLPRunSubgrad(int round) {
  for (int i = 0; i < 1; ++i) {
    vector<DataPoint> centroids;
    SpeechSolution *dual_solution = new SpeechSolution(cluster_problems_);    
    double dual = MPLPSubgradient(10.0 / (float)(i + round + 1), dual_solution);
    
    SetNaturalParams();
    double dual_value = ComputeCompleteDual(dual_solution);
    cerr << "SCORE: Subgrad dual is " << i << " " << dual << " " << dual_value << endl;
    double primal_value = Primal(dual_solution, i, &centroids);
    cerr << "SCORE: Subgrad primal is " << " " << primal_value << endl;
    if (primal_value < best_primal_value_) {
      best_primal_value_ = primal_value;
    } 
    if (dual > best_dual_value_) {
      best_dual_value_ = dual;
    } 

    //if ((round + 1) % 10 ==  0) {
    //LocalSearch(dual_solution);
    //}
    cerr << "GAP: " << best_primal_value_ - best_dual_value_ << " " <<  best_primal_value_ << " " << best_dual_value_ << endl;
    cerr << "SCORE: Final primal value " 
         << best_primal_value_ << endl;

    cerr << "SCORE: Final dual value " 
         << best_dual_value_ << endl;

    delete dual_solution;
  }
}

void SpeechSubgradient::BeamSearch() {
  SpeechSolution solution(cluster_problems_);
  double dual = 0.0;
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    SpeechAlignment *alignment = solution.mutable_alignment(u);
    dual += hmm_astar_solvers_[u]->Solve(alignment, false, *hidden_solver_, *delta_hmm_, *delta_hidden_, best_primal_value_);
  }
  vector<DataPoint> centroids;
  double primal_value = Primal(&solution, 1, &centroids);
  cerr << "Beam " << primal_value << " " << dual << endl;
  if (dual < best_primal_value_ ) {
    best_primal_value_ = dual;
  } 

}

void SpeechSubgradient::LocalSearch(SpeechSolution *dual_solution) {
  // Use the centroids to compute the best primal solution.
  vector<vector<DataPoint> > centers; 
  centers.resize(cluster_problems_.num_modes());
  for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
    centers[mode].resize(cluster_problems_.num_types());
    for (int type = 0; type < cluster_problems_.num_types(); ++type) {
      centers[mode][type] = dual_solution->TypeToSpecial(type, 0);
    }
  }
  double primal_value = 0.0;
  
  SpeechKMeans kmeans(problems_);
  kmeans.SetCenters(centers);
  kmeans.set_use_medians(true);
  primal_value = kmeans.Run(1);
  
  if (primal_value < best_primal_value_) {
    best_primal_value_ = primal_value;
    best_centers_ = centers;
  } 
}



// Runs a round of MPLP. 
void SpeechSubgradient::MPLPRound(int round) {
  if (round > 500) {
    // MPLPRunSubgrad(round);
    SetNaturalParams();
    SpeechSolution *dual_solution = new SpeechSolution(cluster_problems_);
    double dual_value = DualProposal(dual_solution);
    best_dual_value_ = dual_value;
    best_primal_value_ = dual_value;
    cerr << "SCORE: Final primal value " << best_primal_value_ << endl;
    cerr << "SCORE: Final Dual value: " << best_dual_value_ << endl;
    exit(0);
    return;
  }

  // Run a round of coordinate descent. 
  SetMPLPUpdateParams();
  SpeechSolution *dual_solution = new SpeechSolution(cluster_problems_);

  clock_t start = clock();
  MPLPDescentRound(dual_solution);
  cerr << "TIME: Descent: " << clock() - start  << endl;
  // Compute the current dual value. 
  SetNaturalParams();
  double dual_value = ComputeCompleteDual(dual_solution);

  // Compute the primal solution. 
  vector<DataPoint> centroids;
  double primal_value = Primal(dual_solution, round, &centroids);
  if (primal_value < best_primal_value_) best_primal_value_ = primal_value;
  if (dual_value > best_dual_value_) best_dual_value_ = dual_value;
  // if ((round + 1) % 25 == 0) {
  //   LocalSearch(dual_solution);
  // }
  if (round > 30 && (round + 1) % 50 == 0) {
    BeamSearch();
  }
  // Log the dual and primal values.
  cerr << "SCORE: Final primal value " 
       << best_primal_value_ << " " << primal_value << endl;
  cerr << "SCORE: Final Dual value: " << best_dual_value_ << endl;

  cerr << "GAP: " << round << " " << abs(best_primal_value_ - dual_value) << " " <<  best_primal_value_ << " " << dual_value;

  // Reset the parameterization.
  SetMPLPUpdateParams();
  delete dual_solution;
}
