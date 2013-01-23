#include "speech_subgrad.h"
#include "speech_kmeans.h"
#include <time.h>
#include <iostream>
#include <fstream>
using namespace std;
#define RECENTER 1

SpeechSubgradient::SpeechSubgradient(const SpeechProblemSet &problems):
  problems_(problems), cluster_problems_(problems.MakeClusterSet()),
  best_means_(INF) {
  total_subgrad_ = 0;
  recenter_solvers_.resize(problems.utterance_size());
  for (int index = 0; index < problems.utterance_size(); ++index) {
    distance_holders_.push_back(problems.MakeDistances(index));
    hmm_solvers_.push_back(new HMMViterbiSolver(cluster_problems_.problem(index), 
                                                *distance_holders_[index]));
    for (int i = 0; i < cluster_problems_.problem(index).num_states; ++i) {
      int type = cluster_problems_.problem(index).MapState(i);
      recenter_solvers_[index].push_back(new RecenterSolver(type, cluster_problems_.num_modes()));
    }
  }

  for (int type = 0; type < problems.num_types(); ++type) {
    hop_solvers_.push_back(new HOPSolver(type, cluster_problems_.num_modes()));
    kmedian_solvers_.push_back(new KMediansSolver(type, problems, cluster_problems_));
  }
  hidden_solver_ = new HiddenSolver(cluster_problems_);

  // MPLP variables
  hidden_reparameterization_ = cluster_problems_.CreateReparameterization();
  hmm_reparameterization_ = cluster_problems_.CreateReparameterization();
  hidden_reparameterization2_ = cluster_problems_.CreateReparameterization();
  hmm_reparameterization2_ = cluster_problems_.CreateReparameterization();
  delta_hmm_ = cluster_problems_.CreateReparameterization();
  delta_hidden_ = cluster_problems_.CreateReparameterization();  

  recenter_reparameterization_ = cluster_problems_.CreateReparameterization3();
  hop_reparameterization_ = cluster_problems_.CreateReparameterization2();
  recenter_reparameterization2_ = cluster_problems_.CreateReparameterization3();
  hop_reparameterization2_ = cluster_problems_.CreateReparameterization2();
  delta_hop_ = cluster_problems_.CreateReparameterization2();
  delta_recenter_ = cluster_problems_.CreateReparameterization3();  

  // Set random for delta hop.
  
  for (int loc_index = 0; loc_index < cluster_problems_.locations(); ++loc_index) {
    const StateLocation &loc = cluster_problems_.location(loc_index);

  // for (int problem_id = 0; problem_id < cluster_problems_.problems_size(); ++problem_id) {
  
  //   const ClusterProblem &prob = cluster_problems_.problem(problem_id);
  //   for (int state = 0; state < prob.num_states; ++state) {
  //     int type = prob.MapState(state);
    for (int hidden = 0; hidden < cluster_problems_.num_hidden(loc.type); ++hidden) {
      (*delta_hop_)[loc.type][hidden][1] = 10 * rand() / (float) RAND_MAX;
      (*hop_reparameterization2_)[loc.type][hidden][1] = 
        -(*delta_hop_)[loc.type][hidden][1];
      (*recenter_reparameterization_)[loc.problem][loc.state][hidden][1] = 
        (*delta_hop_)[loc.type][hidden][1];
    }
  }
  best_primal_value_ = INF;
}

void SpeechSubgradient::SetReparameterization() {
  for (uint problem = 0; problem < hmm_solvers_.size(); ++problem) {
    hmm_solvers_[problem]->set_reparameterization(hmm_reparameterization_->problem(problem));
    for (int i = 0; i < cluster_problems_.problem(problem).num_states; ++i) {
      recenter_solvers_[problem][i]->
        set_reparameterization(hidden_reparameterization_->state(problem, i),
                               &(*recenter_reparameterization_)[problem][i]);
    }
  }
  for (int type = 0; type < cluster_problems_.num_types(); ++type) {
    hop_solvers_[type]->set_reparameterization(&(*hop_reparameterization_)[type]);
    kmedian_solvers_[type]->set_reparameterization(&(*hidden_reparameterization_));
  }
  //hidden_solver_->set_reparameterization(hidden_reparameterization_);
}

void SpeechSubgradient::SetReparameterization2() {
  for (uint problem = 0; problem < hmm_solvers_.size(); ++problem) {
    hmm_solvers_[problem]->set_reparameterization(hmm_reparameterization2_->problem(problem));
    for (int i = 0; i < cluster_problems_.problem(problem).num_states; ++i) {
      recenter_solvers_[problem][i]->
        set_reparameterization(hidden_reparameterization2_->state(problem, i),
                               &(*recenter_reparameterization2_)[problem][i]);
    }
  }
  for (int type = 0; type < cluster_problems_.num_types(); ++type) {
    hop_solvers_[type]->
      set_reparameterization(&(*hop_reparameterization2_)[type]);
    kmedian_solvers_[type]->set_reparameterization(&(*hidden_reparameterization2_));
  }

  //hidden_solver_->set_reparameterization(hidden_reparameterization2_);
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

  if (false) {
    stringstream buf;
    buf << "/tmp/last_solution" << round;
    fstream output(buf.str().c_str(), ios::out | ios::binary);
    speech::SpeechSolution solution;
    dual_proposal->ToProtobuf(solution, problems_);
    solution.SerializeToOstream(&output);
    output.close();
  }
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
  SetReparameterization2();
  SpeechSolution solution(cluster_problems_);
  double dual = 0.0;
  dual += DualProposal(&solution);
  dual += HiddenDualProposal(&solution);
  vector<vector<int> > hmm = solution.AlignmentAssignments();
  vector<vector<int> > cluster;

  if (!RECENTER) {
    cluster = solution.ClusterAssignments();
  } else if (RECENTER) {
    cluster.resize(cluster_problems_.problems_size());
    for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
      cluster[u].resize(cluster_problems_.problem(u).num_states);
      for (int i = 0; i < cluster_problems_.problem(u).num_states; ++i) { 
        recenter_solvers_[u][i]->Solve();
        cluster[u][i] = recenter_solvers_[u][i]->SegmentCenter();
      }
    }
  }
  
  vector<vector<int> > unary;
  dual += HiddenDualUnaryProposal(&unary);
  Reparameterization *hmm_diff = 
    MPLPDiff(hmm, unary);

  Reparameterization *cluster_diff = 
    MPLPDiff(cluster, unary);

  MPLPAugment(hmm_reparameterization2_, *hmm_diff, rate);
  MPLPAugment(hidden_reparameterization2_, *cluster_diff, rate);
  
  // if (RECENTER) {
  //   for (int type = 0; type < cluster_problems_.num_types(); ++type) {
  //     cluster_dual += hop_solvers_[type]->Solve();
  //   }

  //   vector<vector<int> > unary;
  //   dual += HiddenDualUnaryProposal(&unary);
  //   vector<vector<vector<double > > > *recenter_diff = 
  //     MPLPDiff(recenter, unary);

  //   vector<vector<vector<double > > > *hop_diff = 
  //     MPLPDiff(hop, unary);

  //   MPLPAugment2(recenter_reparameterization2_, *recenter_diff, rate);
  //   MPLPAugment2(hop_reparameterization2_, *hop_diff, rate);
  // }
  for (int loc_index = 0; loc_index < cluster_problems_.locations(); ++loc_index) {
    const StateLocation &loc = cluster_problems_.location(loc_index);
    for (int h = 0; h < cluster_problems_.num_hidden(loc.type); ++h) {
      delta_hmm_->set(loc, h, -hmm_reparameterization2_->get(loc, h));
      delta_hidden_->set(loc, h, -hidden_reparameterization2_->get(loc, h));
      hmm_reparameterization_->set(loc, h,  
                                   delta_hidden_->get(loc, h));
      hidden_reparameterization_->set(loc, h, 
                                      delta_hmm_->get(loc, h));
    }
  }

  SetReparameterization();
  delete hmm_diff;
  delete cluster_diff;
  return dual;
}

void SpeechSubgradient::CheckAlignRound(int u) {
  SetReparameterization2();
  SpeechAlignment alignment;
  double temp = hmm_solvers_[u]->Solve(&alignment);
  cerr << "TEST: check test: " << temp << endl; 
  assert(fabs(temp) < 1e-4);
  SetReparameterization();
}

double SpeechSubgradient::MPLPAlignRound(int problem_num, 
                                         SpeechSolution *dual_solution) {
  int u = problem_num;
  vector<vector<double> > max_marginals;
  const ClusterProblem &problem = cluster_problems_.problem(u);  

  // Find the max-marginal values.
  SpeechAlignment *alignment = dual_solution->mutable_alignment(problem_num);
  double score = hmm_solvers_[u]->MaxMarginals(&max_marginals, alignment);

  // Reparameterize the distribution based on max-marginals.
  for (int i = 0; i < problem.num_states; ++i) {
    int type = problem.MapState(i);
    for (int hidden = 0; hidden < cluster_problems_.num_hidden(type); ++hidden) {
      delta_hmm_->data[u][i][hidden] = (-delta_hidden_->get(u, i, hidden)) + 
        (1.0 / problem.num_states) * (max_marginals[i][hidden]);      
    }
  }
  // Propogate the parameters.
  for (int i = 0; i < problem.num_states; ++i) {
    int type = problem.MapState(i);
    for (int hidden = 0; hidden < cluster_problems_.num_hidden(type); ++hidden) {
      hidden_reparameterization_->data[u][i][hidden] = delta_hmm_->get(u, i, hidden);
      hmm_reparameterization2_->data[u][i][hidden] = -delta_hmm_->get(u, i, hidden);
    }
  }
  if (false) {
    CheckAlignRound(u);
  }
  return score;
}

void SpeechSubgradient::CheckCountRound() {
  SetReparameterization2();
  double temp = 0.0;
  for (int type = 0; type < cluster_problems_.num_types(); ++type) {
    temp += hop_solvers_[type]->Solve();
  }
  cerr << "TEST: Count check test: " << temp << endl; 
  assert(fabs(temp) < 1e-4);
  SetReparameterization();
}

double SpeechSubgradient::MPLPCountRound() {
  double score = 0.0;
  int num_modes = cluster_problems_.num_modes();
  int num_hidden = cluster_problems_.num_hidden(0);
  SetReparameterization();

  for (int type = 0; type < cluster_problems_.num_types(); ++type) {
    vector<vector<double> > center_mu;
    score += hop_solvers_[type]->MaxMarginals(&center_mu);
    for (int hidden = 0; hidden < num_hidden; ++hidden) {
      // double sum_incoming = 0.0; 
      // for (int j = 0; j < problems_.type_occurence_size(type); ++j) {
      //   const StateLocation &loc = problems_.type_occurence(type, j);
      //   sum_incoming += (*delta_recenter_)[loc.problem][loc.state][hidden][1];
      // }
      //   (*recenter_reparameterization_)[loc.problem][loc.state][hidden][1] -= 
      //     (*delta_hop_)[type][hidden][1];
      // }
      (*delta_hop_)[type][hidden][1] = 
        -((*hop_reparameterization_)[type][hidden][1]) +
        (1.0 / num_modes) * (center_mu[hidden][1]);
      // for (int j = 0; j < problems_.type_occurence_size(type); ++j) {
      //   const StateLocation &loc = problems_.type_occurence(type, j);
      //   (*recenter_reparameterization_)[loc.problem][loc.state][hidden][1] += 
      //     (*delta_hop_)[type][hidden][1];
      // }
    }
  }
  for (int type = 0; type < cluster_problems_.num_types(); ++type) {
    for (int hidden = 0; hidden < num_hidden; ++hidden) {
      (*hop_reparameterization2_)[type][hidden][1] = 
        -(*delta_hop_)[type][hidden][1];
    }
  }
  //CheckCountRound();
  return score;
}

void SpeechSubgradient::CheckRecenter(int problem, int i) {
  SetReparameterization2();
  // vector<double> segment_mu;
  // vector<vector<double> > center_mu;
  //recenter_solvers_[problem][i]->MaxMarginals(&segment_mu, &center_mu);
  double temp = recenter_solvers_[problem][i]->Solve();
  cerr << "TEST: " << temp << endl;
  //cerr << problem << " " << i << " " << temp << endl; 
  assert(fabs(temp) < 1e-4);
  SetReparameterization();
}

double SpeechSubgradient::MPLPRecenterRound(int problem, int i) {
  // Resize the max marginal set. 
  //clock_t start = clock();
  double score = 0.0;
  const ClusterProblem &prob = cluster_problems_.problem(problem);  
  int type = prob.MapState(i);
  for (int hidden = 0; 
       hidden < cluster_problems_.num_hidden(type); 
       ++hidden) {
    
    (*recenter_reparameterization_)[problem][i][hidden][1] = 
      (*hop_reparameterization_)[type][hidden][1] 
      + (*delta_hop_)[type][hidden][1]
      - (*delta_recenter_)[problem][i][hidden][1];
    //double sum_incoming = 0.0;
    // for (int j = 0; j < problems_.type_occurence_size(type); ++j) {
    //   const StateLocation &loc = problems_.type_occurence(type, j);
    //   sum_incoming += (*delta_recenter_)[loc.problem][loc.state][hidden][b];
    // }
    // assert(fabs((*hop_reparameterization_)[type][hidden][b] 
    //             - sum_incoming) < 1e-4);
  }
  SetReparameterization();
  
  vector<double> segment_mu;
  vector<vector<double> > center_mu;
  double local_score = 
    recenter_solvers_[problem][i]->MaxMarginals(&segment_mu, &center_mu);
  //clock_t start2 = clock();
  score += local_score;
  int num_hidden = cluster_problems_.num_hidden(type);
  int num_modes = cluster_problems_.num_modes();
  for (int hidden = 0; hidden < num_hidden; ++hidden) {
    delta_hidden_->data[problem][i][hidden] = 
      (-delta_hmm_->get(problem, i, hidden)) + 
      (1.0 / (1 + num_modes)) * (segment_mu[hidden]);

    // remove old deltas. 
    (*hop_reparameterization_)[type][hidden][1] -=
      (*delta_recenter_)[problem][i][hidden][1];
      
    (*delta_recenter_)[problem][i][hidden][1] = 
      (-(*recenter_reparameterization_)[problem][i][hidden][1]) +
      (1.0 / (1 + num_modes)) * (center_mu[hidden][1]);
    
    // add new deltas. 
    (*hop_reparameterization_)[type][hidden][1] +=
      (*delta_recenter_)[problem][i][hidden][1];
  }

  for (int hidden = 0; hidden < num_hidden; ++hidden) {
    hmm_reparameterization_->data[problem][i][hidden] = 
      delta_hidden_->get(problem, i, hidden);
    hidden_reparameterization2_->data[problem][i][hidden] = 
      -delta_hidden_->get(problem, i, hidden);
    
    (*recenter_reparameterization2_)[problem][i][hidden][1] = 
      -(*delta_recenter_)[problem][i][hidden][1];
  }
  //clock_t start3 = clock();
  //CheckRecenter(problem, i);
  //cerr << "Times: " << clock() - start << " " << clock() - start2  << " " << clock() - start3<< endl;
  return score;
}

void SpeechSubgradient::CheckKMedians(int type) {
  SetReparameterization2();
  double temp = kmedian_solvers_[type]->Solve();
  cerr << "TEST: " << temp << endl;
  assert(fabs(temp) < 1e-4);
  SetReparameterization();
}

double SpeechSubgradient::MPLPKMediansRound(int type) {
  // Resize the max marginal set. 
  vector<double>  max_marginals;
  double score = kmedian_solvers_[type]->MaxMarginals(&max_marginals);

  int total_states =  problems_.type_occurence_size(type);
  int num_hidden = cluster_problems_.num_hidden(0);
  // Reparameterize the distribution based on max-marginals.
  for (int j = 0; j < problems_.type_occurence_size(type); ++j) {
    const StateLocation &loc = problems_.type_occurence(type, j);
    for (int hidden = 0; hidden < num_hidden; ++hidden) {
      delta_hidden_->set(loc, hidden,
                         (-delta_hmm_->get(loc, hidden)) + 
                         (1.0 / total_states) * (max_marginals[hidden]));
    }
  }

  for (int j = 0; j < problems_.type_occurence_size(type); ++j) {
    const StateLocation &loc = problems_.type_occurence(type, j);
    for (int hidden = 0; hidden < cluster_problems_.num_hidden(type); ++hidden) {
      hmm_reparameterization_->set(loc, hidden,
                                   delta_hidden_->get(loc, hidden));
      hidden_reparameterization2_->set(loc, hidden,  -delta_hidden_->get(loc, hidden));
    }
  }
  CheckKMedians(type);
  return score;
}


void SpeechSubgradient::DescentRound(SpeechSolution *dual_solution) {
  double score;
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    //for (int u = 0; u < 5; ++u) {
    clock_t start = clock();
    score = MPLPAlignRound(u, dual_solution);
    cerr << "TIME: Align: " << score << " " << clock() - start  << endl;

    if (RECENTER) {
      start = clock();
      for (int i = 0; i < cluster_problems_.problem(u).num_states; ++i) {
        score += MPLPRecenterRound(u, i);
      }
      cerr << "TIME: Recenter: " << score << " " << clock() - start  << endl;
      start = clock();
      score += MPLPCountRound();
      cerr << "TIME: Count: " << clock() - start  << endl;
    }
  }
  if (!RECENTER) {
    for (int l = 0; l < cluster_problems_.num_types(); ++l) {
      if (problems_.type_occurence_size(l) == 0 ) continue;
      MPLPKMediansRound(l);
    }
  }
}

double SpeechSubgradient::ComputeCompleteDual(SpeechSolution *solution) {
  double dual_value = 0.0;
  /*
  double cluster_dual = 0.0;

  if (RECENTER) {
    for (int type = 0; type < cluster_problems_.num_types(); ++type) {
      cluster_dual += hop_solvers_[type]->Solve();
      for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
        solution->set_type_to_hidden(type, mode, 
                                     hop_solvers_[type]->get_mode(mode));
      }
    }
    cerr << "DUAL sum is " << cluster_dual << endl;
    for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
      for (int i = 0; i < cluster_problems_.problem(u).num_states; ++i) { 
        cluster_dual += recenter_solvers_[u][i]->Solve();
      }
    }
  } else {
    for (int type = 0; type < cluster_problems_.num_types(); ++type) {
      cluster_dual += kmedian_solvers_[type]->Solve();
      for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
        solution->set_type_to_hidden(type, mode, 
                                     kmedian_solvers_[type]->get_mode(mode));
      }
    }    
  }
  dual_value += cluster_dual;
  cerr << "DUAL sum is " << dual_value << endl;
  dual_value += DualProposal(solution);
  cerr << "DUAL sum is " << dual_value << endl;
  assert(fabs(dual_value) < 1e-4); */
  
  if (RECENTER) {  
    dual_value += ComputeDualRecenter(solution);
  }
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

double SpeechSubgradient::ComputeDualRecenter(SpeechSolution *solution) {
  int num_modes = cluster_problems_.num_modes();
  double dual_value = 0.0;
  for (int type = 0; type < cluster_problems_.num_types(); ++type) {
    vector<double> best_score(num_modes, INF);
    vector<int> best_hidden(num_modes, -1);
    for (int mode = 0; mode < num_modes; ++mode) {
      for (int hidden = 0; 
           hidden < cluster_problems_.num_hidden(type); 
           ++hidden) {
        bool found = false;
        for (int mode2 = 0; mode2 < mode; ++mode2) {
          if (best_hidden[mode2] == hidden) {
            found = true;
            break;
          }
        }
        if (found) continue;
        double score_on = (*delta_hop_)[type][hidden][1] 
          + (*hop_reparameterization_)[type][hidden][1];
        if (score_on < best_score[mode]) { 
          best_score[mode] = score_on;
          best_hidden[mode] = hidden;
        } 
      }
      dual_value += best_score[mode];
      //solution->set_type_to_hidden(type, mode, best_hidden[mode]);
    }
  }
  return dual_value;
}

// Runs a round of MPLP. 
void SpeechSubgradient::MPLPRound(int round) {
  vector<DataPoint> centroids;
  SetReparameterization();
  SpeechSolution *dual_solution = new SpeechSolution(cluster_problems_);
  DescentRound(dual_solution);
  SetReparameterization2();

  double dual_value = ComputeCompleteDual(dual_solution);

  // Actually centroids here.

  //vector<DataPoint> centroids;
  //double primal_value = Primal(dual_solution, round, &centroids);
  Primal(dual_solution, round, &centroids);
  vector<vector<DataPoint> >centers; 
  centers.resize(cluster_problems_.num_modes());
  for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
    centers[mode].resize(cluster_problems_.num_types());
    for (int type = 0; type < cluster_problems_.num_types(); ++type) {
      centers[mode][type] = 
        dual_solution->TypeToSpecial(type, 0);
      //problems_.center(dual_solution->TypeToHidden(type, mode)).point();        
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

  cerr << "SCORE: Final primal value " << best_primal_value_ << " " 
       << primal_value << endl;
  cerr << "SCORE: Final Dual value: " << dual_value << endl;

  SetReparameterization();
  delete dual_solution;
}
