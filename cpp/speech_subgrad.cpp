#include "speech_subgrad.h"
#include "speech_kmeans.h"
#include <time.h>
#include <iostream>
#include <fstream>
using namespace std;

SpeechSubgradient::SpeechSubgradient(const SpeechProblemSet &problems):
  problems_(problems), cluster_problems_(problems.MakeClusterSet()),
  best_means_(INF), ball_holder_(NULL) {
  for (int index = 0; index < problems.utterance_size(); ++index) {
    distance_holders_.push_back(problems.MakeDistances(index));
    hmm_solvers_.push_back(new HMMViterbiSolver(cluster_problems_.problem(index), 
                                                *distance_holders_[index]));
  }
  hidden_solver_ = new HiddenSolver(cluster_problems_);
  hidden_reparameterization_ = cluster_problems_.CreateReparameterization();
  hmm_reparameterization_ = cluster_problems_.CreateReparameterization();
  hidden_reparameterization2_ = cluster_problems_.CreateReparameterization();
  hmm_reparameterization2_ = cluster_problems_.CreateReparameterization();
  delta_hmm_ = cluster_problems_.CreateReparameterization();
  delta_hidden_ = cluster_problems_.CreateReparameterization();  
  best_primal_value_ = INF;
}

void SpeechSubgradient::SetReparameterization() {
  for (uint problem = 0; problem < hmm_solvers_.size(); ++problem) {
    hmm_solvers_[problem]->set_reparameterization(&(*hmm_reparameterization_)[problem]);
  }
  hidden_solver_->set_reparameterization(hidden_reparameterization_);
}

void SpeechSubgradient::SetReparameterization2() {
  for (uint problem = 0; problem < hmm_solvers_.size(); ++problem) {
    hmm_solvers_[problem]->set_reparameterization(&(*hmm_reparameterization2_)[problem]);
  }
  hidden_solver_->set_reparameterization(hidden_reparameterization2_);
}

// void SpeechSubgradient::Update(const DataPoint &data_point, 
//                                double alpha) { 
//   clock_t start = clock();
//   ClusterSubgrad *subgrad = ClusterSubgrad::FromVector(cluster_problems_, 
//                                                        ball_holder_->balls_size(),
//                                                        data_point);

//   // Setup the reparameterizations.
//   for (DataPoint::const_iterator pos = data_point.begin(); 
//        pos != data_point.end();
//        ++pos) { 
//     int index = pos.index();
//     ClusterIndex cluster_index;
//     subgrad->index(index, &cluster_index);
//     double score = alpha * subgrad->score(cluster_index);
//     // Update each hidden center in the ball around "hidden".
//     int nearby_size = 
//       ball_holder_->nearby_size(cluster_index.ball, cluster_index.partition);
//     //const ClusterProblem & problem = cluster_problems_.problem(cluster_index.problem);
//     for (int i = 0; i < nearby_size; ++i) {
//       int q = ball_holder_->nearby(cluster_index.ball, cluster_index.partition, i);
//       // hmm_solvers_[cluster_index.problem]->Update(cluster_index.state, q, score);
//       // hidden_solver_->Update(problem.MapState(cluster_index.state), q, -score);
//       (*hmm_reparameterization_)[cluster_index.problem][cluster_index.state][q] += score;
//       (*hidden_reparameterization_)[cluster_index.problem][cluster_index.state][q] -= score;
//     }
//   }
//   SetReparameterization();
//   clock_t end = clock();
//   cerr << "TIME: Update time " << end - start << endl; 
// 

double SpeechSubgradient::Primal(SpeechSolution *dual_proposal, int round, vector<DataPoint > *centroids) {

  
  cerr << "Dual Proposal score " << dual_proposal->ScoreSolution() << endl;
  double max_medians = problems_.MaximizeMedians(*dual_proposal, centroids);
  for (int type = 0; type < problems_.num_types(); ++type) {
    dual_proposal->set_type_to_special(type, (*centroids)[type]);
  }
  
  double dual = 0.0;
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    dual += hmm_solvers_[u]->Rescore(u, *dual_proposal);
  }
  dual += hidden_solver_->Rescore(*dual_proposal);
  cerr << "rescore is: " << dual << endl;

  stringstream buf;
  buf << "/tmp/last_solution" << round;
  fstream output(buf.str().c_str(), ios::out | ios::binary);
  speech::SpeechSolution solution;
  dual_proposal->ToProtobuf(solution, problems_);
  solution.SerializeToOstream(&output);
  output.close();

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
  double dual = hidden_solver_->Solve();  
  for (int type = 0; type < problems_.num_types(); ++type) {
    solution->set_type_to_hidden(type, hidden_solver_->TypeToHidden(type));
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
      double best_hidden = INF;
      for (int hidden = 0; 
           hidden < cluster_problems_.num_hidden(type); 
           ++hidden) {
        double trial = 
          (*delta_hmm_)[u][i][hidden] + (*delta_hidden_)[u][i][hidden];
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


// void SpeechSubgradient::Solve(const SubgradInfo &info, 
//                               SubgradResult *result) {
//   result->dual_value = 0.0;
//   SetReparameterization();

//   // The first problem.
//   SpeechSolution dual_proposal(cluster_problems_);
//   clock_t start = clock();
//   vector<vector<DataPoint> > cluster_set(cluster_problems_.num_types());
//   vector<vector< vector<DataPoint> > > group_cluster_set(cluster_problems_.num_types());
//   ClusterSubgrad *problem1 = new ClusterSubgrad(cluster_problems_, 
//                                                 ball_holder_->balls_size());
//   for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
//     assert(cluster_problems_.problems_size() == (int)hmm_solvers_.size());
//     cerr << "Solving HMM " << u << endl;
//     SpeechAlignment *alignment = dual_proposal.mutable_alignment(u);
//     result->dual_value += hmm_solvers_[u]->Solve(alignment);
//     hmm_solvers_[u]->ToSubgrad(u, *ball_holder_, problem1);
//     problems_.AlignmentClusterSet(u, *alignment->mutable_alignment(), &cluster_set);
//     problems_.AlignmentGroupClusterSet(u, *alignment->mutable_alignment(), &group_cluster_set);
//   }
//   clock_t end = clock();
//   cerr << "TIME: HMM Time " << end - start << endl;

//   // The second problem.
//   start = clock();
//   cerr << "Solving hidden." << endl;
//   result->dual_value += hidden_solver_->Solve();
//   hidden_solver_->ToSubgrad(cluster_problems_, *ball_holder_, problem1);
//   end = clock();
//   result->subgradient = problem1->ToVector();
//   cerr << "TIME: Other Time " << end - start << endl;
//   cerr << *problem1 << endl;

//   // Estimate the primal value from the current alignment.
//   start = clock();
  
//   vector<DataPoint> centers;
//   double max_means = problems_.MaximizeCenters(cluster_set, 
//                                                &centers);
//   if (max_means < best_means_) {
//     best_means_ = max_means;
//     best_centers_ = centers;
//   } 
//   cerr << "Primal means: " << best_means_ << " " << max_means << endl;

//   double max_medians;
//   SpeechSolution *solution = 
//     problems_.ApproxMaximizeMedians(dual_proposal, 
//                                     *ball_holder_, 
//                                     &max_medians);
//   // Check solution score.
//   double primal = 0.0;
//   for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
//     primal += hmm_solvers_[u]->PrimalRescore(u, *solution);
//   }
//   assert(fabs(primal - max_medians) < 1e-4);
  
//   double dual = 0.0;
//   for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
//     dual += hmm_solvers_[u]->Rescore(u, *solution);
//   }
//   dual += hidden_solver_->Rescore(*solution);

//   {
//     vector<DataPoint> temp;
//     double primal_real = Primal(&temp);
//     cerr << "Primal real is: " << primal_real << endl;
//   }

//   // Check dual is in sync.
//   for (int t = 0; t < cluster_problems_.num_types(); ++t) {
//     double left =0.0, right = 0.0;
//     int hidden = solution->TypeToHidden(t);
//     left += hidden_solver_->Check(t, hidden);
//     for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
//       right += hmm_solvers_[u]->Check(t, hidden);
//     }
//     assert(fabs(left + right) < 1e-4);
//   }

//   cerr << "Max Medians " << max_medians  << " " 
//        << ball_holder_->ball_epsilon(0) << " " << dual << endl;
//   assert(fabs(dual - max_medians) < 1e-4);
//   result->primal_value = max_medians;
//   end = clock();
//   cerr << "TIME: Primal Time " << end - start << endl;

//   delete problem1;
//   // Debug information.
//   //ClusterSubgrad::Align(*problem1, *problem2);
// }

vector<vector<vector<double> > > *SpeechSubgradient::MPLPDiff(const vector<vector<int> > &a, 
                                                             const vector<vector<int> > &b) const {
  vector<vector<vector<double> > > *diff  = 
    cluster_problems_.CreateReparameterization();
  
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    for (int i = 0; i < cluster_problems_.problem(u).num_states; ++i) {
      (*diff)[u][i][a[u][i]] += 1.0;
      (*diff)[u][i][b[u][i]] -= 1.0;
    }
  }
  return diff;
}

void SpeechSubgradient::MPLPAugment(vector<vector<vector<double> > > *weights, 
                                    const vector<vector<vector<double> > > &augment,
                                    double rate) {
  
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    const ClusterProblem &problem = cluster_problems_.problem(u);
    for (int i = 0; i < problem.num_states; ++i) {
      int type = problem.MapState(i);
      for (int h = 0; h < cluster_problems_.num_hidden(type); ++h) {
        (*weights)[u][i][h] += rate * augment[u][i][h]; 
      }
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
  vector<vector<int> > cluster = solution.ClusterAssignments();
  vector<vector<int> > unary;
  dual += HiddenDualUnaryProposal(&unary);
  vector<vector<vector<double > > > *hmm_diff = 
    MPLPDiff(hmm, unary);

  vector<vector<vector<double > > > *cluster_diff = 
    MPLPDiff(cluster, unary);

  MPLPAugment(hmm_reparameterization2_, *hmm_diff, rate);
  MPLPAugment(hidden_reparameterization2_, *cluster_diff, rate);
  delete hmm_diff;
  delete cluster_diff;
  return dual;
}

double SpeechSubgradient::MPLPAlignRound(int problem_num) {
  int u = problem_num;
  vector<vector<double> > max_marginals;
  const ClusterProblem &problem = cluster_problems_.problem(u);  

  // Resize the max marginal set. 
  max_marginals.resize(problem.num_states); 
  for (int i = 0; i < problem.num_states; ++i) {
    int type = problem.MapState(i);
    max_marginals[i].resize(cluster_problems_.num_hidden(type));
  }
  //cerr << cluster_problems_.num_hidden() << " " << problem.num_states << " " << problem.num_steps << endl;

  // Find the max-marginal values.
  double score = hmm_solvers_[u]->MaxMarginals(&max_marginals);

  // Reparameterize the distribution based on max-marginals.
  for (int i = 0; i < problem.num_states; ++i) {
    int type = problem.MapState(i);
    for (int hidden = 0; hidden < cluster_problems_.num_hidden(type); ++hidden) {
      // Remove the last deltas; 
      //(*hidden_reparameterization_)[u][i][hidden] -= (*delta_hmm_)[u][i][hidden];
      (*delta_hmm_)[u][i][hidden] = (-(*delta_hidden_)[u][i][hidden]) + 
        (1.0 / problem.num_states) * (max_marginals[i][hidden]);
      
      // Add the new delta to the opposite parameterization.
      //(*hidden_reparameterization_)[u][i][hidden] += (*delta_hmm_)[u][i][hidden];
      //cerr << "New hidden parameterization: " <<  u << " " << i << " " << hidden << " " << (*hidden_reparameterization_)[u][i][hidden] << endl;
    }
  }
  for (int i = 0; i < problem.num_states; ++i) {
    int type = problem.MapState(i);
    for (int hidden = 0; hidden < cluster_problems_.num_hidden(type); ++hidden) {
      (*hidden_reparameterization_)[u][i][hidden] = (*delta_hmm_)[u][i][hidden];
      (*hmm_reparameterization2_)[u][i][hidden] = -(*delta_hmm_)[u][i][hidden];
    }
  }
  if (false) {
    SetReparameterization2();
    SpeechAlignment alignment;
    double temp = hmm_solvers_[u]->Solve(&alignment);
    cerr << "temp test: " << temp << endl; 
    assert(fabs(temp) < 1e-4);
  }
  SetReparameterization();
  return score;
}

double SpeechSubgradient::MPLPClusterRound() {
  // Resize the max marginal set. 
  vector<vector<vector<double> > > *max_marginals  = 
    cluster_problems_.CreateReparameterization();

  double score = hidden_solver_->MaxMarginals(max_marginals);

  int total_states = 0;
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    total_states += cluster_problems_.problem(u).num_states;
  }

  // Reparameterize the distribution based on max-marginals.
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    const ClusterProblem &problem = cluster_problems_.problem(u);  
    int num_states = cluster_problems_.problem(u).num_states;
    for (int i = 0; i < num_states; ++i) {
      int type = problem.MapState(i);
      for (int hidden = 0; hidden < cluster_problems_.num_hidden(type); ++hidden) {
        // if (is_eliminated(type, hidden)) {
        //   continue;
        // }

        // Remove the last deltas; 
        //(*hmm_reparameterization_)[u][i][hidden] -= (*delta_hidden_)[u][i][hidden];
        (*delta_hidden_)[u][i][hidden] = (-(*delta_hmm_)[u][i][hidden]) + 
          (1.0 / total_states) * ((*max_marginals)[u][i][hidden]);
        
        // Add the new delta to the opposite parameterization.
        //(*hmm_reparameterization_)[u][i][hidden] += (*delta_hidden_)[u][i][hidden];
        //cerr << "New hmm parameterization: " <<  u << " " << i << " " << hidden << " " << (*hmm_reparameterization_)[u][i][hidden] << endl;
      }
    }
  }
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    const ClusterProblem &problem = cluster_problems_.problem(u);  
    for (int i = 0; i < cluster_problems_.problem(u).num_states; ++i) {
      int type = problem.MapState(i);
      // if (is_eliminated(type, hidden)) {
      //   continue;
      // }
      for (int hidden = 0; hidden < cluster_problems_.num_hidden(type); ++hidden) {
        (*hmm_reparameterization_)[u][i][hidden] = (*delta_hidden_)[u][i][hidden];
        (*hidden_reparameterization2_)[u][i][hidden] = -(*delta_hidden_)[u][i][hidden];
      }
    }
  }
  if (false) {
    SetReparameterization2();
    double temp = hidden_solver_->Solve();
    cerr << "temp test: " << temp << endl; 
    assert(fabs(temp) < 1e-4);
  }
  SetReparameterization();
  delete max_marginals;
  return score;
}


void SpeechSubgradient::MPLPRound(int round) {

  SetReparameterization();
  double score;
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    score = MPLPClusterRound();
    cerr << "MPLP score: " << score << endl;  
    score = MPLPAlignRound(u);
    cerr << "MPLP score: " << score << endl;
  }
  SetReparameterization2();


  double dual_value = 0.0;
  SpeechSolution *solution = new SpeechSolution(cluster_problems_);
  dual_value += hidden_solver_->Solve();  
  dual_value = DualProposal(solution);
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
        // if (is_eliminated(type, hidden)) {
        //   continue;
        // }     
        double trial = 
          (*delta_hmm_)[u][i][hidden] + (*delta_hidden_)[u][i][hidden];
        if (trial < best_hidden) {
          best_hidden = trial;
          (*state_hidden)[i] = hidden;
        }
      }
      dual_value += best_hidden;
    }
  }
 
    // Actually centroids here.
  vector<DataPoint> centroids;
  double primal_value = Primal(solution, round, &centroids);

  if (primal_value < best_primal_value_) {
    best_primal_value_ = primal_value;
    best_centers_ = centroids;
  } 

  vector <DataPoint> centroids2;
  problems_.MaximizeMediansHidden(unary_solution, &centroids2);

  cerr << "Final primal value " << best_primal_value_ << " " <<primal_value << endl;
  cerr << "Final dual value: " << dual_value << endl;

  SetReparameterization();
  delete solution;
}
