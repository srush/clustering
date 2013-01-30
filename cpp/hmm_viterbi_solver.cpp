#include "hmm_viterbi_solver.h"
#include <time.h>

Viterbi *HMMViterbiSolver::InitializeViterbi() {

  clock_t start = clock();
  viterbi_ = new Viterbi(cp_.num_states, 
                         cp_.num_steps, 
                         cp_.num_hidden(0), 
                         1);
  viterbi_->Initialize();

  clock_t end = clock();
  cerr << "TIME: Initializing viterbi " << end - start << endl; 

  start = clock();
  // Update semi_markov weights with hidden scores.
  for (int m = 0; m < cp_.num_steps; m++) {
    for (int c = 0; c < cp_.num_hidden(0); ++c) {
      double score = distances_.get_distance(m, c);
      viterbi_->set_score(m, c, score);
    }
  }

  for (int i = 0; i < cp_.num_states; i++) {
    for (int c = 0; c < cp_.num_hidden(0); ++c) {
      viterbi_->set_lambda(i, c, (*reparameterization_)[i][c]);
    }
  }

  end = clock();
  cerr << "TIME: Initializing viterbi " << end - start << endl; 
  return viterbi_;
}

double HMMViterbiSolver::MaxMarginals(vector<vector<double> > *mu,
                                      SpeechAlignment *alignment) {
  mu->resize(cp_.num_states); 
  for (int i = 0; i < cp_.num_states; ++i) {
    int type = cp_.MapState(i);
    (*mu)[i].resize(cp_.num_hidden(type));
  }

  Viterbi *viterbi = InitializeViterbi();
  clock_t start = clock();
  // Run the semi-markov model forward and backward.
  viterbi->ForwardScores();
  clock_t end = clock();
  cerr << "TIME: Forward time " << end - start << endl;

  double score = viterbi->GetBestPath(alignment->mutable_alignment(), 
                                      &state_to_center_);
  cerr << "PATH: ";
  for (uint i = 0; i < alignment->mutable_alignment()->size(); ++i) {
    cerr << (*alignment->mutable_alignment())[i] << ":" << state_to_center_[i] << ":" << cp_.MapState(i) << " ";
  }
  cerr << endl;
  
  state_to_center_.resize(cp_.num_states + 1);
  //cerr << "PATH2: ";
  vector<int> *hidden = alignment->mutable_hidden_alignment();
  hidden->resize(cp_.num_states + 1);
  for (int state = 0; state < cp_.num_states + 1; ++state) {
    (*hidden)[state] = state_to_center_[state];
    //cerr << state << "/" << state_to_center_[state] << " ";
  }
  //cerr << endl;

  start = clock();
  viterbi->BackwardScores();
  end = clock();
  cerr << "TIME: Backward time " << end - start << endl;


  start = clock();
  // Calculate the mu marginals.
  viterbi->MinMarginals(mu);
  // mu->resize(m);
  // for (int i = 0; i < cp_.num_states_; ++i) {
  //   int type = cp_.MapState(i);
  //   (*mu)[i].resize(cp_.num_hidden(type));
  //   for (int hidden = 0; hidden < cp_.num_hidden(type); ++hidden) {
  //     (*mu)[i][hidden] = INF;
  //   }
  // }

  // Find the max marginals.
  // \max_{s,e} \alpha(s,i) + \beta(e,i+1) + (\sum_{m=s}^e c(m,q) + \lambda \(i, q))
            // if (i == m - 1) {
      //   bound = min(cp_.width_limit(), n - s + 1);
      // }

  // double worst = -INF;
  // for (int s = 0; s < n; s++) {
  //   int bound = min(cp_.width_limit(), n - s + 1);
  //   for (int o = 1; o < bound; ++o) {
  //     if (hidden_holder_.is_pruned(s, o - 1) && o != n - s) {
  //       continue;
  //     }
  //     for (int hidden = 0; hidden < cp_.num_hidden(0); ++hidden) {
  //       // if (hidden_holder_.is_pruned(s, o - 1, hidden) && o != n - s) {
  //       //   continue;
  //       // }
  //       if (hidden_holder_.is_pruned(s, o - 1, hidden) && o != n - s) {
  //         continue;
  //       }
  //       for (int i = 0; i < m; ++i) {
  //         if (o == n - s && i != m - 1) continue; 
  //         double alpha = semi_markov->GetForwardScore(i, s);
  //         double beta = semi_markov->GetBackwardScore(i + 1, s + o);
  //         double hidden_cost = hidden_holder_.Rescore(s, o - 1, i, hidden);
  //         double total_cost = alpha + beta + hidden_cost;
  //         if (total_cost < (*mu)[i][hidden]) {
  //           (*mu)[i][hidden] = total_cost;
  //           //assert((*mu)[i][hidden] >= score - 1e-4);
  //         } 
  //         if (total_cost < INF && total_cost > worst) {
  //           worst = total_cost;
  //         }
  //       }
  //     }
  //   }
  // }
  // for (int i = 0; i < m; ++i) {
  //   int type = cp_.MapState(i); 
  //   for (int hidden = 0; hidden < cp_.num_hidden(type); ++hidden) {
  //     //assert((*mu)[i][hidden] != INF);
  //     if ((*mu)[i][hidden] >= 10000) {
  //       (*mu)[i][hidden] = 10000 + hidden_holder_.hidden_costs(i, hidden);
  //     }
  //   }
  // }
  end = clock();
  cerr << "TIME: Finishing max-marginals " << end -start << endl;
  delete viterbi;
  return score;
}

double HMMViterbiSolver::ScoreSegment(int start, int offset, int state, int center) const {
  double score = 0.0;
  //cerr << start << " " << offset << " " << state << " " << center << endl;
  for (int m = start; m <= start + offset; ++m) {
    score += distances_.get_distance(m, center);
  }
  score += (*reparameterization_)[state][center];
  return score;
}

double HMMViterbiSolver::PrimalScoreSegment(int start, int offset, int state, int center) const {
  double score = 0.0;
  for (int m = start; m <= start + offset; ++m) {
    score += distances_.get_distance(m, center);
  }
  return score;
}

double HMMViterbiSolver::Solve(SpeechAlignment *alignment) {
  Viterbi *viterbi = InitializeViterbi();

  // Run the semi-markov model.
  viterbi->ForwardScores();

  double score = 
    viterbi->GetBestPath(alignment->mutable_alignment(), &state_to_center_);  
  state_to_center_.resize(cp_.num_states);
  double check_score = 0.0;

  vector<int> *hidden = alignment->mutable_hidden_alignment();
  hidden->resize(alignment->alignment_size());
  for (int state = 0; state < alignment->alignment_size() ; ++state) {
    int start, end;
    alignment->StateAlign(state, &start, &end);
    (*hidden)[state] = state_to_center_[state];
    check_score += ScoreSegment(start, end - start, state, state_to_center_[state]);
  }
  assert(fabs(score - check_score) < 1e-4);
  delete viterbi;
  return score;
}

double HMMViterbiSolver::Rescore(int problem, const SpeechSolution &solution) const {
  double dual_score = 0.0;
  const SpeechAlignment &alignment = solution.alignment(problem);
  for (int state = 0; state < cp_.num_states; ++state) {
    int s, e;
    alignment.StateAlign(state, &s, &e);
    int hidden = alignment.HiddenAlign(state);
    dual_score += ScoreSegment(s, e-s, state, hidden);
  }
  return dual_score;
}

double HMMViterbiSolver::PrimalRescore(int problem, 
                                       const SpeechSolution &solution) const {
    double primal_score = 0.0;
    const SpeechAlignment &alignment = solution.alignment(problem);
    for (int state = 0; state < cp_.num_states; ++state) {
      int s, e;
      alignment.StateAlign(state, &s, &e);
      int hidden = alignment.HiddenAlign(state);
      primal_score += PrimalScoreSegment(s, e-s, state, hidden);
    }
    return primal_score;
}
