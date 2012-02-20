#include "hmm_solver.h"
#include <time.h>

SemiMarkov *HMMSolver::InitializeSemiMarkov() {
  SemiMarkov *semi_markov = new SemiMarkov(cp_.num_states, 
                                           cp_.num_steps, 
                                           cp_.width_limit);
  semi_markov->Initialize();

  int m = semi_markov->num_states();
  int n = semi_markov->num_timesteps();

  // Precompute hidden states.
  hidden_holder_.ComputeBestHidden();

  // Update semi_markov weights with hidden scores.
  for (int i = 0; i < m; i++) { 
    for (int s = 0; s < n; s++) {
      for (int o = 0; o < cp_.width_limit; ++o) {
        if (s + o >= n) continue; 
        double score = hidden_holder_.get_score(s, o, i);
        semi_markov->set_score(s, s + o, i, score);
        // TODO: Pruning.
        // bool pruned = hidden_holder_.pruned(s, o);
        // if (pruned && s != 0 && o != 0) {
        //   semi_markov.set_pruned(s, s + o);
        // }
        assert(o == 0 || score != 0.0);
      }
    }
  }
  return semi_markov;
}

double HMMSolver::MaxMarginals(vector<vector<double> > *mu) {
  SemiMarkov *semi_markov = InitializeSemiMarkov();

  int m = semi_markov->num_states();
  int n = semi_markov->num_timesteps();

  // Run the semi-markov model forward and backward.
  semi_markov->ViterbiForward();
  semi_markov->ViterbiBackward();

  vector<int> path;
  double score = semi_markov->GetBestPath(&path);

  // Calculate the mu marginals.
  mu->resize(m);
  for (int i = 0; i < m; ++i) {
    (*mu)[i].resize(cp_.num_hidden);
    for (int hidden = 0; hidden < cp_.num_hidden; ++hidden) {
      (*mu)[i][hidden] = INF;
    }
  }

  
  // Find the max marginals.
  // \max_{s,e} \alpha(s,i) + \beta(e,i+1) + (\sum_{m=s}^e c(m,q) + \lambda \(i, q))
  for (int i = 0; i < m; ++i) { 
    for (int s = 0; s < n; s++) {
      double alpha = semi_markov->GetForwardScore(i, s);
      int bound = min(cp_.width_limit, n - s);
      if (i == m - 1) {
        bound = min(cp_.width_limit, n - s + 1);
      }
      for (int o = 1; o < bound; ++o) {
        double beta = semi_markov->GetBackwardScore(i + 1, s + o);
        for (int hidden = 0; hidden < cp_.num_hidden; ++hidden) {
          double hidden_cost = hidden_holder_.Rescore(s, o - 1, i, hidden);
          double total_cost = alpha + beta + hidden_cost;
          if (total_cost < (*mu)[i][hidden]) {
            (*mu)[i][hidden] = total_cost;
            //assert((*mu)[i][hidden] >= score - 1e-4);
          } 
        }
      }
    }
  }
  return score;
}

double HMMSolver::Solve(SpeechAlignment *alignment) {
  SemiMarkov *semi_markov = InitializeSemiMarkov();

  // Run the semi-markov model.
  semi_markov->ViterbiForward();
  double score = semi_markov->GetBestPath(alignment->mutable_alignment());  
  state_to_center_.resize(cp_.num_states);
  double check_score = 0.0;

  vector<int> *hidden = alignment->mutable_hidden_alignment();
  hidden->resize(alignment->alignment_size());
  for (int state = 0; state < alignment->alignment_size() ; ++state) {
    int start, end;
    alignment->StateAlign(state, &start, &end);
    state_to_center_[state] = hidden_holder_.get_center(start, 
                                                        end - start, 
                                                        state);
    (*hidden)[state] = state_to_center_[state];
    check_score += hidden_holder_.get_score(start, end - start, state);
  }
  assert(fabs(score - check_score) < 1e-4);
  delete semi_markov;
  return score;
}

double HMMSolver::Rescore(int problem, const SpeechSolution &solution) const {
  double dual_score = 0.0;
  const SpeechAlignment &alignment = solution.alignment(problem);
  for (int state = 0; state < cp_.num_states; ++state) {
    int s, e;
    alignment.StateAlign(state, &s, &e);
    int hidden = alignment.HiddenAlign(state);
    dual_score += hidden_holder_.Rescore(s, e-s, state, hidden);
  }
  return dual_score;
}

double HMMSolver::PrimalRescore(int problem, const SpeechSolution &solution) const {
 double primal_score = 0.0;
  const SpeechAlignment &alignment = solution.alignment(problem);
  for (int state = 0; state < cp_.num_states; ++state) {
    int s, e;
    alignment.StateAlign(state, &s, &e);
    int hidden = alignment.HiddenAlign(state);
    primal_score += hidden_holder_.PrimalRescore(s, e-s, state, hidden);
  }
  return primal_score;
}

void HMMSolver::ToSubgrad(int problem, 
                          const BallHolder &ball_holder,
                          ClusterSubgrad *subgrad) const {
  for (int i = 0; i < cp_.num_states; ++i) {
    int hidden = state_to_center_[i];
    for (int b = 0; b < ball_holder.balls_size(); ++b) {
      int partition = ball_holder.partition_for_center(b, hidden);
      ClusterIndex cluster_index(b, problem, i, partition);
      subgrad->On(cluster_index);

    }
  }
}

//for (int n = 0; n < ball_holder.nearby_size(b, partition); ++n) {
//int q = ball_holder.nearby(b, partition, n);
//}
