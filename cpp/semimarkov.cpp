#include <iostream>
#include "semimarkov.h"
#include <time.h>
#define DEBUG 0

void SemiMarkov::Initialize() {
  int n = num_timesteps_;
  int m = num_states_;
 
  // Size the charts correctly.
  best_action_.resize(n + 1);
  best_score_.resize(n + 1);
  best_back_action_.resize(n + 1);
  best_back_score_.resize(n + 1);
  for (int t = 0; t < n + 1; ++t) {
    best_action_[t].resize(m + 1);
    best_score_[t].resize(m + 1);
    best_back_action_[t].resize(m + 1);
    best_back_score_[t].resize(m + 1);
  }
 
  // Initialize the scoring chart.
  scores_ = new SpanChart<vector<double> >(n, width_limit_);
  pruned_ = new SpanChart<bool>(n, width_limit_);
  for (int s = 0; s < n; ++s) {
    for (int o = 0; o < width_limit_; ++o) {
      pruned_->set(s, o, false);
      scores_->get(s, o).resize(m);
      for (int i = 0 ; i < m ; ++i) {
        scores_->get(s, o)[i] = INF;
      }
    }
  }
}

void SemiMarkov::ViterbiForward() {
  clock_t start = clock();
  int n = num_timesteps_;
  int m = num_states_;

  // Score the first round correctly.
  for (int t = 0; t < n + 1; ++t) {
    for (int i = 0; i < m + 1; ++i) {
      best_action_[t][i] = -1;
      best_score_[t][i] = INF;
    }
    best_action_[t][0] = 0;
  }
  best_score_[0][0] = 0.0;

  // The main loop for each cell.
  for (int t = 0; t < n + 1; ++t) {
    for (int s = max(0, t - width_limit_ + 1); s < t; ++s) {
      if (pruned(s, t - 1)) {
        continue;
      }
      for (int i = 1; i < min(t + 1, m + 1); ++i) { 
      // Find best way to transition to time t at state i.
        double local_score = score(s, t - 1, i - 1);
        double trial = best_score_[s][i - 1] + local_score;        
        if (trial < best_score_[t][i])  {
          best_score_[t][i] = trial;
          best_action_[t][i] = s;
        }
      }
    }
  }
  assert(best_score_[n][m] < INF);

  // Forward is done running.
  forward_ = true;
  clock_t end = clock();

  cerr << "Running semi-markov forward " << n << " " << m << " " << end - start << endl;
}

void SemiMarkov::ViterbiBackward() {
  assert(forward_);
  clock_t start = clock();
  int n = num_timesteps_;
  int m = num_states_;

  // Score the last round correctly.
  for (int t = 0; t < n + 1; ++t) {
    for (int i = 0; i < m + 1; ++i) {
      best_back_action_[t][i] = -1;
      best_back_score_[t][i] = INF;
    }
  }
  best_back_score_[n][m] = 0.0;

  // The main loop for each cell. t is current time, s is previous time, 
  // i is current state.
  for (int t = n; t >= 0; --t) {
    for (int s = max(0, t - width_limit_ + 1); s < t; ++s) {
      if (pruned(s, t - 1)) continue;
      for (int i = m; i > 0; --i) { 
        // Find best way to transition to time t at state i.
        double local_score = score(s, t - 1, i - 1);
        double trial = best_back_score_[t][i] + local_score;        
        if (trial < best_back_score_[s][i - 1])  {
          best_back_score_[s][i - 1] = trial;
          best_back_action_[s][i - 1] = t;
        }
      }
    }
  }

  // Backward is done running.
  backward_ = true;
  clock_t end = clock();
  cerr << "Running semi-markov backward " << n << " " << m << " " << end - start <<endl;
}

double SemiMarkov::GetForwardScore(int state, int timestep) const {
  assert(forward_);
  assert(timestep < num_timesteps_);
  assert(state < num_states_);
  return best_score_[timestep][state];
}

double SemiMarkov::GetBackwardScore(int state, int timestep) const {
  assert(backward_);
  if (state == num_states_) {
    if (timestep == num_timesteps_) {
      return 0.0;
    } else {
      return INF;
    }
  }
  assert(timestep < num_timesteps_);
  assert(state < num_states_);
  return best_back_score_[timestep][state];
}

double SemiMarkov::GetBestPath(vector<int> *path) const {
  assert(forward_);

  // Back pointers to find path. 
  int n = num_timesteps_;
  int m = num_states_;
  int cur_time = n;
  int cur_state = m;
  double total_score = best_score_[cur_time][cur_state];
  double check_score = 0.0;
  path->clear();
  path->push_back(n);
  while (cur_state != 0) {
    int last_time = cur_time;
    cur_time = best_action_[cur_time][cur_state];
    assert(cur_time != -1); 
    assert(!pruned(cur_time, last_time - 1));
    check_score += score(cur_time, last_time -1, cur_state - 1);
    path->push_back(cur_time);
    cur_state--;
  }
  assert((int)path->size() == num_states_ + 1);
  cerr << total_score << " " << check_score <<endl;
  //assert(fabs(total_score - check_score) < 1e-4);
  reverse(path->begin(), path->end());
  return total_score;
}

double SemiMarkov::GetBestBackPath(vector<int> *path) const {
  assert(backward_);

  // Back pointers to find path. 
  int m = num_states_;
  int cur_time = 0;
  int cur_state = 0;
  double total_score = best_back_score_[0][0];
  double check_score = 0.0;
  path->clear();
  path->push_back(0);
  while (cur_state != m) {
    int last_time = cur_time;
    cur_time = best_back_action_[cur_time][cur_state];
    assert(cur_time != -1); 
    check_score += score(last_time, cur_time - 1, cur_state);
    path->push_back(cur_time);
    cur_state++;
  }
  assert(fabs(total_score - check_score) < 1e-4);
  return total_score;
}
