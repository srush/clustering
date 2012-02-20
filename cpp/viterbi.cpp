#include <iostream>
#include "viterbi.h"
#include <vector>
#include <cmath>
#include <assert.h>
using namespace std;
#define DEBUG 1
#define INF 1e10

void Viterbi::Initialize() {
  int m = num_states_;
  int n = num_timesteps_;

  // Size the charts correctly.
  best_action_.resize(n + 1);
  best_score_.resize(n + 1);
  scores_.resize(n + 1);
  for (int t = 0; t < n + 1; ++t) {
    best_action_[t].resize(m + 1);
    best_score_[t].resize(m + 1);
    scores_[t].resize(m + 1);
  }
}

double Viterbi::Run(vector<int> *path) {
  int n = num_timesteps_;
  int m = num_states_;

  // Score the first round correctly.  
  for (int t = 0; t < n; ++t) {
    best_score_[t][0] = INF;
  }
  for (int i = 0; i < m; ++i) {
    best_score_[0][i] = INF;
  }
  best_score_[0][0] = 0.0;

  for (int t = 1; t < n; ++t) {
    for (int i = 1; i < m; ++i) {
      double a = best_score_[t - 1][i];
      double b = best_score_[t - 1][i - 1];
      if (a < b) {
        best_score_[t][i] = a;
        best_action_[t][i] = 0;
      } else {
        best_score_[t][i] = b;
        best_action_[t][i] = -1;
      }
      best_score_[t][i] += score(t, i);
    }
  }
  

  int cur_time = n - 1;
  int cur_state = m -1 ;
  double total_score = best_score_[cur_time][cur_state];
  double check_score = 0.0;
  path->clear();
  path->push_back(cur_time + 1);
  while (cur_state != 0) {
    while (best_action_[cur_time][cur_state] == 0) {
      check_score += score(cur_time, cur_state);
      cur_time--;
    }
    check_score += score(cur_time, cur_state);
    path->push_back(cur_time);
    cur_time--;
    cur_state--;
  }
  path->push_back(0);

  assert(fabs(check_score - total_score) < 1e-4);
  reverse(path->begin(), path->end());
  cerr << "Viterbi score: " << total_score << endl;
  cerr << "Alignment: ";
  for (uint p = 0; p < path->size(); ++p) { 
    cerr << (*path)[p] << " ";
  }
  cerr << endl;

  return total_score;
}
