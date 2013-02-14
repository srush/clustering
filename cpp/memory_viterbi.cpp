#include <iostream>
#include <algorithm>
#include "viterbi.h"
#include <vector>
#include <cmath>
#include <assert.h>
using namespace std;
#define DEBUG 1
#define INF 1e10
#define LOGZERO 1e7
typedef unsigned int uint;

void Viterbi::Initialize() {
  // Size the charts correctly.
  int time_size = num_timesteps_ + 1;
  int state_size = num_states_ + 1;
  int center_size = num_centers_;

  scores_.resize(time_size);  
  forward_scores_.resize(time_size);
  backward_scores_.resize(time_size);
  back_pointer_.resize(time_size);
  transition_score_.resize(time_size);
  for (int m = 0; m < time_size; ++m) {
    forward_scores_[m].resize(state_size);
    backward_scores_[m].resize(state_size);
    back_pointer_[m].resize(state_size);
    transition_score_[m].resize(state_size);
    scores_[m].resize(center_size);
    scores_.resize(time_size);  
    for (int i = 0; i < state_size; ++i) {
      forward_scores_[m][i].resize(center_size);
      backward_scores_[m][i].resize(center_size);
      back_pointer_[m][i].resize(center_size);
      transition_score_[m][i].resize(center_size, 0.0);
    }
  }

  lambda_.resize(state_size);
  for (int i = 0; i < state_size; ++i) {
    lambda_[i].resize(center_size);
  }

  for (int m = 0; m <= num_timesteps_; ++m) {
    for (int i = 0; i <= num_states_; ++i) {
      for (int c = 0; c < num_centers_; c++) {
        forward_scores_[m][i][c] = INF;
        backward_scores_[m][i][c] = INF;
        // -2 for back pointers that are not yet set. 
        back_pointer_[m][i][c] = -2;
      }
    }
  }
  ResetChart();
}

// Run the forward viterbi algorith. 
void Viterbi::ForwardScores() {
  // Initialize the forward chart. 
  for (int c = 0; c < num_centers_; ++c) {
    forward_scores_[0][0][c] = lambda_[0][c] + scores_[0][c] + transition_score_[0][0][c];
  }
  
  for (int m = 1; m <= num_timesteps_; ++m) {
    for (int i = 0; i <= num_states_; ++i) {
      double back_score = INF;
      if (num_states_ - i > num_timesteps_ - m) continue;

      int back_center = -1;

      // Find the best possble transition center.
      if (i != 0) {
        for (int c2 = 0; c2 < num_centers_; c2++) {
          double trial;
          trial = forward_scores_[m - 1][i - 1][c2];
          if (trial < back_score) { 
            back_score = trial; 
            back_center = c2;
          }
        }
      }
      for (int c = 0; c < num_centers_; ++c) {
        double w;
        // Last time step must transition to final state.
        if (m == num_timesteps_) {
          w = (c == 0 && i == num_states_) ? 0.0 : INF;
        } else {
          // Last state only valid on last time step.
          w = (i == num_states_) ? INF : 
            scores_[m][c] + transition_score_[m][i][c];
        }
        double stay_score = w + forward_scores_[m - 1][i][c];
        double switch_score = w + back_score;

        if (switch_score < stay_score ) {
          forward_scores_[m][i][c] = switch_score;
          back_pointer_[m][i][c] = back_center;
        } else {
          forward_scores_[m][i][c] = stay_score;
          back_pointer_[m][i][c] = -1;
        }
      }
    }
  } 
}


double Viterbi::score(int time, int center) const {
  assert(time < num_timesteps_);
  assert(center < num_centers_);
  return scores_[time][center];
}

double Viterbi::GetBestPath(vector<int> *path, vector<int> *centers) {
  int cur_time = num_timesteps_ ;
  int cur_state = num_states_;
  int cur_center = 0;
  double total_score = forward_scores_[cur_time][cur_state][cur_center];
  double check_score = 0.0;
  path->clear();
  assert(back_pointer_[cur_time][cur_state][cur_center] != -1);
  while (cur_time > 0) {
    while (back_pointer_[cur_time][cur_state][cur_center] == -1) {
      assert(fabs(check_score + forward_scores_[cur_time][cur_state][cur_center] 
                  - total_score) < 1e-4);
      check_score += score(cur_time, cur_center) + 
        transition_score(cur_time, cur_state, cur_center);
      cur_time--;
      if (cur_time <= 0) {
        break;
      }
    }
    if (cur_time <= 0) {
      break;
    }
    assert(back_pointer_[cur_time][cur_state][cur_center] != -2);

    int next_center = back_pointer_[cur_time][cur_state][cur_center];
    cur_center = next_center;
    assert(cur_center >= -1);
    centers->push_back(cur_center);
    if (cur_time == num_timesteps_ && cur_state == num_states_) {
      path->push_back(cur_time);
      cur_time--;
    } else {
      path->push_back(cur_time - min_width_ + 1);
      cur_time -= min_width_;
    }
    cur_state--;
  }
  assert(cur_state == 0);
  path->push_back(0);
  check_score += lambda(0, cur_center) + score(0, cur_center) + transition_score(0, 0, cur_center);
  assert(fabs(check_score - total_score) < 1e-4);
  reverse(path->begin(), path->end());
  reverse(centers->begin(), centers->end());
  cerr << "VITERBI: Viterbi score " << total_score << endl;
  cerr << "VITERBI: Alignment: ";
  for (uint p = 0; p < path->size(); ++p) { 
    cerr << (*path)[p] << " ";
  }
  cerr << endl;
  cerr << "VITERBI: Centers: ";
  for (uint o = 0; o < centers->size(); ++o) { 
    cerr << (*centers)[o] << " ";
  }
  cerr << endl;

  return total_score;
}
