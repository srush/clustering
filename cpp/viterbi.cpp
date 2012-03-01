#include <iostream>
#include "viterbi.h"
#include <vector>
#include <cmath>
#include <assert.h>
using namespace std;
#define DEBUG 1
#define INF 1e10

void Viterbi::Initialize() {
  // Size the charts correctly.
  scores_.resize(num_timesteps_ + 1);  
  forward_scores_.resize(num_timesteps_ + 1);
  backward_scores_.resize(num_timesteps_ + 1);
  back_pointer_.resize(num_timesteps_ + 1);
  best_back_.resize(num_timesteps_ + 1);
  state_score_.resize(num_timesteps_ + 1);
  for (int m = 0; m < num_timesteps_ + 1; ++m) {
    forward_scores_[m].resize(num_states_ + 1);
    backward_scores_[m].resize(num_states_ + 1);
    back_pointer_[m].resize(num_states_ + 1);
    scores_[m].resize(num_centers_);
    best_back_[m].resize(num_states_ + 1, INF);
    state_score_[m].resize(num_states_, 0.0);
    for (int i = 0; i < num_states_ + 1; ++i) {
      forward_scores_[m][i].resize(num_centers_);
      backward_scores_[m][i].resize(num_centers_ );
      back_pointer_[m][i].resize(num_centers_);
    }
  }

  lambda_.resize(num_states_ + 1);
  for (int i = 0; i < num_states_; ++i) {
    lambda_[i].resize(num_centers_);
  }
  for (int m = 0; m <= num_timesteps_; ++m) {
    for (int i = 0; i <= num_states_; ++i) {
      for (int c = 0; c < num_centers_; c++) {
        forward_scores_[m][i][c] = INF;
        backward_scores_[m][i][c] = INF;
        back_pointer_[m][i][c] = -2;
      }
    }
  }

  ResetChart();
}

void Viterbi::ResetChart() {
  for (int c = 0; c < num_centers_; ++c) {
    forward_scores_[0][0][c] = 0.0;
    backward_scores_[num_timesteps_][num_states_][0] = 0.0;
  }
}

void Viterbi::MinMarginals(vector<vector<double> > *min_marginals) {
  min_marginals->resize(num_states_ + 1);
  for (int i = 0; i < num_states_; ++i) {
    (*min_marginals)[i].resize(num_centers_, INF);
  }

  clock_t start = clock();
  double best_score = forward_scores_[num_timesteps_][num_states_][0];
  for (int m = 0; m <= num_timesteps_; ++m) {
    for (int i = 0; i <= num_states_; ++i) {
      if (i > m) continue;
      for (int c = 0; c < num_centers_; ++c) {
        double pen = 0.0;
        if (m < num_timesteps_ && i < num_states_) {
          pen = lambda(i, c);
        }
        double trial = backward_scores_[m][i][c] + pen;
        if (trial < best_back_[m][i]) {
          best_back_[m][i] = trial;
        }
      }
    }
  }

  for (int i = 0; i < num_states_; ++i) {
    for (int c = 0; c < num_centers_; ++c) {
      (*min_marginals)[i][c] = INF;
    }
  }

  for (int m = 0; m < num_timesteps_; ++m) {
    for (int i = 0; i < num_states_; ++i) {
      if (i > m) continue;
      for (int c = 0; c < num_centers_; ++c) {
        double trial1 = forward_scores_[m][i][c] + backward_scores_[m + 1][i][c];
        double trial2 = forward_scores_[m][i][c] + best_back_[m + min_width_][i + 1];
        if (trial1 < (*min_marginals)[i][c]) {
          (*min_marginals)[i][c] = trial1;
        }
        if (trial2 < (*min_marginals)[i][c]) {
          (*min_marginals)[i][c] = trial2;
        }
        assert((*min_marginals)[i][c] - best_score > -1e-4);
      }
    }
  }
  for (int i = 0; i < num_states_; ++i) {
    double best = INF;
    for (int c = 0; c < num_centers_; ++c) {
      assert((*min_marginals)[i][c] - best_score > -1e-4);
      best = min(best, (*min_marginals)[i][c]);
    }
    assert(fabs(best - best_score) < 1e-4);
  }
  clock_t end = clock();
  cerr << "MinMarginals time " << end -start << endl;

  cerr << best_score << endl;
}


void Viterbi::ForwardScores() {
  for (int c = 0; c < num_centers_; ++c) {
    forward_scores_[0][0][c] = lambda_[0][c] + scores_[0][c] + state_score_[0][0];
  }
  for (int m = 1; m <= num_timesteps_; ++m) {
    for (int i = 0; i <= num_states_; ++i) {
      if (i > m) continue; 
      double b = INF;
      int back_center = -1;
      if (i != 0 && m >= min_width_) {
        for (int c2 = 0; c2 < num_centers_; c2++) {
          forward_scores_[m][i][c2] = INF;
          back_pointer_[m][i][c2] = -2;
          double trial = forward_scores_[m - min_width_][i - 1][c2];
          if (trial < b) { 
            b = trial; 
            back_center = c2;
          }
          assert(back_center != -1);
        }
      }
      for (int c = 0; c < num_centers_; ++c) {
        double w;
        if (m == num_timesteps_) {
          if (c == 0 && i == num_states_ ) {
            w = 0.0;
          } else {
            w = INF;
          }
        } else {
          if (i == num_states_) {
            w = INF;
          } else {
            w = scores_[m][c] + state_score_[m][i];
          }
        }
        double stay_score = w + forward_scores_[m - 1][i][c];
        double pen = 0.0;
        if (i < num_states_) {
          pen = lambda_[i][c];
        }
        if (m - min_width_ >= 0) {
          for (int pre = 0; pre < min_width_; ++pre) {
            pen += scores_[m - pre][c] + state_score_[m - pre][i];
          }
        } else {
          pen = INF;
        }
        double switch_score =  w + b + pen;

        if (switch_score < stay_score ) {
          forward_scores_[m][i][c] = switch_score;
          back_pointer_[m][i][c] = back_center;
        } else {
          forward_scores_[m][i][c] = stay_score;
          back_pointer_[m][i][c] = -1;
        }
        // if (forward_scores_[m][i][c] < INF) {
        //   cerr << m << " " << i << " " << c << " " << forward_scores_[m][i][c] << " " << back_pointer_[m][i][c] << " " << w<< endl;
        // }
      }
    }
  } 
}

void Viterbi::BackwardScores() {
  for (int m = num_timesteps_ - 1; m >= 0 ; --m) {
    for (int i = num_states_ - 1; i >= 0 ; --i) {
      double b = INF;
      if (i == num_states_ - 1 && m == num_timesteps_ - 1) {
        b = 0.0;
      } else {
        for (int c2 = 0; c2 < num_centers_; c2++) {
          double pen = 0.0;
          if (i + 1 != num_states_) {
            pen = lambda_[i + 1][c2];
          }
          b = min(b, backward_scores_[m + min_width_][i + 1][c2] + pen);
          backward_scores_[m][i][c2] = INF;
        }
      } 
      for (int c = 0; c < num_centers_; ++c) {
        double w = scores_[m][c];
        double stay_score = backward_scores_[m + 1][i][c];
        double switch_score = b;
        if (stay_score < switch_score) {
          backward_scores_[m][i][c] = w + stay_score;
        } else {
          backward_scores_[m][i][c] = w + switch_score;
        }
      }
    }
  }
  double best = INF;
  for (int c2 = 0; c2 < num_centers_; c2++) {
    best = min(best, backward_scores_[0][0][c2] + lambda(0, c2));
  }
  assert(fabs(best - forward_scores_[num_timesteps_][num_states_][0]) < 1e-4);
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
  //path->push_back(cur_time + 1);

  while (cur_time > 0) {
    while (back_pointer_[cur_time][cur_state][cur_center] == -1) {
      check_score += score(cur_time, cur_center) + state_score(cur_time, cur_state);
      cur_time--;
      //cerr << cur_time << " " << cur_state << " " << check_score << endl;
      if (cur_time <= 0) {
        break;
      }
    }
    if (cur_time <= 0) {
      break;
    }
    assert(back_pointer_[cur_time][cur_state][cur_center] != -2);

    //cerr << cur_time << " " << cur_state << " " << cur_center << " " << check_score << " " << score(cur_time, cur_state) << " " <<  lambda(cur_state, cur_center) << endl;
    if (!(cur_time == num_timesteps_ && cur_state == num_states_)) {
      check_score += state_score(cur_time, cur_state) + score(cur_time, cur_center) + lambda(cur_state, cur_center);
    }
    cur_center = back_pointer_[cur_time][cur_state][cur_center];
    assert(cur_center >= -1);
    path->push_back(cur_time);
    centers->push_back(cur_center);
    cur_time--;
    cur_state--;
    //cerr << cur_time << " " << cur_state << " " << cur_center << " " << check_score << endl;
  }
  assert(cur_state == 0);
  path->push_back(0);
  check_score += lambda(0, cur_center) + score(0, cur_center) + state_score(0, 0);
  assert(fabs(check_score - total_score) < 1e-4);
  reverse(path->begin(), path->end());
  reverse(centers->begin(), centers->end());
  cerr << "Viterbi score: " << total_score << endl;
  cerr << "Alignment: ";
  for (uint p = 0; p < path->size(); ++p) { 
    cerr << (*path)[p] << " ";
  }
  cerr << endl;

  return total_score;
}
