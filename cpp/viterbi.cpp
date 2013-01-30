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

double LogAdd(double a, double b) {
  assert(a >= 0);
  assert(b >= 0);
  if (b > LOGZERO) {
    return a;
  } else if (a > LOGZERO) {
    return b;
  } else {
    //double c = -log(exp(-a) + exp(-b));
    //assert(c >= 0);
    double c = -max(-a, -b) + log1p(exp( -fabs(a - b) ));
    return c;
  }
}

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
      backward_scores_[m][i].resize(center_size, INF);
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

void Viterbi::ResetChart() {
  for (int c = 0; c < num_centers_; ++c) {
    forward_scores_[0][0][c] = 0.0;
  }
  backward_scores_[num_timesteps_][num_states_][0] = 0.0;
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
      if (i != 0 && m >= min_width_) {
        for (int c2 = 0; c2 < num_centers_; c2++) {
          // Reset the forward scores
          double trial;
          if (m == num_timesteps_ && i == num_states_)  {
            trial = forward_scores_[m - 1][i - 1][c2];
          } else  {
            trial = forward_scores_[m - min_width_][i - 1][c2];
          } 
          if (!use_sum_) { 
            if (trial < back_score) { 
              back_score = trial; 
              back_center = c2;
            }
          } else {
            back_score = LogAdd(back_score, trial);
          }
        }
      }
      for (int c = 0; c < num_centers_; ++c) {
        double w;

        // Last time step must transition to final state.
        if (m == num_timesteps_) {
          if (c == 0 && i == num_states_) {
            w = 0.0;
          } else {
            w = INF;
          }
        } else {
          // Last state only valid on last time step.
          if (i == num_states_) {
            w = INF;
          } else {
            w = scores_[m][c] + transition_score_[m][i][c];
          }
        }
        double stay_score = w + forward_scores_[m - 1][i][c];
        double pen = 0.0;
        if (i < num_states_) pen = lambda_[i][c];

        // Add the transition scores.
        double switch_score;
        if (m - min_width_ >= 0 && 
            m - (min_width_ * (i + 1)) >= 0 && 
            i != num_states_) {
          for (int pre = 1; pre < min_width_; ++pre) {
            pen += scores_[m - pre][c] + transition_score_[m - pre][i][c];
          }
          switch_score = w + back_score + pen;
        } else if (i == num_states_) {
          switch_score = w + back_score + pen;
        } else {
          switch_score = INF;
        }

        if (!use_sum_) {
          if (switch_score < stay_score ) {
            forward_scores_[m][i][c] = switch_score;
            back_pointer_[m][i][c] = back_center;
          } else {
            forward_scores_[m][i][c] = stay_score;
            back_pointer_[m][i][c] = -1;
          }
        } else {
          forward_scores_[m][i][c] = LogAdd(stay_score, switch_score);
        }
      }
    }
  } 
}

void Viterbi::BackwardScores() {
  for (int m = num_timesteps_ - 1; m >= 0 ; --m) {
    for (int i = num_states_ - 1; i >= 0 ; --i) {
      if (i > m) {
        continue;
      }
      double b = INF;
      if (i == num_states_ - 1 && m == num_timesteps_ - 1) {
        // Free to transition out of last state.
        b = 0.0;
      } else if (m + min_width_ < num_timesteps_)  {
        for (int c2 = 0; c2 < num_centers_; c2++) {
          double pen = 0.0;
          if (i + 1 != num_states_) {
            pen = lambda_[i + 1][c2];
          }
          if (!use_sum_) {
            b = min(b, backward_scores_[m + min_width_][i + 1][c2] + pen);
          } else {
            b = LogAdd(b, backward_scores_[m + min_width_][i + 1][c2] + pen);
          }
          backward_scores_[m][i][c2] = INF;
        }
      } 
      for (int c = 0; c < num_centers_; ++c) {
        double w = scores_[m][c] + transition_score_[m][i][c];
        double stay_score = backward_scores_[m + 1][i][c] + w;
        double cost = 0.0;
        if (m + min_width_ < num_timesteps_) {
          for (int pre = 0; pre < min_width_; ++pre) {
            cost += scores_[m + pre][c] + transition_score_[m + pre][i][c];
          }
        } else if (i == num_states_ - 1 && m == num_timesteps_ - 1) { 
          cost = w;
        } else {
          cost = INF;
        }
        double switch_score = b + cost;
        if (!use_sum_) {
          if (stay_score < switch_score) {
            backward_scores_[m][i][c] = stay_score;
          } else {
            backward_scores_[m][i][c] = switch_score;
          }
        } else {
          backward_scores_[m][i][c] = LogAdd(stay_score, switch_score);
        }
      }
    }
  }
  
  double best;
  if (!use_sum_) {
    best = INF;
    for (int c2 = 0; c2 < num_centers_; c2++) {
      best = min(best, backward_scores_[0][0][c2] + lambda(0, c2));
    }
    assert(fabs(best - forward_scores_[num_timesteps_][num_states_][0]) < 1e-4);
  } else {
//     best = 0.0;
//     for (int c2 = 0; c2 < num_centers_; c2++) {
//       best += backward_scores_[0][0][c2] + lambda(0, c2);
//     }
  }

}


void Viterbi::Marginals(vector<vector<vector<double> > > *marginals) {
  assert(use_sum_);
  marginals->resize(num_timesteps_ + 1);
  for (int m = 0; m < num_timesteps_; ++m) {
    (*marginals)[m].resize(num_states_);      
    for (int state = 0; state < num_states_; ++state) {
      (*marginals)[m][state].resize(num_centers_, 0.0);
    }
  }
  double normalization = 
    forward_scores_[num_timesteps_][num_states_][0];

  // The best score transitioning out of. 
  vector< vector<double> > best_back(num_timesteps_ + 1);
  for (int m = 0; m <= num_timesteps_; ++m) {
    best_back[m].resize(num_states_ + 1, INF);
    for (int i = 0; i <= num_states_; ++i) {
      if (i > m) continue;
      for (int c = 0; c < num_centers_; ++c) {
        double pen = 0.0;
        if (m < num_timesteps_ && i < num_states_) {
          pen = lambda(i, c);
        }
        double trial = backward_scores_[m][i][c] + pen;
        best_back[m][i] = LogAdd(best_back[m][i], trial); 
      }
    }
  }

  for (int m = 0; m < num_timesteps_; ++m) {
    for (int state = 0; state < num_states_; ++state) {
      for (int c = 0; c < num_centers_; ++c) {

        if (forward_scores_[m][state][c] > 1e7 || 
            backward_scores_[m + 1][state][c] > 1e7) {
          (*marginals)[m][state][c] = 0.0;
        } else {
          (*marginals)[m][state][c] =  
            exp(-((forward_scores_[m][state][c] + backward_scores_[m + 1][state][c]) 
                  - normalization));
        }

        if (forward_scores_[m][state][c] > 1e7 || 
            best_back[m + 1][state + 1] > 1e7) {
          (*marginals)[m][state][c] += 0.0;
        } else if (m - (min_width_ * (state + 1)) >= 0) {
          (*marginals)[m][state][c] +=  
            exp(-((forward_scores_[m][state][c] + best_back[m + 1][state+1]) 
                  - normalization));
        }
        assert((*marginals)[m][state][c] >= 0.0);
        assert((*marginals)[m][state][c] <= 1.0);
      }
    }
  }
}

void Viterbi::MinMarginals(vector<vector<double> > *min_marginals) {
  assert(!use_sum_);
  min_marginals->resize(num_states_ + 1);
  for (int i = 0; i < num_states_; ++i) {
    (*min_marginals)[i].resize(num_centers_, INF);
  }

  vector< vector<double> > best_back(num_timesteps_ + 1);

  clock_t start = clock();
  double best_score = forward_scores_[num_timesteps_][num_states_][0];
  for (int m = 0; m <= num_timesteps_; ++m) {
    best_back[m].resize(num_states_ + 1, INF);
    for (int i = 0; i <= num_states_; ++i) {
      if (i > m) continue;
      for (int c = 0; c < num_centers_; ++c) {
        double pen = 0.0;
        if (m < num_timesteps_ && i < num_states_) {
          pen = lambda(i, c);
        }
        double trial = backward_scores_[m][i][c] + pen;

        if (trial < best_back[m][i]) {
          best_back[m][i] = trial;
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
        double trial2;
        if (m - (min_width_ * (i + 1)) >= 0) {
          trial2 = forward_scores_[m][i][c] + best_back[m + 1][i + 1];
        } else {
          // Not a valid transition.
          trial2 = INF;
        }
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
  cerr << "TIME: Min-Marginals " << end - start << endl;
  cerr << "VITERBI: best score is " << best_score << endl; 
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
    if (!(cur_time == num_timesteps_ && cur_state == num_states_)) {
      for (int t = 0; t < min_width_; ++t) {
        check_score += transition_score(cur_time - t, cur_state, cur_center) 
          + score(cur_time - t, cur_center);
      }
      check_score += lambda(cur_state, cur_center);
    }
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
