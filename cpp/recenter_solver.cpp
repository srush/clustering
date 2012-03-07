
#include "recenter_solver.h"

bool RecenterSolver::IsChosen(int center, int use_sequence) const {
  for (int choice = 0; choice < num_choices_ - 1; ++choice) {
    if (best_hidden_[use_sequence][choice] == center) {
      return true;
    }
  }
  return false;
}

double RecenterSolver::AllBut(int center, int center2, int num_needed) const {
  double score = 0.0;
  int have = 0;
  for (int choice = 0; have < num_needed; ++choice) {
    if (best_hidden_[0][choice] == center || best_hidden_[0][choice] == center2) 
      continue;
    score += best_score_[0][choice];
    ++have;
  }
  return score + all_off_;
}


double RecenterSolver::Solve() {
  all_off_ = 0.0;
  int num_centers = center_param_->size();
  double total_score = 0.0;
  for (int use_segment = 0; use_segment <= 1; ++use_segment) {
    best_score_[use_segment].clear();
    best_hidden_[use_segment].clear();
    best_score_[use_segment].resize(num_choices_ + 1, 1e20);
    best_hidden_[use_segment].resize(num_choices_ + 1, -1);
    for (int choice = 0; choice <= num_choices_; ++choice) {
      for (int q = 0; q < num_centers; ++q) {
        bool seen = false;
        for (int choice2 = 0; choice2 < num_choices_; ++choice2) {
          if (choice2 < choice && best_hidden_[use_segment][choice2] == q) {
            seen = true;
            break;
          } 
          if (use_segment && choice2 < num_choices_ - 1 && 
              best_hidden_[0][choice2] == q) {
            seen = true;
            break;
          }
        }
        if (seen) continue;
        assert(fabs((*center_param_)[q][0]) < 1e-4);
        double score = 0.0;
        score = (*center_param_)[q][1] - (*center_param_)[q][0];
        if (use_segment) {
          score += (*segment_param_)[q];
        }
        if (score < best_score_[use_segment][choice]) {
          best_score_[use_segment][choice] = score;
          best_hidden_[use_segment][choice] = q;
        }
      }
    }
  }
  for (int q = 0; q < num_centers; ++q) {
    total_score += (*center_param_)[q][0];
    all_off_ += (*center_param_)[q][0];
  }
  assert(fabs(all_off_) < 1e-4);
  for (int choice = 0; choice < num_choices_ - 1; ++choice) {
    total_score += best_score_[0][choice];
  }

  double best = best_score_[1][0];

  // Either add a segment from the list, or choose the best segment.
  for (int choice = 0; choice < num_choices_ - 1; ++choice) {
    double trial = (*segment_param_)[best_hidden_[0][choice]] + 
      (*center_param_)[best_hidden_[0][num_choices_ - 1]][1];

    if (trial < best) {
      best = trial;
    }
  }  
  total_score += best;
  //cerr << "RECENTER " <<type_ << " "<< best_hidden_[1][0]  << (*center_param_)[best_hidden_[1][0]][1] << " " << (*segment_param_)[best_hidden_[1][0]] << " " << (*center_param_)[best_hidden_[1][1]][1] << " " << AllBut(best_hidden_[1][0], -1, 1) << " " << best_hidden_[0][0] << " " << best_score_[0][0] << " " << (*segment_param_)[best_hidden_[0][0]] << " "<< (*center_param_)[best_hidden_[0][0]][1]<< " " << (*segment_param_)[best_hidden_[0][0]] +(*center_param_)[best_hidden_[0][0]][1] << " " << total_score << endl;
  //cerr << best_hidden_[0][1] << " " << best_score_[0][1] << endl;
  return total_score;
}

//   double total_cost = 0.0;
//   double best_score = 1e20;
//   best_center_ = -1;
//   double backup_best_score = 1e20;
//   best_backup_center_ = -1;

//   // Find the best center. 
//   for (int center = 0; center < num_centers; ++center) {
//     if ((*center_param_)[center][1] < (*center_param_)[center][0]) {
//       total_cost += (*center_param_)[center][1] - (*center_param_)[center][0];
//       double trial = (*segment_param_)[center];
//       if (trial < best_score) {
//         backup_best_score = best_score;
//         best_backup_center_ = best_center_;
//         best_score = trial;
//         best_center_ = center;
//       } else if (trial < backup_best_score) {
//         backup_best_score = trial;
//         best_backup_center_ = center;        
//       }

//       cerr << center << " " << (*center_param_)[center][1] << endl;
//     } else {
//       double trial = (*segment_param_)[center] 
//         + (*center_param_)[center][1] - (*center_param_)[center][0];
//       if (trial < best_score) {
//         backup_best_score = best_score;
//         best_backup_center_ = best_center_;        
//         best_score = trial;
//         best_center_ = center;
//       } else if (trial < backup_best_score) {
//         backup_best_score = trial;
//         best_backup_center_ = center;        
//       }
//     }
//   }
//   for (int center = 0; center < num_centers; ++center) {
//     total_cost += (*center_param_)[center][0];
//   }
//   cerr << total_cost << " " << best_center_ << " " << best_score << endl;
//   return total_cost + best_score;
// }
  
// Compute max-marginals.

double RecenterSolver::MaxMarginals(vector<double> *segment_mu,
                                    vector<vector<double> > *center_mu) {
  //clock_t start = clock();
  int num_centers = center_param_->size();
  segment_mu->resize(num_centers);
  center_mu->resize(num_centers);
  double best_score = Solve();

  //int best_seg = best_hidden_[1][0]; 
  // double segment_score = (*segment_param_)[best_seg] 
  //   + (*center_param_)[best_seg][1] - (*center_param_)[best_seg][0];

  //int backup_best_seg = best_hidden_[1][1]; 
  // double backup_segment_score = (*segment_param_)[backup_best_seg]
  //   + (*center_param_)[backup_best_seg][1] - (*center_param_)[backup_best_seg][0];


  for (int center = 0; center < num_centers; ++center) {
    (*center_mu)[center].resize(2);
    (*segment_mu)[center] = AllBut(center, -1, num_choices_ - 1)
      + (*segment_param_)[center] 
      + (*center_param_)[center][1] - (*center_param_)[center][0];
    
    if (IsChosen(center, 0)) {
      // Center is definitely on. 
      (*center_mu)[center][1] = best_score;
      // if (best_hidden_[1][0] == center) {
      //   (*center_mu)[center][0] = AllBut(center, backup_best_seg, num_choices_ - 1) 
      //     + backup_segment_score;
      // } else {
      //   (*center_mu)[center][0] = AllBut(center, best_seg, num_choices_ - 1) 
      //     + segment_score;
      // }
    } else {
      // cost to turn the center on. 
      double turn_on;

      if (num_choices_ > 1) {
        turn_on = AllBut(-1, -1, num_choices_ - 2) + 
          (*center_param_)[center][1] - (*center_param_)[center][0];
        double best = best_score_[1][0];
        // Either add a segment from the list, or choose the best segment.
        for (int choice = 0; choice < num_choices_ - 1; ++choice) {
          double trial = (*segment_param_)[best_hidden_[0][choice]] +
            + (*center_param_)[best_hidden_[0][num_choices_ - 2]][1]; 
          if (trial < best) {
            best = trial;
          }
        }
        turn_on += best;
      }  else {
        turn_on = AllBut(center, -1, num_choices_ - 1) + 
          (*segment_param_)[center] + 
          (*center_param_)[center][1] - (*center_param_)[center][0];
      }
      (*center_mu)[center][1] = min(turn_on, (*segment_mu)[center]);
      // if (best_hidden_[1][0] == center) {
      //   (*center_mu)[center][0] = AllBut(center, backup_best_seg, num_choices_ - 1) 
      //     + backup_segment_score;
      //   assert((*center_mu)[center][0] - best_score > -1e-4);
      // } else {
      //   (*center_mu)[center][0] = AllBut(center, best_seg, num_choices_ - 1) 
      //     + segment_score;
      //   //cerr << center << " " << best_seg << " " << (*center_mu)[center][0] << " " << best_score << endl;
      //   assert((*center_mu)[center][0] - best_score > -1e-4);
      // }
    }
  }
  for (int center = 0; center < num_centers; ++center) {
    assert((*center_mu)[center][1] - best_score > -1e-4);
    //cerr << best_score << " " << (*center_mu)[center][0] << endl;
    //assert((*center_mu)[center][0] - best_score > -1e-4);
    assert((*segment_mu)[center] - best_score > -1e-4);
  }
  // assert(fabs((*segment_mu)[best_seg] - best_score) < 1e-4);
  // assert(fabs((*center_mu)[best_seg][1] - best_score) < 1e-4);
  // if (backup_segment_score > segment_score) {
  //   assert((*segment_mu)[backup_best_seg] - 
  //          best_score > -1e-4);
  // }
  //cerr << (*center_mu)[68][1] << " " << (*center_mu)[29][1] << endl;
  //cerr << (*segment_mu)[68] << " " << (*segment_mu)[29] << endl;
  // assert(fabs((*segment_mu)[best_backup_center_] - 
  //             (best_score - segment_score + backup_segment_score)) < 1e-4);
  //cerr << "Recenter Marg: " << clock() - start << endl;
  return best_score;
}                      

