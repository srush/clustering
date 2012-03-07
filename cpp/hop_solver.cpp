
#include "hop_solver.h"

double HOPSolver::Solve() {
  double total_score = 0.0;
  best_score_.clear();
  best_hidden_.clear();
  best_score_.resize(num_choices_ + 1, 1e20);
  best_hidden_.resize(num_choices_ + 1, -1);
  for (int choice = 0; choice <= num_choices_; ++choice) {
    for (uint q = 0; q < reparameterization_->size(); ++q) {
      bool seen = false;
      for (int choice2 = 0; choice2 < choice; ++choice2) {
        if (best_hidden_[choice2] == (int)q) {
          seen = true;
          break;
        }
      }
      if (seen) continue;
      double score = (*reparameterization_)[q][1] - (*reparameterization_)[q][0];
      if (score < best_score_[choice]) {
        best_score_[choice] = score;
        best_hidden_[choice] = (int)q;
      }
    }
  }
  for (uint q = 0; q < reparameterization_->size(); ++q) {
    total_score += (*reparameterization_)[q][0];
  }
  for (int choice = 0; choice < num_choices_; ++choice) {
    int q = best_hidden_[choice];
    total_score += (*reparameterization_)[q][1] - (*reparameterization_)[q][0];
  }
  //cerr << "HOPType " << type_ << " " << best_hidden_[0] << " " << total_score << endl;
  return total_score;
}
  
// Compute max-marginals.
double HOPSolver::MaxMarginals(vector<vector<double> > *mu) {
  double total_score = Solve();
  mu->resize(reparameterization_->size());
  for (uint q = 0; q < reparameterization_->size(); ++q) {
    (*mu)[q].resize(2);
    bool found = false;
    for (int choice = 0; choice < num_choices_; ++choice) {
      if (best_hidden_[choice] == (int)q) {
        found = true;
        break;
      }
    }
    if (found) {
      (*mu)[q][1] = total_score;
      (*mu)[q][0] = total_score + best_score_[num_choices_] 
        - ((*reparameterization_)[q][1] - (*reparameterization_)[q][0]);
    } else {
      (*mu)[q][1] = total_score - best_score_[num_choices_ - 1] 
        + (*reparameterization_)[q][1] - (*reparameterization_)[q][0];
      (*mu)[q][0] = total_score;
    }
  }
  for (uint q = 0; q < reparameterization_->size(); ++q) {
    for (int b = 0; b <=1; ++b) {
      assert((*mu)[q][b] - total_score > -1e-4); 
    }
  }
  assert(fabs((*mu)[best_hidden_[0]][1] - total_score) < 1e-4); 
  assert(fabs((*mu)[best_hidden_[num_choices_]][0] - total_score) < 1e-4); 
  return total_score;
}

//double Rescore(const SpeechSolution &solution) const;

//double Check(int type, int hidden);
                      

