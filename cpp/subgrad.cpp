#include "subgrad.h"
#include <time.h>

void SubgradSolver::Run() {
  clock_t start = clock();
  while (RunOneRound() && round_ < iterations_) {
    round_++;
    clock_t end = clock();
    cerr << "TIME: Iteration time " << end - start << endl;
    start = clock();
  }
}

double ExponentialWaitRate::get_rate(const SubgradInfo& info) {
  int last = info.duals.size() - 1;
  if (last < 2) {
    current_rate_ = start_rate_;
    bad_count_ = 0;
  } else {
    if (info.duals[last] < info.duals[last -1]) {
      bad_count_++;
    }
    if (bad_count_ > 10) {
      current_rate_ = current_rate_ * 0.9;
      bad_count_ = 0;
    } 
  }
  return current_rate_;
}


bool SubgradSolver::RunOneRound() {
  SubgradResult result;
  SubgradInfo info(round_, dual_values_, primal_values_);
  clock_t start = clock();
  problem_->Solve(info, &result);
  clock_t end = clock();
  cerr << "TIME: Solve time " << end - start << endl;
  start = clock();
  if (round_ == 1) {
    best_dual_value_ = result.dual_value;
    best_primal_value_ = result.primal_value;
  }
  if (result.dual_value > best_dual_value_) {
    best_dual_value_ = result.dual_value;
  }
  if (result.primal_value < best_primal_value_) {
    best_primal_value_ = result.primal_value;
  }
  cerr << "Round: " << round_ << endl;
  cerr << "Subgrad norm: " << norm_1(result.subgradient) << endl;
  cerr << "Dual value: " <<  best_dual_value_ << " " <<result.dual_value << endl;
  cerr << "Primal value: " << best_primal_value_ << " " << result.primal_value << endl;
  if (norm_1(result.subgradient) == 0.0) {
    return false;
  }

  // Update the solver 
  double alpha = rate_->get_rate(info) / norm_1(result.subgradient);
  cerr << "Alpha: " << alpha << endl;
  end = clock();
  cerr << "TIME: Subgrad Time " << end - start << endl;
  
  start = clock();
  problem_->Update(result.subgradient, alpha);
  end = clock();
  cerr << "TIME: Update Time " << end - start << endl;

  dual_values_.push_back(result.dual_value);
  return true;
}

