#include "hidden_presolve.h"

#define INF 1e20

void HiddenHolder::set_reparameterization(const vector<vector<double> > *reparameterization) {
   reparameterization_ = reparameterization;
   assert((int)reparameterization_->size() == cp_.num_states);
   //assert((int)(*reparameterization_)[0].size() == cp_.num_hidden);
}


HiddenHolder::HiddenHolder(const ClusterProblem &cp,
                           const DistanceHolder &distances) 
  : cp_(cp), distances_(distances) {
  best_score_ = new SpanChart<vector<double> >(cp_.num_steps, cp.width_limit());
  best_hidden_ = new SpanChart<vector<int> >(cp_.num_steps, cp.width_limit());
  primal_score_ = new SpanChart<vector<double> >(cp_.num_steps, cp.width_limit()); 
  for (int s = 0; s < cp_.num_steps; ++s) {
    for (int o = 0; o < cp_.width_limit(); ++o) {
      if (is_pruned(s, o)) {
        continue;
      }
      best_score_->get(s, o).resize(cp.num_states);
      best_hidden_->get(s, o).resize(cp.num_states);
      primal_score_->get(s, o).resize(cp.num_states);
    }
  }

  // Initialize all scores to INF.
  for (int s = 0; s < cp_.num_steps; ++s) {
    for (int o = 0; o < cp_.width_limit(); ++o) {
      if (is_pruned(s, o)) {
        continue;
      }
      for (int i = 0; i < cp_.num_states; ++i) {
        best_score_->get(s, o)[i] = INF;
      }
    }
  }

  // used_.resize(cp_.num_steps);
  // //, cp.width_limit);
  // for (int s = 0; s < cp_.num_steps; ++s) {
  //   used_[s].resize(cp_.width_limit());
  //   for (int o = 0; o < cp_.width_limit(); ++o) {
  //     used_[s][o].clear();
  //     for (int h = 0; h < cp_.num_hidden(); ++h) {
  //       //double distance = distances_.get_distance(s, o, h);
  //       used_[s][o].push_back(h);
  //     }
  //   }
  // }
}

void HiddenHolder::ComputeBestHidden() {
  clock_t start = clock();
  for (int s = 0; s < cp_.num_steps; ++s) {
    for (int o = 0; o < cp_.width_limit(); ++o) {
      if (is_pruned(s, o)) {
        continue;
      }
      for (int i = 0; i < cp_.num_states; ++i) {
        best_score_->get(s, o)[i] = INF;
        best_hidden_->get(s, o)[i] = 0;
        primal_score_->get(s, o)[i] = 0;
      }
    }
  }
  int count = 0;
  int prune_count = 0;
  for (int s = 0; s < cp_.num_steps; ++s) {
    for (int o = 0; o < cp_.width_limit(); ++o) {
      if (is_pruned(s, o)) {
        continue;
      }
      for (int h = 0; h < cp_.num_hidden(0); ++h) {
        if (distances_.is_pruned(s, o, h)) {
          prune_count++;
          continue;
        }
        double distance = distances_.get_distance(s, o, h); 
        count++;
        for (int i = 0; i < cp_.num_states; ++i) {
          //int p = cp_.MapState(i);
          double cost = distance + (*reparameterization_)[i][h]; 
          if (cost < best_score_->get(s, o)[i]) {
            best_hidden_->get(s, o)[i] = h;
            best_score_->get(s, o)[i] = cost; 
            primal_score_->get(s, o)[i] = distance; 
          }

        }
      }

      // for (int i = 0; i < cp_.num_states; ++i) {
      //   int p = cp_.MapState(i);
      //   for (int index = 0; index < cp_.num_hidden(p); ++index) {
      //     int h = cp_.hidden_for_type(p, index);
      //     double distance = distances_.get_distance(s, o, h);
      //     double cost = distance + (*reparameterization_)[i][index]; 
      //     if (cost < best_score_->get(s, o)[i]) {
      //       best_hidden_->get(s, o)[i] = index;
      //       best_score_->get(s, o)[i] = cost; 
      //       primal_score_->get(s, o)[i] = distance; 
      //     }
      //     ++count;
      //   }
      // }
    }
  }
  clock_t end = clock();
  cerr << "Presolving semi-markov " << end - start << " " << count << " " << prune_count << " " << cp_.num_hidden(0) << endl; 
}

