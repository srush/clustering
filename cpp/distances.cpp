#include "distances.h"

// void ThinDistanceHolder::Initialize(const vector<Center> &center_set,
//                                     const vector<const DataPoint *> &time_steps) {
//   int n = time_steps.size();

//   distance_.resize(n);
//   for (int s = 0; s < n; ++s) {
//     distance_[s].resize(center_set.size());
//     for (uint c = 0; c < center_set.size(); ++c) {
//       distance_[s][c] =
//         dist(*time_steps[s], center_set[c].point());
//     }
//   }
// }

void ThinDistanceHolder::Initialize(const vector<Center> &center_set,
                                    const vector<vector<const DataPoint *> > &time_steps) {
  int n = time_steps.size();

  distance_.resize(n);
  for (int s = 0; s < n; ++s) {
    distance_[s].resize(center_set.size(), 0.0);
    for (uint c = 0; c < center_set.size(); ++c) {
      for (uint i = 0; i < time_steps[s].size(); ++i) {
        distance_[s][c] +=
          dist(*time_steps[s][i], center_set[c].point());
      }
    }
  }
}


void DistanceHolder::Initialize() {
  int n = time_steps_.size();

  // Build chart.
  pruned_ = new SpanChart<bool>(time_steps_.size(), 
                                width_limit_);
  
  hidden_allowed_ = new SpanChart<set<int> >(time_steps_.size(), 
                                        width_limit_);

  distance_ = new SpanChart<vector <double> >(time_steps_.size(), 
                                              width_limit_);
  for (int s = 0; s < n; ++s) {
    for (int o = 0; o < width_limit_; ++o) {
      distance_->get(s, o).resize(center_set_.size());
    }
  }
}

double DistanceHolder::Distance(int time_index, int cluster_index) {
  if (distance_cache_[time_index][cluster_index] == -1) {
    distance_cache_[time_index][cluster_index] = 
      dist(*time_steps_[time_index], center_set_[cluster_index].point());
  }
  return distance_cache_[time_index][cluster_index];
}

void DistanceHolder::ComputeDistances() {
  int n = time_steps_.size();
  // Initialize the distance cache to NULL.
  distance_cache_.resize(n);
  for (int s = 0; s < n; ++s) {
    distance_cache_[s].resize(center_set_.size());
    for (uint q = 0; q < center_set_.size(); ++q) {
      distance_cache_[s][q] = -1;
    }
  }

  // Prime the cache.
  for (int s = 0; s < n; ++s) {
    for (uint q = 0; q < center_set_.size(); ++q) {
      Distance(s, q); 
    }
  }
  

  cerr << "compute dist" << endl;
  for (uint q = 0; q < center_set_.size(); ++q) {
    for (int s = 0; s < n; ++s) {
      for (int o = 0; o < width_limit_; ++o) {
        if (s + o >= n) continue;  
        vector<double> &centers = distance_->get(s, o);
        if (o == 0) {
          centers[q] = Distance(s + o, q);
        } else {
          centers[q] = distance_->get(s, o - 1)[q] + Distance(s + o, q);
        }
        if (centers[q] / (o + 1) < 5.0) {
          hidden_allowed_->get(s, o).insert(q);
        } 
      }
    }
  }

  SpanChart<DataPoint > sum_points(time_steps_.size(), 
                                   width_limit_);

  for (int s = 0; s < n; ++s) {
    for (int o = 0; o < width_limit_; ++o) {
      if (s + o >= n) continue;  
      if (o == 0) {
        sum_points.set(s, o, *time_steps_[s]);
      } else { 
        sum_points.set(s, o, 
                       sum_points.get_const(s, o - 1 ) + *time_steps_[s + o]);
      }
    }
  }

  // Compute pruning. 
  for (int s = 0; s < n; ++s) {
    for (int o = 0; o < width_limit_; ++o) {
      if (s + o >= n) continue;  
      if (o == 0) {
        pruned_->set(s, o, false);
        continue;
      }
      double total = 0.0;
      const DataPoint &centroid = sum_points.get_const(s, o) / (float)(o +1);
      for (int o2 = 0; o2 <= o; ++o2) {
        total += dist(centroid, *time_steps_[s + o2]);
      }
      double average = total / (float) o; 
      bool prune = (average > 5.0);
      pruned_->set(s, o, prune);
    }
  }
  
  //distance_cache_.clear();
  initialized_ = true;
}

void DistanceHolder::ComputeLocalDistances() {
  int n = time_steps_.size();
  span_centroid_cost_.resize(n);
  for (int s = 0; s < n; ++s) {
    span_centroid_cost_[s].resize(width_limit_);
    DataPoint sum = *time_steps_[s];
    for (int o = 1; o < width_limit_; ++o) {
      sum += *time_steps_[s + o];
      DataPoint centroid = sum / (o + 1);
      double cost = 0;
      for (int o2 = 0; o2 < width_limit_; ++o) {
        cost += dist(centroid, *time_steps_[s + o2]);
      }
      span_centroid_cost_[s][o] = cost; 
    }
  }
}

double dist(const DataPoint &a, const DataPoint &b) {
  double d = norm_2(a - b);
  return d * d;
  //return norm_1(a - b);
}


void BallHolder::ConstructRelations() {
  int Q = center_set_.size();
  shared_balls_.resize(epsilons_.size());
  partition_for_center_.resize(epsilons_.size());
  for (uint e = 0; e < epsilons_.size(); ++e) {
    shared_balls_[e].resize((int)pow(2.0, epsilons_[e]) + 1);
    partition_for_center_[e].resize(Q);
  }
 
  for (int e = 0; e < (int)epsilons_.size(); ++e) {
    int epsilon = epsilons_[e];
    for (int q1 = 0; q1 < Q; ++q1) {
      int part = 0;
      for (int a = 0; a < epsilon; ++a) {
        part += 
          (a >= (int)center_set_[q1].address.size() ||
           !center_set_[q1].address[a]) ? 0 : (int)pow(2.0, a);
      }
      partition_for_center_[e][q1] = part;
      shared_balls_[e][part].push_back(q1);
    }
  }
}


std::ostream &operator<<(std::ostream &os, const BallHolder &holder) {
  for (int b = 0; b < holder.balls_size(); ++b) {
    for (int part = 0; part < holder.partition_size(b); ++part) {
      os << b << " " << part << " ";
      for (int n = 0; n < holder.nearby_size(b, part); ++n) {
        int q2 = holder.nearby(b, part, n);
        os << " " << q2;
      }
      os << endl;
    }
  }
  return os;
}
