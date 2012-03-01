#ifndef DISTANCES_H
#define DISTANCES_H

#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <set>
#include <vector>
#include "span_chart.h"
using namespace std;
typedef boost::numeric::ublas::mapped_vector<double> DataPoint;

double dist(const DataPoint&, const DataPoint&);

struct Center {
  const DataPoint &point() const { return *point_; } 
  vector<bool> address;
  DataPoint *point_;
private:

};

class ThinDistanceHolder {
 public:
  ThinDistanceHolder(const vector<Center> &center_set,
                     const vector<const DataPoint *> &time_steps) :
  center_set_(center_set), 
    time_steps_(time_steps)
    {
      cerr << "Thin Distance Holder with "
           << center_set_.size() << " " << time_steps_.size() << endl;
    } 

  void Initialize();

  double get_distance(int time, int center) const {
    //cassert(initialized_);
    return distance_[time][center];
  }
 private:
  // Distance between each time step to a center.
  vector<vector<double > > distance_;

  const vector<Center> &center_set_;
  const vector<const DataPoint *> &time_steps_;
};

// Precompute the distance from each span to each cluster center. 
class DistanceHolder {
 public:
  DistanceHolder(const vector<Center> &center_set,
                 const vector<const DataPoint *> &time_steps, 
                 int width_limit) :
  width_limit_(width_limit),
  center_set_(center_set), 
    time_steps_(time_steps), 
    initialized_(false){
    cerr << "Distance Holder with " << width_limit << " " 
         << center_set_.size() << " " << time_steps_.size() << endl;
  } 

  ~DistanceHolder() { delete distance_; }

  void Initialize();

  // Compute the distance between a time index and a cluster index. 
  // Cache for future access.  
  double Distance(int time_index, int cluster_index);

  void ComputeDistances();
  void ComputeLocalDistances();  

  double get_distance(int start, int offset, int center) const {
    return distance_->get(start, offset)[center];
  }

  double get_distance(int time, int center) const {
    assert(initialized_);
    return distance_cache_[time][center];
  }

  bool is_distance_used(int start, int offset, int center) const {
    return distance_->get(start, offset)[center] < (offset - start) * 5.0;
  }

  double span_centroid_cost (int s, int o) const { return span_centroid_cost_[s][o]; }

  bool is_pruned(int s, int o, int h) const {
    if (s == 0) {
      return false;
    } 
    return hidden_allowed_->get(s, o).find(h) == 
      hidden_allowed_->get(s, o).end();
  }

  bool is_pruned(int s, int o) const {
    if (o > width_limit_) {
      return true;
    } 

    int n = time_steps_.size();
    if (s + o >= n) {
      return true;
    }
    if (s == 0) {
      return false;
    } 

    return pruned_->get_const(s, o);
  }

 private:
  int width_limit_;

  const vector<Center> &center_set_;
  const vector<const DataPoint *> &time_steps_;
  
  // The distance from each span to each possible center.
  SpanChart<vector <double> > *distance_;

  // Pruning set for spans.
  SpanChart<bool> *pruned_;
  SpanChart<set<int> > *hidden_allowed_;

  // Distance between each time step to a center.
  vector<vector<double > > distance_cache_;
  
  // Sum of distances from a span to its centroid. 
  vector<vector<double> > span_centroid_cost_;  

  bool initialized_;
};

class BallHolder {
 public:
  BallHolder(vector<int> epsilons, 
             const vector<Center> &center_set):
    epsilons_(epsilons),
    center_set_(center_set) {}

  // Construct the mapping from each center to nearby centers.
  void ConstructRelations();

  int nearby_size(int ball, int partition) const { 
    return shared_balls_[ball][partition].size(); 
  }
 
  int nearby(int ball, int partition, int index) const { 
    assert(partition < partition_size(ball));
    return shared_balls_[ball][partition][index];
  }
  
  // The number of total balls.
  int balls_size() const { return epsilons_.size(); }

  // Return the partition that a center is in. 
  int partition_for_center(int ball, int center) const {
    return partition_for_center_[ball][center]; 
  }

  // The number of partitions in the current ball.
  int partition_size(int ball) const { return shared_balls_[ball].size(); } 

  // The number of centers available.
  int center_size() const { return center_set_.size(); } 

  // The epsilon of the given ball.
  double ball_epsilon(int ball) const { return epsilons_[ball]; }

 private:
  // The distance for two centers to be considered "close".
  vector<int> epsilons_;

  // A mapping from each ball to the nearby balls.
  vector<vector<vector<int> > > shared_balls_;

  // The partitions for each center.
  vector<vector<int> > partition_for_center_;

  // The set of all available centers.
  const vector<Center> &center_set_;
};

std::ostream &operator<<(std::ostream &os, const BallHolder &holder);

#endif
