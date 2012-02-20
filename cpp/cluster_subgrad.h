#ifndef CLUSTER_SUBGRAD_H
#define CLUSTER_SUBGRAD_H

#include "cluster_problem.h"
#include "data_point.h"

struct ClusterIndex {
  ClusterIndex(int _ball, int _problem, int _state, int _partition) 
  : ball(_ball), problem(_problem), state(_state), partition(_partition) {}
  ClusterIndex() {} ;
  int ball;
  int problem;
  int state;
  //int hidden;
  int partition;
};

class ClusterSubgrad {
 public:
 ClusterSubgrad(const ClusterSet &cs, int num_balls) 
   : cs_(cs), 
    num_balls_(num_balls),
    structure_(num_balls_ * cs_.problems_size() 
               * cs_.max_states() * cs_.num_hidden()) {}

  const DataPoint &ToVector() const { return structure_; } 

  static ClusterSubgrad *FromVector(const ClusterSet &cp,
                                    int num_balls,
                                    const DataPoint &wvector
                                    );  

  void On(const ClusterIndex &cluster_index) {
    int index = to_index(cluster_index);
    structure_[index] = 1;
  }

  void Update(const ClusterIndex &cluster_index, double score) {
    int index = to_index(cluster_index);
    structure_[index] += score;
  }

  void Set(const ClusterIndex &cluster_index, double score) {
    int index = to_index(cluster_index);
    structure_[index] = score;
  }

  double score(const ClusterIndex &cluster_index) const {
    int index = to_index(cluster_index);
    return structure_[index];
  }

  int to_index(const ClusterIndex &cluster_index) const {
    return 
      cluster_index.ball * cs_.problems_size() * cs_.num_hidden() * cs_.max_states() +
      cluster_index.problem * cs_.num_hidden() * cs_.max_states() +
      cluster_index.state * cs_.num_hidden() + 
      cluster_index.partition;
  }

  void index(int index, ClusterIndex *cluster_index) const {
    cluster_index->ball = index / (cs_.num_hidden() * cs_.max_states() * cs_.problems_size());
    cluster_index->problem = index / (cs_.num_hidden() * cs_.max_states()) % cs_.problems_size();
    cluster_index->state = (index / cs_.num_hidden()) % cs_.max_states();
    cluster_index->partition = index % cs_.num_hidden();
  }

  static void Align(const ClusterSubgrad &s1, 
                    const ClusterSubgrad &s2);

 private:
  const ClusterSet &cs_;

  int num_balls_;

  // Map from each problem state to hidden.
  DataPoint structure_;

  // Formatter for subgrad.
  friend std::ostream &operator << (std::ostream &os, const ClusterSubgrad &subgrad); 
};

std::ostream &operator<<(std::ostream &os, const ClusterSubgrad &subgrad);

#endif
