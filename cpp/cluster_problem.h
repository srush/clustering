#ifndef CLUSTER_PROBLEM_H
#define CLUSTER_PROBLEM_H

#include <vector>
#include <iostream>
#include <assert.h>
using namespace std;

struct ClusterSet;

struct ClusterProblem {
  ClusterProblem(int _num_steps, 
                 int _num_states, 
                 const vector<int> &state_to_type);
  
  int MapState(int state) const {
    return state_to_type_[state];
  }

  // Pruning.
  void prune_state_range(int s, int start, int end);

  int width_limit() const;
  int num_hidden(int type) const;
  int hidden_for_type(int type, int hidden_index) const;

  void set_gold_segmentation(vector<int> gold_alignment) {
    gold_alignment_ = gold_alignment;
    has_gold_ = true;
  }

  void GoldStateAlign(int state, int *s, int *e) const {
    assert(state < num_states);
    (*s) = gold_alignment_[state];
    (*e) = gold_alignment_[state + 1];
  }

  int num_steps;
  int num_states;
  // The master cluster set. 
  const ClusterSet *cluster_set_;

private:

  // Mapping from the states to their underlying type.
  vector<int> state_to_type_;

  // The gold segmentation for this sentence.
  bool has_gold_; 
  vector<int> gold_alignment_;

};



struct ClusterSet {
public:
  ClusterSet(const vector<ClusterProblem *> cluster_problems,
             int num_hidden,
             int num_types,
             int num_balls,
             int width_limit);
  
  // Creates an reparameterization array for the cluster problem.
  vector<vector<vector<double > > > *CreateReparameterization() const;

  int problems_size() const { return cluster_problems_.size(); } 
  const ClusterProblem &problem(int index) const {
    return *cluster_problems_[index];
  }

  int max_states() const { return max_states_; }

  int num_hidden(int type) const { return type_hidden_[type].size(); }

  int hidden_for_type(int type, int hidden_index) const {
    assert(hidden_index < num_hidden(type));
    return type_hidden_[type][hidden_index];
  } 

  int num_types() const { return num_types_; }

  //void set_num_balls(int num_balls) { num_balls_ = num_balls; } 

  //int num_balls() const { return num_balls_; }
  
  int width_limit() const {  return width_limit_; }

  // Pruning.
  void add_type_hidden(int type, int hidden) { 
    type_hidden_[type].push_back(hidden);
  } 

  void clear_hidden_types() {
    for (int type = 0; type < num_types(); ++type) {
      type_hidden_[type].clear();
    }
  }

private:
  vector<ClusterProblem *> cluster_problems_; 

  int width_limit_;
  int max_states_;
  int num_hidden_;
  int num_types_;

  // List of hidden variables allowed for a type.
  vector<vector<int> > type_hidden_; 

  // Number of epsilon ball constraints to use.
  //int num_balls_;
  
};

#endif
