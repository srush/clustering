#ifndef CLUSTER_PROBLEM_H
#define CLUSTER_PROBLEM_H

#include <vector>
#include <iostream>
using namespace std;

struct ClusterProblem{
  ClusterProblem(int _num_steps, 
                 int _num_states, 
                 const vector<int> &state_to_type) 
  : num_steps(_num_steps), 
    num_states(_num_states), 
    state_to_type_(state_to_type)
  {}
  
  int MapState(int state) const {
    return state_to_type_[state];
  }

  int num_steps;
  int num_hidden;
  int width_limit;
  int num_states;
  int num_types;
private:
  vector<int> state_to_type_;
};

struct ClusterSet {
public:
ClusterSet(const vector<ClusterProblem *> cluster_problems,
           int num_hidden,
           int num_types,
           int num_balls,
           int width_limit)
: cluster_problems_(cluster_problems), width_limit_(width_limit),
    num_hidden_(num_hidden), num_types_(num_types), num_balls_(num_balls) {
    for (uint index = 0; index < cluster_problems.size(); ++index) {
      max_states_ = max(max_states_, cluster_problems_[index]->num_states);
    }
  }
  
  // Creates an reparameterization array for the cluster problem.
  vector<vector<vector<double > > > *CreateReparameterization() const {
    vector<vector<vector<double> > > *reparameterization = 
      new vector<vector<vector<double> > >(problems_size());
    for (int problem_id = 0; problem_id < problems_size(); ++problem_id) {
      cerr << problem_id << " " << problem(problem_id).num_states << endl;
      (*reparameterization)[problem_id].resize(problem(problem_id).num_states);
      for (int state = 0; state < problem(problem_id).num_states; ++state) {
        (*reparameterization)[problem_id][state].resize(num_hidden(), 0.0);
      }
    }
    return reparameterization;
  } 

  int problems_size() const { return cluster_problems_.size(); } 
  const ClusterProblem &problem(int index) const {
    return *cluster_problems_[index];
  }

  int max_states() const { return max_states_; }

  int num_hidden() const { return num_hidden_; }

  int num_types() const { return num_types_; }

  void set_num_balls(int num_balls) { num_balls_ = num_balls; } 

  int num_balls() const { return num_balls_; }
private:
  vector<ClusterProblem *> cluster_problems_; 

  int width_limit_;
  int max_states_;
  int num_hidden_;
  int num_types_;

  // Number of epsilon ball constraints to use.
  int num_balls_;
  
};

#endif
