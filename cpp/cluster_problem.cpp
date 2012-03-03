#include "cluster_problem.h"

ClusterSet::ClusterSet(const vector<ClusterProblem *> cluster_problems,
                       int num_hidden,
                       int num_types,
                       int num_balls,
                       int width_limit,
                       int num_modes)
  : cluster_problems_(cluster_problems), width_limit_(width_limit),
    num_hidden_(num_hidden), num_types_(num_types), //num_balls_(num_balls),
    type_hidden_(num_types), num_modes_(num_modes){
  for (uint index = 0; index < cluster_problems.size(); ++index) {
      max_states_ = max(max_states_, cluster_problems_[index]->num_states);
  }
}


vector<vector<vector<double > > > *ClusterSet::CreateReparameterization() const {
  vector<vector<vector<double> > > *reparameterization = 
    new vector<vector<vector<double> > >(problems_size());
  for (int problem_id = 0; problem_id < problems_size(); ++problem_id) {
    const ClusterProblem &prob = problem(problem_id);
    cerr << problem_id << " " << prob.num_states << endl;
    (*reparameterization)[problem_id].resize(prob.num_states);
    for (int state = 0; state < prob.num_states; ++state) {
      int type = prob.MapState(state);
      (*reparameterization)[problem_id][state].resize(num_hidden(type), 0.0);
    }
  }
  return reparameterization;
} 


ClusterProblem::ClusterProblem(int _num_steps, 
                 int _num_states, 
                 const vector<int> &state_to_type)
  : num_steps(_num_steps), 
    num_states(_num_states), 
    state_to_type_(state_to_type), 
    has_gold_(false)
  {}

int ClusterProblem::width_limit() const {
  return cluster_set_->width_limit();
}

int ClusterProblem::num_hidden(int type) const {
  return cluster_set_->num_hidden(type);
}

int ClusterProblem::hidden_for_type(int type, int hidden_index) const {
  return cluster_set_->hidden_for_type(type, hidden_index);
}


