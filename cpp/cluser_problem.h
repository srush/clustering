struct ClusterProblem{
  ClusterProblem(int _num_steps, 
                 int _width_limit, 
                 int _num_states, 
                 int _num_types, 
                 int _num_hidden,
                 const vector<int> &state_to_type) 
  : num_steps(_num_steps), width_limit(_width_limit), 
    num_states(_num_states), num_types(_num_types), num_hidden(_num_hidden),
    state_to_type_(state_to_type)
  {}
  
  int MapState(int state) const {
    return state_to_type_[state];
  }

  int num_steps;
  int width_limit;
  int num_states;
  int num_types;
  int num_hidden;  
private:
  const vector<int> &state_to_type_;
};
