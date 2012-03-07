#include "kmedians.h"
#include "gurobi_c++.h"
#include "hidden_kmedian_solver.h"

kmedians *KMediansSolver::InitializeKMedians() {
  kmedians *kmeds = new kmedians(cp_.num_hidden(0), 
                                 sp_.type_occurence_size(type_), 
                                 cp_.num_modes());
  for (int j = 0; j < sp_.type_occurence_size(type_); ++j ) {
    StateLocation loc = sp_.type_occurence(type_, j);
    for (int center = 0; center < cp_.num_hidden(0); ++center ) {
      kmeds->set_score(j, center, 
                       (*repar_)[loc.problem][loc.state][center]);
    }
  }
  return kmeds;

}

double KMediansSolver::Solve() {
  kmedians *kmed = InitializeKMedians();
  GRBEnv env = GRBEnv();
  env.set(GRB_IntParam_OutputFlag, 0);
  GRBModel model = GRBModel(env);
  kmed->ConstructLP(model);
  return kmed->Solve(model, &centers_);
}


double KMediansSolver::MaxMarginals(vector<double> *center_mu) {  
  kmedians *kmed = InitializeKMedians();
  GRBEnv env = GRBEnv();
  env.set(GRB_IntParam_OutputFlag, 0);
  GRBModel model = GRBModel(env);
  kmed->ConstructLP(model);
  center_mu->resize(cp_.num_hidden(0));
  double best = 1e20;
  for (int center = 0; center < cp_.num_hidden(0); ++center ) {
    kmed->Marginalize(model, center);
    (*center_mu)[center] = kmed->Solve(model, &centers_);
    best = min(best, (*center_mu)[center]);
  }
  int num_points = sp_.type_occurence_size(type_);
  cerr << "Done type " << type_ << " " << best << " " << num_points << endl; 
  return best;
}
