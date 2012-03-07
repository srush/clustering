#ifndef KMEDIANS_H
#define KMEDIANS_H

#include "gurobi_c++.h"

class kmedians {
 public:
   kmedians(int num_centers, int num_points, int K): 
  num_centers_(num_centers), 
    num_points_(num_points),
    K_(K)
  {
    has_margin_ = false;
    scores.resize(num_points);
    for (int points = 0; points < num_points; ++points) {
      scores[points].resize(num_centers_);
    }
  } 

  GRBVar point(int point, int center) {
    return t[point][center];
  } 

  void ConstructLP(GRBModel &model);

  double Solve(GRBModel &model, vector<int> *centers);

  void Marginalize(GRBModel &model, int center); 

  void set_score(int point, int center, double score) {
    scores[point][center] = score;
  }


 private:
  int num_centers_;
  int num_points_;
  int K_;
  vector<GRBVar> r;
  vector<vector<GRBVar> > t;
  vector<vector<double> > scores;
  GRBConstr margin_;

  bool has_margin_; 
};

#endif
