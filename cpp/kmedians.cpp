#include "gurobi_c++.h"
#include <iostream>
#include <vector>
#include <sstream>
#include "kmedians.h"
using namespace std;


void kmedians::ConstructLP(GRBModel &model) {
  r.resize(num_centers_);
  t.resize(num_points_);
  for (int type = 0; type < num_points_; ++type) {
    t[type].resize(num_centers_);   
  }

  try{
    for (int center = 0; center < num_centers_; ++center)  {
      stringstream buf;
      buf << "r_" << center;
      r[center] = 
        model.addVar(0.0, 1.0, 0, GRB_BINARY, buf.str());
    }
    
    for (int type = 0; type < num_points_; ++ type) {
      for (int center = 0; center < num_centers_; ++center)  {
        stringstream buf;
        buf << "t_" << type << "_" << center;
        t[type][center] = 
          model.addVar(0.0, 1.0, scores[type][center], GRB_BINARY, buf.str());
      }
    }
    
    model.update();
    GRBLinExpr l;
    for (int center = 0; center < num_centers_; ++center) {
      l += r[center];
    }
    model.addConstr(l == K_);
    
    for (int type = 0; type < num_points_; ++type) {
      GRBLinExpr l;
      for (int center = 0; center < num_centers_; ++center) {
        l += t[type][center];
      }
      model.addConstr(l == 1);
    }
    
    for (int type = 0; type < num_points_; ++ type) {
      for (int center = 0; center < num_centers_; ++center) {
        model.addConstr(t[type][center] <= r[center]);
      }
    }
    model.set(GRB_IntAttr_ModelSense, 1);
    model.update();
  } catch(GRBException e) {
    cout << "Error code = " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  } catch(...) {
    cout << "Exception during optimization" << endl;
  }

}

void kmedians::Marginalize(GRBModel &model, int center) {
  try{
    if (has_margin_) {
      model.remove(margin_);
    }
    margin_ = model.addConstr(r[center] == 1);
    model.update();
    has_margin_ = true;
  } catch(GRBException e) {
    cout << "Error code = " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  } catch(...) {
    cout << "Exception during optimization" << endl;
  }

}

double kmedians::Solve(GRBModel &model, vector<int> *centers) {
    double score = 0.0;
    centers->clear();
    try { 
      model.optimize();
      score = model.get(GRB_DoubleAttr_ObjVal);
      for (int center = 0; center < num_centers_; ++center) {
        int x = (int)r[center].get(GRB_DoubleAttr_X);
        if (x == 1) {
          centers->push_back(center);
        }
      }
    } catch(GRBException e) {
      cout << "Error code = " << e.getErrorCode() << endl;
      cout << e.getMessage() << endl;
    } catch(...) {
      cout << "Exception during optimization" << endl;
    }
    return score;
  }
