#include "gurobi_c++.h"
#include <iostream>
#include <vector>
#include <sstream>
#include "gaussian.h"
using namespace std;

int main(int argc, char **argv) {
  try{
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);
    
    int K = 5;
    int num_centers = 200;
    int num_types = 200; 
    kmedians kmed(num_centers, num_types, K);
  
    //double total = 0.0;
    for (int type = 0; type < num_types; ++type) {
      for (int center = 0; center < num_centers; ++center) {
        kmed.set_score(type, center, rand() / (float)RAND_MAX);;
        //scores[type][center] = 
        //r[type][center] = rand();
      }
      int real_center = (int) (5 * (rand() / (float)RAND_MAX));
      kmed.set_score(type, real_center, 0.05);
      //scores[type][real_center] = 0.05;
      //cerr << type << " " << best_center<<  endl;
      //total += best;
    }

    // double best = 1e20;
    // int best_center = -1;
    // for (int center = 0; center < num_centers; ++center) {
    //   double score = 0.0;
    //   for (int type = 0; type < num_types; ++type) {
    //     score += scores[type][center];
    //   }
    //   if (score < best) {
    //     best = score;
    //     best_center = center;
    //   }
    // }
    // cerr << best << " " <<  best_center << endl;

    // Exhaustive search  
    vector<int> on(K);
    for (int k = 0; k < K; ++k) {
      on[k] = 0;
    }

    kmed.ConstructLP(model);
    for (int center = 0; center < num_centers; ++center) {
      kmed.Marginalize(model, center);
      model.optimize();
    }

    /*for (int type = 0; type < num_types; ++ type) {
      for (int center = 0; center < num_centers; ++center) {
        int x = (int)kmed.point(type, center).get(GRB_DoubleAttr_X);
        if (x == 1) {
          cerr << type << " " << center << " " << kmed.point(type, center).get(GRB_DoubleAttr_Obj) << endl;
        }
      }
      }*/

  } catch(GRBException e) {
    cout << "Error code = " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  } catch(...) {
    cout << "Exception during optimization" << endl;
  }
}
