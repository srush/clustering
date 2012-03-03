#include "alignment_lp.h"
#include "gurobi_c++.h"
/*
void AlignmentLP::ConstructLP() {

  try{
  GRBEnv env = GRBEnv();
  GRBModel model = GRBModel(env);

  // Create all the variables.
  int width_limit = cs_.width_limit();
  x.resize(cs_.problems_size());
  y.resize(cs_.problems_size());
  int count = 0;
  for (int u = 0; u < cs_.problems_size(); ++u) {
    const ClusterProblem &problem = cs_.problem(u);
    int t = problem.num_steps;
    x[u].resize(problem.num_states);
    y[u].resize(problem.num_states);
    for (int i = 0; i < problem.num_states; ++i) {
      int p = problem.MapState(i);
      x[u][i].resize(cs_.num_hidden(p));
      for (int q = 0; q < cs_.num_hidden(p); ++q) {
        x[u][i][q].resize(t);
        for (int s = 0; s < t; ++s) {
          x[u][i][q][s].resize(width_limit);
          for (int o = 0; o < width_limit; ++o) {
            // make x variables. 
            //double score;
            stringstream buf;
            buf << "x_" << u << "_" << i << "_" << q << "_" << s << "_" << o;
            int global_q = problem.hidden_for_type(p, q);
            double score = distances_[u]->get_distance(s, o, global_q);
            if (distances_[u]->is_distance_used(s, o, q)) {
              x[u][i][q][s][o] = 
                model.addVar(0.0, 1.0, score, GRB_CONTINUOUS, buf.str());
              count++;
            }
            // score x variables
            //cerr << buf.str() << endl;
          }
        }
      }
      y[u][i].resize(cs_.num_hidden(p));
      for (int q = 0; q < cs_.num_hidden(p); ++q) {
        // make y variables.
        stringstream buf;
        buf << "y_" << u << "_" << i << "_" << q;
        y[u][i][q] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, buf.str()); 
      }
    }
  }
  cerr << "count "<< count  << endl;
  z.resize(cs_.num_types());
  for (int p = 0; p < cs_.num_types(); ++p) {
    z[p].resize(cs_.num_hidden(p));
    for (int q = 0; q < cs_.num_hidden(p); ++q) {
      // make z variables.
      stringstream buf;
      buf << "z_" << p << "_" << q;
      z[p][q] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, buf.str());
    }
  }

  model.update();
  // Make constraints.
  // Constraint 1
  // for all p, sum_q z(p,q) = 1
  // For all p's.
  for (int p = 0; p < cs_.num_types(); ++p) {
    // Sum out q's.
    GRBLinExpr l;
    for (int q = 0; q < cs_.num_hidden(p); ++q) {
      l += z[p][q];
    }
    // Equal to 1.
    model.addConstr(l == 1);
  }
  cerr << "Done Constraint 1" << endl;

  // Constraint 2
  // for all u, i, y(i,q) = 
  for (int u = 0; u < cs_.problems_size(); ++u) {
    const ClusterProblem &problem = cs_.problem(u);
    for (int i = 0; i < problem.num_states; ++i) {
      int p = problem.MapState(i);
      for (int q = 0; q < cs_.num_hidden(p); ++q) {
        // y = z
        model.addConstr(y[u][i][q] == z[p][q]);
      }
    }
    // Equal to 1.
  }
  cerr << "Done Constraint 2" << endl;

  // Constraint 3.
  for (int u = 0; u < cs_.problems_size(); ++u) {
    const ClusterProblem &problem = cs_.problem(u);
    int t = problem.num_steps;
    for (int i = 0; i < problem.num_states; ++i) {
      int p = problem.MapState(i);
      for (int q = 0; q < cs_.num_hidden(p); ++q) {
        // Sum out over s and e.
        GRBLinExpr l;
        for (int s = max(0, t - width_limit + 1); s < t; ++s) {
          for (int o = 0; o < width_limit; ++o) {
            if (distances_[u]->is_distance_used(s, o, q)) {
              l += x[u][i][q][s][o];
            }
          }
        }
        // constraint y = x
        model.addConstr(l == y[u][i][q]);
      }
    }
  }
  cerr << "Done Constraint 3" << endl;

  // Constraint 5.
  {
    GRBLinExpr l;
    int count = 0;
    for (int u = 0; u < cs_.problems_size(); ++u) {
      const ClusterProblem &problem = cs_.problem(u);
      int p = problem.MapState(0);
      for (int q = 0; q < cs_.num_hidden(p); ++q) {
        for (int o = 0; o < width_limit; ++o) {
          l += x[u][0][q][0][o];
        }
      }
    }
    // Constraint
    cerr << "add connstraint 5" << " " << count << endl;
    model.addConstr(l == 1);
  }
  cerr << "Done Constraint 5" << endl;


  // Constraint 4.
  {
    for (int u = 0; u < cs_.problems_size(); ++u) {
      const ClusterProblem &problem = cs_.problem(u);
      for (int i = 0; i < problem.num_states - 1; ++i) {
        int p1 = problem.MapState(i);
        int p2 = problem.MapState(i + 1);
        for (int q1 = 0; q1 < cs_.num_hidden(p1); ++q1) {
          for (int q2 = 0; q2 < cs_.num_hidden(p2); ++q2) {
            int t = problem.num_steps;
            for (int m = 0; m < t; ++m) {
              // sum before.
              GRBLinExpr l1;  
              for (int s = max(m - width_limit, 0); s < m; ++s) {
                int o = m - s;
                assert(o > 0);
                assert(o < width_limit);
                if (distances_[u]->is_distance_used(s, o, q1)) {
                  l1 += x[u][i][q1][s][o];
                }
              }
              // sum after.
              GRBLinExpr l2;
              for (int e = m + 1; e < m + width_limit; ++e) {
                int o = e - m;
                assert(o > 0);
                assert(o < width_limit);
                if (distances_[u]->is_distance_used(m, o, q2)) {
                  l2 += x[u][i + 1][q2][m][o];
                }
              }
              model.addConstr(l1 == l2);
            }
          }
        }
      }
    }
  }
  cerr << "Done Constraint 4" << endl;

  model.update();
  // Done!
  model.optimize();

  } catch(GRBException e) {
    cout << "Error code = " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  } catch(...) {
    cout << "Exception during optimization" << endl;
  }
}

*/
