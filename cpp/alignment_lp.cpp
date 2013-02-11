#include "alignment_lp.h"
#include "gurobi_c++.h"

#define VAR_TYPE GRB_CONTINUOUS

void AlignmentLP::ConstructLP(SpeechSolution *proposal) {
  try {
  GRBEnv env = GRBEnv();
  GRBModel model = GRBModel(env);
  //model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);

  // Create all the variables.
  s.resize(cs_.problems_size());
  s2.resize(cs_.problems_size());
  //position_var.resize(cs_.problems_size()); 
  // Create the hmms. 
  for (uint u = 0; u < s.size(); ++u) {
    const ClusterProblem &problem = cs_.problem(u);
    int M = problem.num_steps;
    s[u].resize(M);
    s2[u].resize(M);
    //position_var.resize(M); 
    for (int m = 0; m < M; ++m) {
      s[u][m].resize(problem.num_states);
      s2[u][m].resize(problem.num_states);

      for (uint n = 0; n < s[u][m].size(); ++n) {
        s[u][m][n].resize(cs_.num_hidden(0));
        {
          stringstream buf;
          buf << "s2_" << u << "_" << m << "_" << n;
          s2[u][m][n] = 
            model.addVar(0.0, 1.0, 0.0, 
                         VAR_TYPE, buf.str());
        }

        for (uint o = 0; o < s[u][m][n].size(); ++o) {
          s[u][m][n][o].resize(3);
          for (uint f = 0; f <= 2; ++f) {
            stringstream buf;
            buf << "s_" << u << "_" << m << "_" << n << "_" << o << "_" << f;
            double score = distances_[u]->get_distance(m, o);
            s[u][m][n][o][f] = 
              model.addVar(0.0, 1.0, score, 
                           VAR_TYPE, buf.str());
          }
        }
      }
      // position_var[u][m].resize(cs_.num_types());
      // for (uint l = 0; l < position_var[u][m].size(); ++l) {
      //   position_var[u][m][l].resize(cs_.num_hidden(0));
      //   for (uint o = 0; o < position_var[u][m][l].size(); ++o) {
      //     stringstream buf;
      //     buf << "position_" << u << "_" << m << "_" <<  l << "_" << o;
      //     position_var[u][m][l][o] =
      //       model.addVar(0.0, 1.0, 0.0, 
      //                    VAR_TYPE, buf.str());
      //   }
      // }
    }
  }
  r.resize(cs_.num_types());
  for (uint l = 0; l < r.size(); ++l) {
    r[l].resize(cs_.num_hidden(0));
    for (uint o = 0; o < r[l].size(); ++o) {
      stringstream buf;
      buf << "r_" << l << "_" << o;
      r[l][o] = 
        model.addVar(0.0, 1.0, 0.0, 
                     VAR_TYPE, buf.str());
    }
  }

  t.resize(cs_.problems_size());
  for (uint u = 0; u < t.size(); ++u) {
    const ClusterProblem &problem = cs_.problem(u);
    t[u].resize(problem.num_states);
    for (uint n = 0; n < t[u].size(); ++n) {
      t[u][n].resize(cs_.num_hidden(0));
      for (uint o = 0; o < t[u][n].size(); ++o) {
        stringstream buf;
        buf << "t_" << u << "_" << n << "_" << o;
        t[u][n][o] = 
          model.addVar(0.0, 1.0, 0.0, 
                       VAR_TYPE, buf.str());
      }
    }
  }
  model.update();
  // Make constraints.
  // Constraint for M

  // for all l, sum_q r(l,o) = 1
  for (int l = 0; l < cs_.num_types(); ++l) {
    // Sum out o's.
    GRBLinExpr sum;
    for (int o = 0; o < cs_.num_hidden(0); ++o) {
      sum += r[l][o];
    }
    // Equal to 1.
    model.addConstr(sum == 1);
  }
  cerr << "Done Constraint 1" << endl;

  // Constraint for N
  // for all m,n,o, i, s(u, m, n, o) = s(m-1, n, o, 0) +  
  // s(m-1, n, o, 1)
  
  for (int u = 0; u < cs_.problems_size(); ++u) {
    const ClusterProblem &problem = cs_.problem(u);
    int M = problem.num_steps;
    int N = problem.num_states;

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        for (int o = 0; o < cs_.num_hidden(0); ++o) {

          if (m != 0 && n != 0) {            
            GRBLinExpr sum;
            model.addConstr(s[u][m][n][o][0] + 
                            s[u][m][n][o][2] == 
                            s[u][m - 1][n][o][0] + 
                            s[u][m - 1][n][o][1], "incoming");
          }

          if (m != M - 1 && n != N - 1) {
            model.addConstr(s[u][m][n][o][0] + 
                            s[u][m][n][o][1] == 
                            s[u][m + 1][n][o][0] + 
                            s[u][m + 1][n][o][2], "Outgoing");            
          }
          GRBLinExpr sum, sum2;
          for (int o2 = 0; o2 < cs_.num_hidden(0); ++o2) {
            sum += s[u][m][n][o2][1];
            sum2 += s[u][m][n][o2][2];
          }
          model.addConstr(s2[u][m][n] == sum, "Outgoing");
          if (m != M - 1 && n != N - 1) {
            model.addConstr(sum2 == 
                            s2[u][m + 1][n + 1], "Outgoing");
          }
        }
      }
    }
    {
      GRBLinExpr sum;
      for (int o = 0; o < cs_.num_hidden(0); ++o) {
        sum += s[u][0][0][o][1];
      }
      model.addConstr(sum == 1, "Starting");
    }
    
    {
      GRBLinExpr sum;
      for (int o = 0; o < cs_.num_hidden(0); ++o) {
        sum += s[u][problem.num_steps - 1][problem.num_states - 1][o][0];
      }
      model.addConstr(sum == 1);
    }
  }

  // Tying constraints 3.
  // forall n, o, r(y_n, o) = t(n,o)
  for (int u = 0; u < cs_.problems_size(); ++u) {
    const ClusterProblem &problem = cs_.problem(u);
    for (int n = 0; n < problem.num_states; ++n) {
      GRBLinExpr sum;
      for (int o = 0; o < cs_.num_hidden(0); ++o) {
        sum += t[u][n][o];
      }
      model.addConstr(sum == 1);
    }
    for (int n = 0; n < problem.num_states; ++n) {
      int l = problem.MapState(n);
      for (int o = 0; o < cs_.num_hidden(0); ++o) {
        model.addConstr(r[l][o] == t[u][n][o]);

        GRBLinExpr sum;
        for (int m = 0; m < problem.num_steps; ++m) {
          sum += s[u][m][n][o][1];
        }
        model.addConstr(sum == t[u][n][o]);
      }
    }
  }
  cerr << "Done Constraint 3" << endl;
  
  if (true)  {
    for (int u = 0; u < cs_.problems_size(); ++u) {
      const ClusterProblem &problem = cs_.problem(u);
      int M = problem.num_steps;
      int N = problem.num_states;
      for (int m = 0; m < M; ++m) {
        for (uint l = 0; l < r.size(); ++l) {
          vector <int> locations;
          for (int n = 0; n < N; ++n) {
            if ((int)l == problem.MapState(n)) {
              locations.push_back(n);
            }
          }

          for (int o = 0; o < cs_.num_hidden(0); ++o) {
            GRBLinExpr sum;
            for (uint occ_index = 0; occ_index < locations.size(); ++occ_index) {
              int n = locations[occ_index];
              sum += s[u][m][n][o][0] + s[u][m][n][o][1] + s[u][m][n][o][2];
            }
            //model.addConstr(position_var[u][m][l][o] == sum);
            model.addConstr(sum <= r[l][o]);
          }
        }
      }
    }
  }

  // model.addConstr(r[7][2] == 0.5);
  // model.addConstr(r[7][3] == 0.5);
  // model.addConstr(r[0][1] == 0.5);
  // model.addConstr(r[0][6] == 0.5);
  // model.addConstr(r[13][4] == 0.5);
  // model.addConstr(r[13][9] == 0.5);
  model.update();
  //model.write("temp.lp");

  // 
  // // Done!
   model.optimize();
   if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL) {
     model.computeIIS();
     model.write("temp.ilp");
   }

  vector<double> costs;
  for (uint u = 0; u < s.size(); ++u) {
    SpeechAlignment *align = proposal->mutable_alignment(u);
    const ClusterProblem &problem = cs_.problem(u);
    vector<int> *state_hidden = align->mutable_hidden_alignment();
    vector<int> *state_align = align->mutable_alignment();
    state_hidden->resize(problem.num_steps);
    state_align->resize(problem.num_steps);
    
    int M = problem.num_steps;
    int N = 0;
    for (int m = 0; m < M; ++m) {
      N = s[u][m].size();
      costs.resize(s[u][m].size());
      for (uint n = 0; n < s[u][m].size(); ++n) {
        for (uint o = 0; o < s[u][m][n].size(); ++o) {
          for (uint f = 0; f <= 2; ++f) {
            if (s[u][m][n][o][f].get(GRB_DoubleAttr_X) != 0) {
              (*state_hidden)[m] = o;
              (*state_align)[m] = problem.MapState(n);
              string position;
              if (f == 0) {
                position = "I";
              } else if (f == 1) {
                position = "B";
              } else { 
                position = "O";
              }
              cerr << "s " << m << " " << n << " " << o << " " << position << " "
                   << s[u][m][n][o][f].get(GRB_DoubleAttr_X) 
                   << " " << s[u][m][n][o][f].get(GRB_DoubleAttr_Obj) 
                   << " " << problem.MapState(n) << endl;
              costs[n] += s[u][m][n][o][f].get(GRB_DoubleAttr_X) 
                * s[u][m][n][o][f].get(GRB_DoubleAttr_Obj);
            }
          }
        }
      }
    }
    for (int n = 0; n < N; ++n) {
      cerr << n << " " << costs[n] << endl;
    }
  }


  for (uint u = 0; u < t.size(); ++u) {
    for (uint n = 0; n < t[u].size(); ++n) {
      for (uint o = 0; o < t[u][n].size(); ++o) {
        if (t[u][n][o].get(GRB_DoubleAttr_X) != 0) {
          cerr << "t " <<  n << " " << o << " " <<
            t[u][n][o].get(GRB_DoubleAttr_X) << endl;
        }
      }
    }
  }

  for (uint l = 0; l < r.size(); ++l) {
    for (uint o = 0; o < r[l].size(); ++o) {
      if (r[l][o].get(GRB_DoubleAttr_X) != 0) {
        cerr << "r " << l << " " << o << " " <<
          r[l][o].get(GRB_DoubleAttr_X) << endl;
      }
    }
  }


  } catch(GRBException e) {
    cout << "Error code = " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  } catch(...) {
    cout << "Exception during optimization" << endl;
  }
}

