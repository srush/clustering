#include "distances.h"
#include "hidden_solver.h"
#define INF 1e20

HiddenSolver::HiddenSolver(const ClusterSet &cs) 
  : cs_(cs) {
  type_dirty_.resize(cs_.num_types());
  hidden_costs_.resize(cs_.num_types());
  for (int t = 0; t < cs_.num_types(); ++t) {
    hidden_costs_[t].resize(cs_.num_hidden(t));
  }
  best_score_.resize(cs_.num_types());
  best_hidden_.resize(cs_.num_types());
}

// Compute max-marginals.
double HiddenSolver::MaxMarginals(vector<vector<vector<double> > > *mu) {  
  clock_t start = clock();
  double score = Solve();

  // $\sum_{p\neq t(i)} \max_{q'} \sum_{i':t(i') = p} \lambda(i',q') + \sum_{i':t(i') = t(i)} \lambda(i, q)$
  for (int u = 0; u < cs_.problems_size(); ++u) {
    const ClusterProblem &problem = cs_.problem(u);
    for (int i = 0; i < problem.num_states; ++i) {  
      int marginal_type = problem.MapState(i);
      for (int hidden = 0; hidden < cs_.num_hidden(marginal_type); ++hidden) {
        // if (is_eliminated(type, hidden)) {
        //   continue;
        // }
        for (int t = 0; t < cs_.num_types(); ++t) {
          if (t == marginal_type) continue;
          (*mu)[u][i][hidden] += best_score_[t];
        }
        (*mu)[u][i][hidden] += hidden_costs_[marginal_type][hidden]; 
      }
    }
  }
  for (int u = 0; u < cs_.problems_size(); ++u) {
    const ClusterProblem &problem = cs_.problem(u);
    for (int i = 0; i < problem.num_states; ++i) {  
      int type = problem.MapState(i);
      for (int hidden = 0; hidden < cs_.num_hidden(type); ++hidden) {
        // if (is_eliminated(type, hidden)) {
        //   continue;
        // }

        assert((*mu)[u][i][hidden] - score > -1e-4);
      }
    }
  }
  clock_t end = clock();
  cerr << "Running clustering " << end - start << endl;
  return score;
}

double HiddenSolver::Solve() {
  double total_score = 0.0;
  for (int t = 0; t < cs_.num_types(); ++t) {
    best_score_[t] = INF;
    for (int q = 0; q < cs_.num_hidden(t); ++q) {
      // if (is_eliminated(type, hidden)) {
      //   continue;
      // }
      double score = hidden_costs_[t][q];
      if (score < best_score_[t]) {
        best_score_[t] = score;
        best_hidden_[t] = q;
      }
    }
    total_score += best_score_[t];
  }
  cerr << "Hidden score: " << total_score << endl;
  return total_score;
}


double HiddenSolver::Rescore(const SpeechSolution &solution) const {
  double dual_score = 0.0;
  for (int t = 0; t < cs_.num_types(); ++t) {
    int hidden = solution.TypeToHidden(t);
    //cerr << t << " " << hidden << " " << hidden_costs_[t][hidden] << endl;
    dual_score += hidden_costs_[t][hidden];
  }
  return dual_score;
}

double HiddenSolver::Check(int type, int hidden) {
  assert(type < cs_.num_types());
  return hidden_costs_[type][hidden];
}

// void HiddenSolver::ToSubgrad(const ClusterSet &cs, 
//                              const BallHolder &ball_holder,
//                              ClusterSubgrad *subgrad) const {
//   for (int u = 0; u < cs.problems_size(); ++u) {
//     const ClusterProblem &cp = cs.problem(u);
//     for (int i = 0; i < cp.num_states; ++i) {
//       int hidden = best_hidden_[cp.MapState(i)];
//       // Anything near the hidden state is turned on.
//       for (int b = 0; b < ball_holder.balls_size(); ++b) {
//         int partition = ball_holder.partition_for_center(b, hidden);
//         //for (int n = 0; n < ball_holder.nearby_size(b, partition); ++n) {
//         //int q = ball_holder.nearby(b, partition, n);
//         ClusterIndex cluster_index(b, u, i, partition);
//         subgrad->Update(cluster_index, -1);
//           //}
//       }
//     }
//   }
// }
