#include "hmm_astar_solver.h"
#include <time.h>

Search<HMMState, Expander> *HMMAStarSolver::InitializeAStar(bool exact) {
  return NULL;
}

double HMMAStarSolver::Solve(SpeechAlignment *alignment, bool exact, 
                             const HiddenSolver &hidden_solver,
                             const Reparameterization &delta_hmm,
                             const Reparameterization &delta_hidden,
                             double upper_bound) {
  // Compute Heuristic
  cerr << cp_.num_states << " " << cp_.num_steps << endl;
  Viterbi viterbi(cp_.num_states, 
                  cp_.num_steps, 
                  cp_.num_hidden(0), 
                  1);
  viterbi.Initialize();

  vector<bool> first_seen(cp_.num_states, true);
  for (int i = 0; i < cp_.num_states; ++i) {
    for (int j = i - 1; j >= 0; --j) {
      if (cp_.MapState(i) == cp_.MapState(j)) {
        first_seen[i] = false;
        break;
      }
    }
  }
  double best_dual_cluster = 0;
  vector<double> best(cp_.num_types(), INF);
  for (int l = 0; l < cp_.num_types(); ++l) {
    for (int c = 0; c < cp_.num_hidden(l); ++c) {
      double cost = hidden_solver.DualCost(l, c); 
      if (cost < best[l]) {
        best[l] = cost; 
      }
    }
    best_dual_cluster += best[l];
  }

  double cur_dual_cluster = best_dual_cluster;
  vector<double> state_heuristic(cp_.num_states); 
  for (int i = 0; i < cp_.num_states; ++i) {
    if (first_seen[i]) {
      cur_dual_cluster -= best[cp_.MapState(i)];
    }
    state_heuristic[i] = cur_dual_cluster;
  }
  


  // Update semi_markov weights with hidden scores.
  for (int m = 0; m < cp_.num_steps; m++) {
    for (int c = 0; c < cp_.num_hidden(0); ++c) {
      double score = distances_.get_distance(m, c);
      viterbi.set_score(m, c, score);
    }
  }
  for (int i = 0; i < cp_.num_states; i++) {
    for (int c = 0; c < cp_.num_hidden(0); ++c) {
      int l = cp_.MapState(i);
      viterbi.set_lambda(i, c, (*reparameterization_)[i][c] +  
                         delta_hmm.get(problem_, l, c) + delta_hidden.get(problem_, l, c));
    }
  }
  viterbi.ForwardScores();
  double back = viterbi.BackwardScores();
  cerr << "Best Score " << viterbi.GetBestScore() << " " << back << " " << endl;

  Heuristic *heuristic = new Heuristic(cp_.num_steps, cp_.num_states, cp_.num_hidden(0));
  Scorer *scorer = new Scorer(cp_.num_steps, cp_.num_states, cp_.num_hidden(0));
  
  for (int i = 0; i < cp_.num_states; i++) {
    for (int c = 0; c < cp_.num_hidden(0); ++c) {
      double lambda = (*reparameterization_)[i][c];
      int l = cp_.MapState(i);
      if (first_seen[i]) {
        lambda += hidden_solver.DualCost(l, c);
      }
      lambda += delta_hmm.get(problem_, l, c) + delta_hidden.get(problem_, l, c);
      scorer->set_lambda(i, c, lambda);
    }
  }


  for (int m = 0; m < cp_.num_steps; m++) {
    for (int i = 0; i < cp_.num_states; i++) {
      for (int c = 0; c < cp_.num_hidden(0); ++c) {
        double score = distances_.get_distance(m, c);
        scorer->set_score(m, i, c, score);
        heuristic->set_score(m, i, c, 
                             viterbi.backward_from_state(m, i, c) - score + state_heuristic[i]);
      }
    }
  }
  
  //if (round_ % 3 == 1) {
  for (int t = 0; t < cp_.num_types(); ++t) {
    //if (conflicts_[t]) {
      //if (enforced_ >= 11) break;
      //if (!enforced_constraints_[t]) {
        enforced_constraints_[t] = true;
        enforced_++;
        ////break;
        //}
        //}
  }
  
  round_++;
  Expander *expander = new Expander(scorer, heuristic, &cp_, enforced_constraints_);
  Search<HMMState, Expander> *search;
  //if (!exact) {
    // TEST
    search = new BeamMemory(expander, 10);
    
    FastBeamSearch<ThinState> *fbs = 
      new FastBeamSearch<ThinState>(cp_.num_states, cp_.num_steps, cp_.num_hidden(0));
    fbs->Initialize();
    ThinState state;
    Merger merger(scorer, heuristic, &cp_);
  // } else {
  //   search = new AStarMemory(expander);
  // }



  //Search<HMMState, Expander> *astar = InitializeAStar(exact);

  // Run the semi-markov model.
  int rounds = 0;
  double score = fbs->Run(1000, merger, &state, upper_bound, exact);
  cerr << "SCORE: " << score << endl;
  state.dictionary->show();
  cerr << endl;

  cerr << "ASCORE: " << score << endl;
  cerr << "AStar rounds: "<<  rounds << endl;
  state.dictionary->show();
  cerr << endl;

  vector<int> *align = alignment->mutable_alignment();
  state.dictionary->DumpToAlignment(align);
  align->push_back(cp_.num_steps);

  // double score = 
  //   viterbi->GetBestPath(alignment->mutable_alignment(), &state_to_center_);  
  // state_to_center_.resize(cp_.num_states);
  // double check_score = 0.0;

  vector<int> *hidden = alignment->mutable_hidden_alignment();
  hidden->resize(cp_.num_states);
  for (int i = 0; i < cp_.num_states ; ++i) {
    int type  = cp_.MapState(i);
    (*hidden)[i] = state.dictionary->center(type);
  }
  
  conflicts_.clear();
  conflicts_.resize(cp_.num_types(), 0);
  state.dictionary->FillConflicts(&conflicts_);
  cerr << "Conflicts: "; 
  for (int t = 0; t < cp_.num_types(); ++t) {
    if (conflicts_[t]) cerr << t << " ";
  }
  cerr << endl;
  // assert(fabs(score - check_score) < 1e-4);
  // delete viterbi;
  delete search;
  return score;
}

