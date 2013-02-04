#include "hmm_astar_solver.h"
#include <time.h>

AStarMemory *HMMAStarSolver::InitializeAStar() {
  // Compute Heuristic
  cerr << cp_.num_states << " " << cp_.num_steps << endl;
  Viterbi viterbi(cp_.num_states, 
                  cp_.num_steps, 
                  cp_.num_hidden(0), 
                  1);
  viterbi.Initialize();

  // Update semi_markov weights with hidden scores.
  for (int m = 0; m < cp_.num_steps; m++) {
    for (int c = 0; c < cp_.num_hidden(0); ++c) {
      double score = distances_.get_distance(m, c);
      viterbi.set_score(m, c, score);
    }
  }
  for (int i = 0; i < cp_.num_states; i++) {
    for (int c = 0; c < cp_.num_hidden(0); ++c) {
      viterbi.set_lambda(i, c, (*reparameterization_)[i][c]);
    }
  }
  viterbi.ForwardScores();
  double back = viterbi.BackwardScores();
  cerr << "Best Score " << viterbi.GetBestScore() << " " << back << " " << endl;

  Heuristic *heuristic = new Heuristic(cp_.num_steps, cp_.num_states, cp_.num_hidden(0));
  Scorer *scorer = new Scorer(cp_.num_steps, cp_.num_states, cp_.num_hidden(0));
  
  for (int i = 0; i < cp_.num_states; i++) {
    for (int c = 0; c < cp_.num_hidden(0); ++c) {
      scorer->set_lambda(i, c, (*reparameterization_)[i][c]);
    }
  }


  for (int m = 0; m < cp_.num_steps; m++) {
    for (int i = 0; i < cp_.num_states; i++) {
      for (int c = 0; c < cp_.num_hidden(0); ++c) {
        double score = distances_.get_distance(m, c);
        scorer->set_score(m, i, c, score);
        heuristic->set_score(m, i, c, viterbi.backward_from_state(m, i, c) - score);
      }
    }
  }
  
  if (round_ % 10 == 0) {
  for (int t = 0; t < cp_.num_types(); ++t) {
    if (conflicts_[t]) {
      if (enforced_ >= 11) break;
      if (!enforced_constraints_[t]) {
        enforced_constraints_[t] = true;
        enforced_++;
        break;
      }
    }
  }
  }
  round_++;
  Expander *expander = new Expander(scorer, heuristic, &cp_, enforced_constraints_);
  AStarMemory *astar = new AStarMemory(expander);
  return astar;
}


double HMMAStarSolver::Solve(SpeechAlignment *alignment) {
  AStarMemory *astar = InitializeAStar();

  // Run the semi-markov model.
  HMMState state;
  int rounds = 0;
  double score = astar->Run(&state, &rounds);
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
  delete astar;
  return score;
}

