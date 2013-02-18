#include "hmm_beam_search_solver.h"
#include <time.h>


double HMMBeamSearchSolver::Solve(SpeechSolution *solution, 
                                  bool exact, 
                                  const HiddenSolver &hidden_solver,
                                  const Reparameterization &delta_hmm,
                                  const Reparameterization &delta_hidden,
                                  double upper_bound) {
  double total_score = 0.0;
  vector<bool> have_seen(cs_.num_types(), false);
  Beam<ThinState> final;
  for (int u = 0; u < cs_.problems_size(); ++u) {
    const ClusterProblem &cp = cs_.problem(u); 
    vector<bool> first_seen(cp.num_states, false);    
    vector<bool> new_seen(cs_.num_types(), false);

    for (int i = 0; i < cp.num_states; ++i) {
      int type = cp.MapState(i);
      if (!have_seen[type]) {
        have_seen[type] = true;
        new_seen[type] = true;
        first_seen[i] = true;
      }
    }

    double best_dual_cluster = 0;
    vector<double> best(cp.num_types(), INF);
    for (int l = 0; l < cp.num_types(); ++l) {
      if (!new_seen[l]) continue;
      for (int c = 0; c < cp.num_hidden(l); ++c) {
        double cost = hidden_solver.DualCost(l, c); 
        if (cost < best[l]) {
          best[l] = cost; 
        }
      }
      best_dual_cluster += best[l];
    }

    double cur_dual_cluster = best_dual_cluster;
    vector<double> state_heuristic(cp.num_states, 0.0); 
    for (int i = 0; i < cp.num_states; ++i) {
      if (first_seen[i]) {
        cur_dual_cluster -= best[cp.MapState(i)];
      }
      state_heuristic[i] = cur_dual_cluster;
    }

    // Compute Heuristic
    cerr << cp.num_states << " " << cp.num_steps << endl;
    Viterbi viterbi(cp.num_states, 
                    cp.num_steps, 
                    cp.num_hidden(0), 
                    1);
    viterbi.Initialize();

    // Update semi_markov weights with hidden scores.
    for (int m = 0; m < cp.num_steps; m++) {
      for (int c = 0; c < cp.num_hidden(0); ++c) {
        double score = distances_[u]->get_distance(m, c);
        viterbi.set_score(m, c, score);
      }
    }
    for (int i = 0; i < cp.num_states; i++) {
      for (int c = 0; c < cp.num_hidden(0); ++c) {
        //int l = cp.MapState(i);
        viterbi.set_lambda(i, c, reparameterization_->get(u, i, c) +  
                           delta_hmm.get(u, i, c) + 
                           delta_hidden.get(u, i, c));
      }
    }
    viterbi.ForwardScores();
    double back = viterbi.BackwardScores();
    cerr << "Best Score " << 
      viterbi.GetBestScore() << " " << 
      back << " " << endl;

    Heuristic *heuristic = new Heuristic(cp.num_steps, 
                                         cp.num_states, 
                                         cp.num_hidden(0));
    Scorer *scorer = new Scorer(cp.num_steps, 
                                cp.num_states, 
                                cp.num_hidden(0));
  
    for (int i = 0; i < cp.num_states; i++) {
      for (int c = 0; c < cp.num_hidden(0); ++c) {
        double lambda = reparameterization_->get(u, i, c);
        int l = cp.MapState(i);
        if (first_seen[i]) {
          lambda += hidden_solver.DualCost(l, c);
        }
        lambda += delta_hmm.get(u, i, c) + delta_hidden.get(u, i, c);
        scorer->set_lambda(i, c, lambda);
      }
    }
    

    for (int m = 0; m < cp.num_steps; m++) {
      for (int i = 0; i < cp.num_states; i++) {
        for (int c = 0; c < cp.num_hidden(0); ++c) {
          double score = distances_[u]->get_distance(m, c);
          scorer->set_score(m, i, c, score);
          double heuristic_score = 
            viterbi.backward_from_state(m, i, c) - score + state_heuristic[i];
          heuristic->set_score(m, i, c, 
                               heuristic_score);
        }
      }
    }
  
  
    round_++;
    const DictTree *dictionary = NULL;
    double score = 0.0;
    int rounds = 0;
    bool fail = false;
    if (true || !exact) {
      // TEST
      //search = new BeamMemory(expander, 10);
      
      FastBeamSearch<ThinState> *fbs = 
        new FastBeamSearch<ThinState>(cp.num_states, 
                                      cp.num_steps, 
                                      cp.num_hidden(0));
      fbs->Initialize();
      ThinState state;
      Merger merger(scorer, heuristic, &cp);
      if (final.elements() > 0) {
        merger.set_start_beam(&final);
      }
      //Search<HMMState, Expander> *astar = InitializeAStar(exact);
      
      // Run the semi-markov model.
      
      score = fbs->Run(10000, merger, &state, upper_bound, exact, &fail);
      dictionary = state.dictionary;
      final = fbs->final(); 
    }
    if (fail) {
      return INF;
    }
    total_score = score;
    cerr << "SCORE: " << score << endl;
    dictionary->show();
    cerr << endl;

    cerr << "ASCORE: " << score << endl;
    cerr << "AStar rounds: "<<  rounds << endl;
    dictionary->show();
    cerr << endl;
    
    cerr << "Alignment" << endl;
    SpeechAlignment *alignment = solution->mutable_alignment(u);
    vector<int> *align = alignment->mutable_alignment();
    dictionary->DumpToAlignment(align);
    align->push_back(cp.num_steps);
    cerr << "hidden" << endl;
    vector<int> *hidden = alignment->mutable_hidden_alignment();
    hidden->resize(cp.num_states);
    for (int i = 0; i < cp.num_states ; ++i) {
      int type  = cp.MapState(i);
      (*hidden)[i] = dictionary->center(type);
    }

  }
  cerr << "done" << endl;
  // assert(fabs(score - check_score) < 1e-4);
  // delete viterbi;
  //delete search;
  return total_score;
}

