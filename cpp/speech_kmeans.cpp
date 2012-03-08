#include "speech_kmeans.h"
#include "semimarkov.h"
#include "viterbi.h"

SpeechKMeans::SpeechKMeans(const SpeechProblemSet &problem_set): 
  problems_(problem_set), cluster_problems_(problem_set.MakeClusterSet()),
  path_(cluster_problems_.problems_size()),
  mode_centers_(cluster_problems_.problems_size()) {
  num_types_ = cluster_problems_.num_types();
  num_features_ = problem_set.num_features();
  distances_.resize(cluster_problems_.problems_size());
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    distances_[u] = problems_.MakeDistances(u);
  }
  use_gmm_ = false;
  use_medians_ = false;
}

double SpeechKMeans::Run(int rounds) {
  vector<vector<vector<DataPoint> > > phoneme_states(num_types_);
  int num_modes = cluster_problems_.num_modes();
  vector<vector<DataPoint> > center_estimators(num_modes);  
  vector<vector<double> > center_counts(num_modes);  

  double round_score = 0.0;
  for (int round = 0; round < rounds; ++round) {
    if (use_gmm_) {
      for (int mode = 0; mode < num_modes; ++mode) {
        center_estimators[mode].resize(num_types_);
        center_counts[mode].resize(num_types_);
        for (int type = 0; type < num_types_; ++type) {
          center_estimators[mode][type].resize(problems_.num_features(), 0.0);
          center_counts[mode][type] = 0.0;
        }
      }
    }

    round_score = 0.0;
    int total_correctness = 0;
    for (int utterance_index = 0; 
         utterance_index < problems_.utterance_size(); 
         ++utterance_index) {
      int correctness; 
      if (!use_gmm_) {
        round_score += Expectation(utterance_index, &correctness, &phoneme_states);
        total_correctness += correctness;
      } else {
        GMMExpectation(utterance_index, center_estimators, center_counts);
        round_score += Expectation(utterance_index, &correctness, &phoneme_states);
        total_correctness += correctness;
      }
    }
    if (!use_gmm_) {
      Maximization(phoneme_states);
    } else {
      GMMMaximization(center_estimators, center_counts);
    }
    cerr << "Round score: " << round << " " <<  round_score << " " << total_correctness << endl;
  }
  return round_score;
}
/*
double SpeechKMeans::Expectation(int utterance_index,
                                 int *correctness,
                                 vector<vector<DataPoint> > *sets) {
  int u = utterance_index;
  const ClusterProblem &cluster_problem = 
    cluster_problems_.problem(utterance_index);
  const Utterance &utterance = problems_.utterance(utterance_index);

  int width_limit = cluster_problems_.width_limit();
  SemiMarkov semimarkov(cluster_problem.num_states, 
                        cluster_problem.num_steps,
                        width_limit);
  semimarkov.Initialize();


   // Set the weights based on the current centers.
  clock_t start = clock();
  int count = 0;

  for (int s = 0; s < cluster_problem.num_steps; ++s) {
    for (int o = 0; o < width_limit; ++o) {
      if (s + o >= cluster_problem.num_steps) {
        continue;
      }
      if (distances_[u]->is_pruned(s, o)) {
        semimarkov.set_pruned(s, s + o);
      }
    }
  }


  for (int i = 0; i < cluster_problem.num_states; ++i) {
    const DataPoint &center = centers_[cluster_problem.MapState(i)];
    vector<double> dists;
    for (int s = 0; s < cluster_problem.num_steps; ++s) {
      dists.push_back(dist(center,
                           utterance.sequence(s)));
    }
    for (int s = 0; s < cluster_problem.num_steps; ++s) {
      double total = 0.0;
      for (int o = 0; o < width_limit; ++o) {
        total += dists[s + o];
        if (s + o >= cluster_problem.num_steps) {
          break;
        }
        if (!distances_[u]->is_pruned(s, o)) { 
          semimarkov.set_score(s, s + o, i, total);
        }
        ++count;
      }
    }
  }
  cerr << "score setting " << clock() - start << " " << count << endl;  
  
  // Run semimarkov algorithm.
  //vector<int> path;
  semimarkov.ViterbiForward();
  double score = semimarkov.GetBestPath(&path_[u]);
  (*correctness) = utterance.ScoreAlignment(path_[u]);
  cerr << "Correctness: " << *correctness << endl; 

  // Make the cluster sets.
  problems_.AlignmentClusterSet(utterance_index, path_[u], sets);

  assert(path_[u].size() - 1 == (uint)cluster_problem.num_states);
  cerr << endl;
  return score;
}
*/


double SpeechKMeans::Expectation(int utterance_index,
                                 int *correctness,
                                 vector<vector<vector<DataPoint> > > *sets) {
  int u = utterance_index;
  const ClusterProblem &cluster_problem = 
    cluster_problems_.problem(utterance_index);
  const Utterance &utterance = problems_.utterance(utterance_index);

  // Duplicate each state for every mode.
  Viterbi viterbi(cluster_problem.num_states,
                  cluster_problem.num_steps, cluster_problems_.num_modes(), 3);
  viterbi.Initialize();


   // Set the weights based on the current centers.
  clock_t start = clock();

  for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
    for (int i = 0; i < cluster_problem.num_states; ++i) {
      const DataPoint &center = centers_[mode][cluster_problem.MapState(i)];
      for (int s = 0; s < cluster_problem.num_steps; ++s) {
        double score = dist(center,
                            utterance.sequence(s));
        viterbi.set_state_score(s, i, mode, score);
      }
    }
  }
  cerr << "score setting " << clock() - start << endl;  
  
  // Run semimarkov algorithm.
  //vector<int> path;
  viterbi.ForwardScores();
  double score = viterbi.GetBestPath(&path_[u], &mode_centers_[u]);
  (*correctness) = utterance.ScoreAlignment(path_[u]);
  cerr << "Correctness: " << *correctness << endl; 

  // Make the cluster sets.
  sets->resize(cluster_problems_.num_modes());
  for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
    (*sets)[mode].resize(num_types_);
  }
  problems_.AlignmentClusterSet(utterance_index, path_[u], mode_centers_[u], sets);

  // collapse to modes. 
  assert(path_[u].size() - 1 == (uint)cluster_problem.num_states);
  cerr << endl;
  return score;
}


double SpeechKMeans::GMMExpectation(int utterance_index,
                                    vector<vector<DataPoint> > &estimators, 
                                    vector<vector<double> > &counts) {
  const ClusterProblem &cluster_problem = 
    cluster_problems_.problem(utterance_index);
  const Utterance &utterance = problems_.utterance(utterance_index);

  // Duplicate each state for every mode.
  Viterbi viterbi(cluster_problem.num_states,
                  cluster_problem.num_steps, cluster_problems_.num_modes(), 3);
  viterbi.Initialize();


   // Set the weights based on the current centers.
  clock_t start = clock();

  for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
    for (int i = 0; i < cluster_problem.num_states; ++i) {
      const DataPoint &center = centers_[mode][cluster_problem.MapState(i)];
      for (int s = 0; s < cluster_problem.num_steps; ++s) {
        double score = dist(center,
                            utterance.sequence(s));
        viterbi.set_state_score(s, i, mode, score);
      }
    }
  }
  cerr << "score setting " << clock() - start << endl;  
  
  // Run semimarkov algorithm.
  //vector<int> path;
  viterbi.set_use_sum();
  viterbi.ForwardScores();
  double score = viterbi.GetBestScore();
  viterbi.BackwardScores();
  vector<vector<vector<double> > > marginals;
  viterbi.Marginals(&marginals);
  for (int s = 0; s < cluster_problem.num_steps; ++s) {
    for (int i = 0; i < cluster_problem.num_states; ++i) {
      int type = cluster_problem.MapState(i);
      for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
        double p = marginals[s][i][mode];
        assert(p <= 1.0);
        assert(p >= 0.0);
        estimators[mode][type] += p * utterance.sequence(s);
        counts[mode][type] += p;
      }
    }
  }
  return score;
}


void SpeechKMeans::GMMMaximization(const vector<vector<DataPoint> > &estimators, 
                                   const vector<vector<double> > &counts) {  
  centers_.clear();
  int num_modes = cluster_problems_.num_modes();
  centers_.resize(num_modes);
  for (int mode = 0; mode < num_modes; ++mode) {
    for (int type = 0; type < cluster_problems_.num_types(); ++type) {
      centers_[mode].push_back(estimators[mode][type] / counts[mode][type]);
    }
  }
}

void SpeechKMeans::Maximization(const vector<vector<vector<DataPoint> > > &sets) {
  centers_.clear();
  int num_modes = cluster_problems_.num_modes();
  centers_.resize(num_modes);
  if (!use_medians_) {
    for (int mode = 0; mode < num_modes; ++mode) {
      for (int p = 0; p < num_types_; ++p) {
        DataPoint total(num_features_);
        for (uint i = 0; i < sets[mode][p].size(); ++i) {
          total += sets[mode][p][i];
        }
        total = total / float(sets[mode][p].size());
        centers_[mode].push_back(total);
      }
    }
  } else {
    for (int mode = 0; mode < num_modes; ++mode) {
      for (int p = 0; p < num_types_; ++p) {
        DataPoint total(num_features_);
        for (uint i = 0; i < sets[mode][p].size(); ++i) {
          total += sets[mode][p][i];
        }
        total = total / float(sets[mode][p].size());
        double best = 1e10;
        DataPoint median(num_features_);
        for (uint i = 0; i < sets[mode][p].size(); ++i) {
          double trial = dist(sets[mode][p][i], total);
          if (trial < best) {
            best = trial;
            median = sets[mode][p][i];
          }
        }
        centers_[mode].push_back(median);
      }
    }
  }
}

void SpeechKMeans::InitializeCenters() {
  srand(12);
  centers_.clear();
  centers_.resize(cluster_problems_.num_modes());
  for (int p = 0; p < num_types_; ++p) {
    for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
      DataPoint point(num_features_);
      for (int feat = 0; feat < num_features_; ++feat) {
        point[feat] = rand() / (float) RAND_MAX;
      }
      centers_[mode].push_back(point);
    }
  }
}

void SpeechKMeans::SetCenters(const vector<vector<DataPoint> > &centers) {
  centers_ = centers;
}

SpeechSolution *SpeechKMeans::MakeSolution() {
  const ClusterSet &cluster_set = problems_.MakeClusterSet();
  SpeechSolution *solution = new SpeechSolution(cluster_set);
  for (int p = 0; p < cluster_problems_.num_types(); ++p) {
    for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
      solution->set_type_to_special(p, mode, centers_[mode][p]);
    }
  }

  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    const ClusterProblem &problem = cluster_problems_.problem(u);
    SpeechAlignment *align = solution->mutable_alignment(u);
    vector<int> *solution = align->mutable_alignment();
    vector<int> *modes = align->mutable_mode_align();
    modes->resize(problem.num_states + 1);
    solution->resize(problem.num_states + 1);
    for (int i = 0; i < problem.num_states + 1; ++i) {
      (*solution)[i] = path_[u][i];
      if (i < problem.num_states) {
        (*modes)[i] = mode_centers_[u][i];
      }
    }
  }
  return solution;
}
