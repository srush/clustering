#include "speech_kmeans.h"
#include "semimarkov.h"

SpeechKMeans::SpeechKMeans(const SpeechProblemSet &problem_set): 
  problems_(problem_set), cluster_problems_(problem_set.MakeClusterSet()),
  path_(cluster_problems_.problems_size()) {
  num_types_ = cluster_problems_.num_types();
  num_features_ = problem_set.num_features();
  distances_.resize(cluster_problems_.problems_size());
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    distances_[u] = problems_.MakeDistances(u);
  }
}

double SpeechKMeans::Run(int rounds) {
  vector<vector<DataPoint> > phoneme_states(num_types_);
  double round_score = 0.0;
  for (int round = 0; round < rounds; ++round) {
    round_score = 0.0;
    int total_correctness = 0;
    for (int utterance_index = 0; 
         utterance_index < problems_.utterance_size(); 
         ++utterance_index) {
      int correctness; 
      round_score += Expectation(utterance_index, &correctness, &phoneme_states);
      total_correctness += correctness;
    }
    Maximization(phoneme_states);
    cerr << "Round score: " << round << " " <<  round_score << " " << total_correctness << endl;
  }
  return round_score;
}

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

void SpeechKMeans::Maximization(const vector<vector<DataPoint> > &sets) {
  centers_.clear();
  if (!use_medians_) {
    for (int p = 0; p < num_types_; ++p) {
      DataPoint total(num_features_);
      for (uint i = 0; i < sets[p].size(); ++i) {
        total += sets[p][i];
      }
      total = total / float(sets[p].size());
      centers_.push_back(total);
    }
  } else {
    for (int p = 0; p < num_types_; ++p) {
      DataPoint total(num_features_);
      for (uint i = 0; i < sets[p].size(); ++i) {
        total += sets[p][i];
      }
      total = total / float(sets[p].size());
      double best = 1e10;
      DataPoint median(num_features_);
      for (uint i = 0; i < sets[p].size(); ++i) {
        double trial = dist(sets[p][i], total);
        if (trial < best) {
          best = trial;
          median = sets[p][i];
        }
      }
      centers_.push_back(median);
   }
  }
}

void SpeechKMeans::InitializeCenters() {
  srand(12);
  centers_.clear();
  for (int p = 0; p < num_types_; ++p) {
    DataPoint point(num_features_);
    for (int feat = 0; feat < num_features_; ++feat) {
      point[feat] = rand() / (float) RAND_MAX;
    }
    centers_.push_back(point);
  }
}

void SpeechKMeans::SetCenters(const vector<DataPoint> &centers) {
  centers_ = centers;
}

SpeechSolution *SpeechKMeans::MakeSolution() {
  const ClusterSet &cluster_set = problems_.MakeClusterSet();
  SpeechSolution *solution = new SpeechSolution(cluster_set);
  for (int p = 0; p < cluster_problems_.num_types(); ++p) {
    solution->set_type_to_special(p, centers_[p]);
  }

  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    const ClusterProblem &problem = cluster_problems_.problem(u);
    SpeechAlignment *align = solution->mutable_alignment(u);
    vector<int> *solution = align->mutable_alignment();
    solution->resize(problem.num_states + 1);
    for (int i = 0; i < problem.num_states + 1; ++i) {
      (*solution)[i] = path_[u][i];
    }
  }
  return solution;
}
