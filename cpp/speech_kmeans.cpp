#include "speech_kmeans.h"
#include "viterbi.h"

SpeechKMeans::SpeechKMeans(const SpeechProblemSet &problem_set): 
  problems_(problem_set), cluster_problems_(problem_set.MakeClusterSet()) {
  num_types_ = cluster_problems_.num_types();
  num_features_ = problem_set.num_features();
}

double SpeechKMeans::Run(int rounds) {
  vector<vector<DataPoint> > phoneme_states(num_types_);
  double round_score = 0.0;
  for (int round = 0; round < rounds; ++round) {
    round_score = 0.0;
    for (int utterance_index = 0; 
         utterance_index < problems_.utterance_size(); 
         ++utterance_index) {
      round_score += Expectation(utterance_index, &phoneme_states);
    }
    Maximization(phoneme_states);
    cerr << "Round score: " << round << " " <<  round_score << endl;
  }
  return round_score;
}

double SpeechKMeans::Expectation(int utterance_index,
                                 vector<vector<DataPoint> > *sets) {
  const ClusterProblem &cluster_problem = 
    cluster_problems_.problem(utterance_index);
  const Utterance &utterance = problems_.utterance(utterance_index);

  Viterbi viterbi(cluster_problem.num_states, 
                  cluster_problem.num_steps);
  viterbi.Initialize();

   // Set the weights based on the current centers.
  for (int i = 0; i < cluster_problem.num_states; ++i) {
    for (int j = 0; j < cluster_problem.num_steps; ++j) {
        double score = dist(centers_[cluster_problem.MapState(i)], 
                            utterance.sequence(j));
        viterbi.set_score(j, i, score);
    }
  }
  
  // Run viterbi algorithm.
  vector<int> path;
  double score = viterbi.Run(&path);
  int correctness = utterance.ScoreAlignment(path);
  cerr << "Correctness: " << correctness << endl; 

  // Make the cluster sets.
  problems_.AlignmentClusterSet(utterance_index, path, sets);

  assert(path.size() - 1 == (uint)cluster_problem.num_states);
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
