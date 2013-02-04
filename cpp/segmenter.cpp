#include "segmenter.h"
#include "semimarkov.h"


Utterance *Segmenter::Run(const ClusterProblem &problem,
                          const Utterance &utterance) const {
  int num_states = problem.num_states;
  int num_timesteps = problem.num_steps;
  int width_limit = problem.width_limit();
  SemiMarkov semi_markov(num_states, num_timesteps, width_limit);
  semi_markov.Initialize();
  for (int s = 0; s < num_timesteps; ++s) {
    for (int e = s + 1; e < s + width_limit; ++e) {
      if (e >= num_timesteps) continue;
      double score = utterance.BestSequenceScore(s, e);
      for (int i = 0; i < num_states; ++i) {
        semi_markov.set_score(s, e, i, score);
      }
    }
  }
  semi_markov.ViterbiForward();
  semi_markov.ViterbiBackward();
  vector<int> path;
  vector<pair<double, int> > best_splits;
  vector<int > ordered_splits;
  double final_score = semi_markov.GetBestPath(&path);
  for (int t = 0; t < num_timesteps; ++t) {
    double m = INF;
    for (int i = 0; i < num_states; ++i) {
      
      double score = semi_markov.GetForwardScore(i, t) 
        + semi_markov.GetBackwardScore(i, t);
      if (score < m) {
        m = score;
      }
    }
    best_splits.push_back(pair<double, int>(m, t));

  }
  cerr << "sorting " << num_states << " " << num_timesteps << endl;
  sort(best_splits.begin(), best_splits.end());
  for (int i = 0; i < num_states * 3 + 1; ++i) {
    ordered_splits.push_back(best_splits[i].second);
  }
  sort(ordered_splits.begin(), ordered_splits.end());
  for (int i = 0; i < num_states * 3 + 1; ++i) {
    cerr << "split" << " " << ordered_splits[i] << endl;
  }
  cerr << final_score << endl;
  ordered_splits.push_back(num_timesteps);
  // const vector<int> &correct = utterance.get_gold();
  // for (uint i = 0; i < correct.size(); ++i) {
  //   cerr << "correct"  << " " << correct[i] << endl;
  // }
  Utterance *new_utterance = new Utterance(utterance);
  vector<vector<const DataPoint *> > sequence;
  MakeSequence(utterance, ordered_splits, &sequence);
  new_utterance->set_sequence(sequence);
  return new_utterance;
}

void Segmenter::MakeSequence(const Utterance &utterance, 
                  vector<int> order,  
                  vector<vector<const DataPoint *> > *seg) const {
  seg->resize(order.size() - 1);
  for (uint i = 0; i < order.size() - 1; ++i) {
    for (int j = order[i]; j < order[i + 1]; ++j) {
      (*seg)[i].push_back(&utterance.sequence(j, 0));
      cerr << j << endl;
    }
  }
}
