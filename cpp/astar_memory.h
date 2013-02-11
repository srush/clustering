
#ifndef ASTAR_MEMORY_H
#define ASTAR_MEMORY_H

#include <vector>
#include <algorithm>
#include <assert.h>
#include "astar.h"
#include "beam_search.h"
#include "cluster_problem.h"
using namespace std;
#define INF 1e20 

class DictTree {
 public:
 DictTree() : back_(NULL), hash_(1) {} 

 /* DictTree(int split, const DictTree *back)  */
 /*   : split_(split), phoneme(), back_(back) { */
 /*    hash_ = back->hash_; */
 /*  }  */


 DictTree(int split, 
          int phoneme, 
          int center, 
          int centers,
          bool update_hash,
          const DictTree *back) 
   : split_(split), phoneme_(phoneme), center_(center), back_(back) {
    if (update_hash) {
      hash_ = back->hash_ * centers + center;
    } else {
      hash_ = back->hash_;
    }
  } 

  const DictTree *add(int split, int phoneme, 
                      int center, int centers, bool update_hash) const {
    return new DictTree(split, phoneme, center, centers, update_hash, this);
  }

  const DictTree *remove(int phoneme, int center, int centers) const {
    return this;
  }

  int center(int phoneme) const {
    const DictTree *cur = this;
    while (cur->back_ != NULL) {
      if (cur->phoneme_ == phoneme) return cur->center_;
      cur = cur->back_;
    }
    return -1;
  }

  void show() const {
    const DictTree *cur = this;
    while (cur->back_ != NULL) {
      cerr << cur->phoneme_ << ":" << cur->center_ << ":" << cur->split_ << " " ;
      cur = cur->back_;
    }
  }

  int hash() const {
    return hash_;
  }


  void DumpToAlignment(vector<int> *alignment) const {
    alignment->clear();
    Dump(alignment);
    alignment->push_back(0);
    reverse(alignment->begin(), alignment->end());
  }

  void FillConflicts(vector<int> *conflicts) const {
    const DictTree *cur = this;
    while (cur->back_ != NULL) {
      if (cur->center_ != center(cur->phoneme_)) 
        (*conflicts)[cur->phoneme_] = true;
      cur = cur->back_;
    }
  }

 private:
  void Dump(vector<int> *alignment) const {
    if (back_ == NULL) return;
    
    alignment->push_back(split_);
    return back_->Dump(alignment);
  }

  int split_;

  int phoneme_;

  int center_;

  const DictTree *back_; 

  int hash_;
};


class Dictionary {
 public:
 Dictionary(int phonemes, int num_centers) : phonemes_map_(phonemes, -1), centers_(num_centers), 
    hash_(1) {}
 Dictionary(const vector<int> &phonemes) : phonemes_map_(phonemes) {}

  bool is_fixed(int phoneme) { return phonemes_map_[phoneme] != -1; }

  int center(int phoneme) { return phonemes_map_[phoneme]; }

  Dictionary *add(int phoneme, int center) {
    Dictionary *d = new Dictionary(phonemes_map_);
    d->phonemes_map_[phoneme] = center;
    d->hash_ = hash_ * centers_ + center;
    return d;
  }

  void show() {
    for (unsigned int i = 0; i < phonemes_map_.size(); ++i) {
      if (phonemes_map_[i] != -1) 
        cerr << i << ":" << phonemes_map_[i] << " ";
    }
  }

  int hash() {
    return hash_;
  }
 private:
  vector<int> phonemes_map_;
  int centers_;
  int hash_;
};


class HMMState {
 public:
  HMMState() {}
 HMMState(const DictTree *_dict,
          int _time_step,
          int _state,
          int _center) 
    : dictionary(_dict),
    time_step(_time_step),
    state(_state),
    center(_center) {}

  string to_string() {
    cerr << time_step << " " << state << " " << center << " ";
    dictionary->show();
    return "";
  }

  const DictTree *dictionary;

  int time_step;

  int state;
  int center;

};

class ThinState {
 public:
  ThinState() {}
 ThinState(const DictTree *_dict,
          int _center) 
    : dictionary(_dict),
    center(_center) {}

  string to_string() {
    cerr << center << " ";
    dictionary->show();
    return "";
  }

  const DictTree *dictionary;
  int center;
};

class Heuristic {
 public:
 Heuristic(int num_timesteps, int num_states, int num_centers)
   : scores_(num_timesteps) {
    for (unsigned int i = 0; i < scores_.size(); ++i) {
      scores_[i].resize(num_states);
      for (unsigned  int j = 0; j < scores_[i].size(); ++j) {
        scores_[i][j].resize(num_centers);
      }
    }
  }

  double score(int time_step, int state, int center) const {
    //cerr << time_step << " " << state << " " << center << " " << scores_[time_step][state][center] << endl;
    return scores_[time_step][state][center];
  }

  void set_score(int time_step, int state, int center, double value) {
    scores_[time_step][state][center] = value;
  }

 private:
  vector<vector<vector<double> > > scores_;
};

class Scorer {
 public:
 Scorer(int num_timesteps, int num_states, int num_centers) 
   : scores_(num_timesteps), lambda_(num_states) {
    for (unsigned int i = 0; i < lambda_.size(); ++i) {
      lambda_[i].resize(num_centers);
    }
    for (unsigned int i = 0; i < scores_.size(); ++i) {
      scores_[i].resize(num_states);
      for (unsigned  int j = 0; j < scores_[i].size(); ++j) {
        scores_[i][j].resize(num_centers);
      }
    }    
  }

  double score(int time_step, int state, int center) const {
    return scores_[time_step][state][center];
  }

  double lambda(int state, int center) const {
    return lambda_[state][center];
  }

  void set_lambda(int state, int center, double value) {
    lambda_[state][center] = value;
  }

  void set_score(int time_step, int state, int center, double value) {
    scores_[time_step][state][center] = value;
  }

  
 private:
  vector<vector<vector<double> > > scores_;
  vector<vector<double> >  lambda_;
};

class Expander {
 public:
 Expander(const Scorer *scorer, const Heuristic *heuristic, const ClusterProblem *cp, 
          const vector<bool> &enforce_consistency) 
   : scorer_(scorer), 
    heuristic_(heuristic),
    cp_(cp) {
      // Precompute important information. 
      vector<int> type_information;
      for (int i = 0; i < cp_->num_states; ++i) {
        int type = cp_->MapState(i);
        type_information.push_back(type);
      }
      seen_before_.resize(cp_->num_states, false);
      seen_after_.resize(cp_->num_states, false);
      //int upto = 3;
      for (int i1 = 0; i1 < cp_->num_states; ++i1) {
        if (!enforce_consistency[cp_->MapState(i1)]) continue;
        for (int i2 = 0; i2 < i1; ++i2) {
          if (type_information[i1] == type_information[i2]) {
            seen_before_[i1] = true;
            break;
          }
        }

        for (int i2 = i1 + 1; i2 < cp_->num_states; ++i2) {
          if (type_information[i1] == type_information[i2]) {
            seen_after_[i1] = true;
            break;
          }
        }
      }
    }

  Node<HMMState> *start() const {
    HMMState *state = new HMMState(new DictTree(), -1, -1, -1);
    return new Node<HMMState>(0, 0, state);
  }

  bool is_final(const Node<HMMState> &start) const {
    //TODO
    return (start.state->state == cp_->num_states - 1) && 
      (start.state->time_step == cp_->num_steps - 1);
  }

  void Expand(const Node<HMMState> &start, 
              vector<Node<HMMState> > *children) const {
    // Stay in the same state.
    if (start.state->time_step >= cp_->num_steps - 1) return;
    int new_timestep = start.state->time_step + 1;
    if (new_timestep != 0) {
      int state = start.state->state;
      int center = start.state->center;
      double score = start.score + 
        scorer_->score(new_timestep, state, center);
      double heuristic = heuristic_->score(new_timestep, state, center);
      HMMState *new_state =
        new HMMState(start.state->dictionary, 
                     new_timestep, 
                     state, 
                     center);
      children->push_back(Node<HMMState>(score, heuristic, new_state));
    }

    // Move to the next state.
    if (start.state->state < cp_->num_states - 1) {
      int old_state = start.state->state;
      int old_phoneme = cp_->MapState(old_state);
      int old_center = start.state->center;
      int new_state = old_state + 1;
      int new_phoneme = cp_->MapState(new_state);
      const DictTree *new_dictionary; 
      if (old_state == -1) {
        new_dictionary = start.state->dictionary;
      } else {
        bool update_hash = true;
        if (seen_before_[old_state]) {
          if (seen_after_[old_state]) { 
            update_hash = false;
          } else {
            update_hash = false;
          }
        } else if (seen_after_[old_state]) {
          update_hash = true;
        } else {
          update_hash = false;
        }
        new_dictionary = 
          start.state->dictionary->add(new_timestep, 
                                       old_phoneme, 
                                       old_center, 
                                       cp_->num_hidden(old_phoneme),
                                       update_hash);
      }

      if (seen_before_[new_state]) {
        int center = new_dictionary->center(new_phoneme);
        HMMState *state = 
          new HMMState(new_dictionary, new_timestep, new_state, center);
        double heuristic = heuristic_->score(new_timestep, new_state, center);
        double score = start.score + 
          scorer_->score(new_timestep, new_state, center) + 
          scorer_->lambda(new_state, center);
        children->push_back(Node<HMMState>(score, heuristic, state));    
      } else {
        for (int center = 0; center < cp_->num_hidden(new_phoneme); ++center) {
          // If it is seen again, need to remember state. 
          double heuristic = 
            heuristic_->score(new_timestep, new_state, center);
          double score = start.score + 
            scorer_->score(new_timestep, new_state, center) + 
            scorer_->lambda(new_state, center); 
          HMMState *hmm_state = 
            new HMMState(new_dictionary, new_timestep, new_state, center);
          children->push_back(Node<HMMState>(score, heuristic, hmm_state));
        }
      }
    }
  }

  int hash(const Node<HMMState> &node) const {
    int hash = cp_->num_steps * cp_->num_states * cp_->num_hidden(0) * node.state->dictionary->hash() + 
      cp_->num_steps * cp_->num_states * node.state->center +
      cp_->num_steps * node.state->state + node.state->time_step;
    //node.state->dictionary->show();
    
    //cerr << endl << node.state->center << " " << node.state->state << " " << node.state->time_step << " " << hash % 10000000 << endl;
    return hash;
  }

 private:
  const Scorer *scorer_;
  const Heuristic *heuristic_;
  const ClusterProblem *cp_;

  vector<bool> seen_before_;
  vector<bool> seen_after_;

};

class Merger : public BaseMerger<ThinState> {
 public:
  Merger(const Scorer *scorer, 
         const Heuristic *heuristic, 
         const ClusterProblem *cp) 
    : scorer_(scorer),
    heuristic_(heuristic),
    cp_(cp)
    {}
  
  void MergeBeams(int time, int state, const Beam<ThinState> &one, 
                  const Beam<ThinState> &two, 
                  Beam<ThinState> * new_beam) const {
    
    int phoneme = cp_->MapState(state);
    if (one.element(0).state->dictionary->center(phoneme) != -1) {
      SimpleMerge(time, state, one, two, new_beam);
    } else {
      ComplexMerge(time, state, one, two, new_beam);
    }
  }

 private:
  const Scorer *scorer_;
  const Heuristic *heuristic_;
  const ClusterProblem *cp_;

  struct Next {
    Next(bool s, int g, double sc) 
    : stay(s), go(g), score(sc) {} 
    bool stay;
    int go; 
    double score;
    int operator<(const Next &other) const {
      return score > other.score;
    }
  };

  double new_stay_score(int time, int state, 
                   const Node<ThinState> &node) const {
    return node.score + 
      scorer_->score(time, state, node.state->center);
  }
  double new_switch_score(int time, int state, 
                          const Node<ThinState> &node, int center) const {
    return node.score + 
      scorer_->score(time, state, center) + 
      scorer_->lambda(state, center);
  }


  double heuristic(int time, int state, 
                   const Node<ThinState> &node) const {
    return
      heuristic_->score(time, state, node.state->center);
  }

  double total_stay (int time, int state, 
                     const Node<ThinState> &node) const {
    return new_stay_score(time, state, node) + heuristic(time, state, node);
  }

  double total_switch(int time, int state, 
                      const Node<ThinState> &node, int center) const {
    return node.score + 
      scorer_->score(time, state, center) + 
      scorer_->lambda(state, center) + 
      heuristic_->score(time, state, center);
  }

  void ComplexMerge(int time, int state,
                   const Beam<ThinState> &last_beam, 
                   const Beam<ThinState> &cur_beam, 
                   Beam<ThinState> * new_beam) const {
    vector<pair<double, int> >ordered_centers(cp_->num_hidden(0));
    for (int c = 0; c < cp_->num_hidden(0); ++c) {
      double extra_score = scorer_->score(time, state, c) + 
        scorer_->lambda(state, c) + 
        heuristic_->score(time, state, c);
      ordered_centers.push_back(pair<double, int>(extra_score, c));
    }
    sort(ordered_centers.begin(), ordered_centers.end());



    // The current centers for nodes in last_berm
    vector<int> pointer_leave(1, 0);
    

    // The current position for nodes in cur_beam.
    int pointer_stay = 0;

    priority_queue<Next> queue;

    // Initialize cur beam.
    if (cur_beam.elements() > 0) {
      double total = total_stay(time, state, cur_beam.element(0));
      Next next(true, -1, total);
      queue.push(next);
    }

    // Initialize last beam.
    if (last_beam.elements() > 0) {
      double total = 
        total_switch(time, state, last_beam.element(0), 
                     ordered_centers[0].second);
      Next next(false, 0, total);
      queue.push(next);
    }
       
    while (!queue.empty() && new_beam->elements() < new_beam->k()) {
      Next next = queue.top();
      queue.pop();
      if (next.stay) {
        // Add node from cur_beam.
        Node<ThinState> old_node = 
          cur_beam.element(pointer_stay);
        int center = old_node.state->center;
        double heu = heuristic_->score(time, state, center);
        double score = new_stay_score(time, state, old_node);
        Node<ThinState> new_node(score, heu, old_node.state);
        new_beam->Add(new_node);
        pointer_stay++;
        if (pointer_stay < cur_beam.elements()) {
          double total = 
            total_stay(time, state, 
                        cur_beam.element(pointer_stay));
          Next next(true, -1, total);
          queue.push(next);
        }
      } else {
        // Add node from last beam.
        Node<ThinState> old_node = last_beam.element(next.go);
        int center = ordered_centers[pointer_leave[next.go]].second;
        double heu = heuristic_->score(time, state, center);
        double score = old_node.score + 
          scorer_->score(time, state, center);
        //todo
        int old_phoneme = cp_->MapState(state - 1);
        ThinState *new_state = 
          new ThinState(
                        old_node.state->dictionary->add(time, old_phoneme, 
                                                        old_node.state->center, cp_->num_hidden(0), true), 
                        center);
        new_beam->Add(Node<ThinState>(score, heu, new_state));

        pointer_leave[next.go]++;
        if (pointer_leave[next.go] < (int)ordered_centers.size()) {
          double total = 
            total_switch(time, state, 
                         last_beam.element(next.go), 
                         ordered_centers[pointer_leave[next.go]].second); 
          Next next(false, next.go, total);
          queue.push(next);
        }
        
        if (next.go + 1 == (int)pointer_leave.size() && 
            next.go + 1 < last_beam.elements()) {
          pointer_leave.resize(next.go + 1, 0);
          double total = 
            total_switch(time, state, 
                         last_beam.element(next.go + 1), 
                         ordered_centers[0].second); 
          Next next(false, next.go + 1, total);
          queue.push(next);
        } 
      } 
    } 
  }

  void SimpleMerge(int time, int state,
                   const Beam<ThinState> &one, 
                   const Beam<ThinState> &two, 
                   Beam<ThinState> * new_beam) const {
    int pointer1 = 0, pointer2 = 0;
    const Node<ThinState> *node1, *node2;
    do {
      node1 = (pointer1 < one.elements()) ? (&one.element(pointer1)) : (NULL);
      node2 = (pointer2 < two.elements()) ? (&two.element(pointer2)) : (NULL);
      double score1 = INF;
      double heuristic1 = INF;
      int center;
      if (node1 != NULL) {

        int phoneme = cp_->MapState(state);
        center = node1->state->dictionary->center(phoneme);
        heuristic1 = heuristic_->score(time, state, center);
        score1 = new_switch_score(time, state, *node1, center);
      }
                                               
      double score2 = INF;
      double heuristic2 = INF;
      if (node2 != NULL) {
        score2 = new_stay_score(time, state, *node2);
        heuristic2 = heuristic(time, state, *node2);
      }
      if (score1 + heuristic1 < score2 + heuristic2) {
        // Extend node1.
        int old_phoneme = cp_->MapState(state - 1);
        ThinState *new_state = 
          new ThinState(
                        node1->state->dictionary->add(time, old_phoneme, 
                                                      node1->state->center, cp_->num_hidden(0), true), 
                        center);
        
        new_beam->Add(Node<ThinState>(score1, heuristic1, new_state));
        pointer1++;
      } else if (node2 != NULL) {
        new_beam->Add(Node<ThinState>(score2, heuristic2, node2->state));
        pointer2++;
      }
    } while ((node1 != NULL || node2 != NULL) && 
             new_beam->elements() < new_beam->k());
  }
};

typedef AStar<HMMState, Expander> AStarMemory;

#endif
