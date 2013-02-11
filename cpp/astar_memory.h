
#ifndef ASTAR_MEMORY_H
#define ASTAR_MEMORY_H

#include <vector>
#include <algorithm>
#include <assert.h>
#include "astar.h"
#include "cluster_problem.h"
using namespace std;

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

  int order(const Node<HMMState> &node) const {
    return (cp_->num_steps + 1) * (node.state->state + 1)
      + (node.state->time_step + 1);
  }

  int order_next_state(const Node<HMMState> &node) const {
    return (cp_->num_steps + 1) * (node.state->state + 2)
      + (node.state->time_step + 2);
  }

  int TotalOrder() const {
    return (cp_->num_steps + 2) * (cp_->num_states + 2);
  }

  bool is_final(const Node<HMMState> &start) const {
    //TODO
    return (start.state->state == cp_->num_states - 1) && 
      (start.state->time_step == cp_->num_steps - 1);
  }

  void Expand(const Node<HMMState> &start, 
              vector<Node<HMMState> > *children,
              bool use_worst,
              double worst) const {
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
        double heuristic = heuristic_->score(new_timestep, new_state, center);
        double score = start.score + 
          scorer_->score(new_timestep, new_state, center) + 
          scorer_->lambda(new_state, center);
        if (!use_worst || heuristic + score < worst) {
          HMMState *state = 
            new HMMState(new_dictionary, new_timestep, new_state, center); 
          children->push_back(Node<HMMState>(score, heuristic, state));    
        }
      } else {
        for (int center = 0; center < cp_->num_hidden(new_phoneme); ++center) {
          // If it is seen again, need to remember state. 
          double heuristic = 
            heuristic_->score(new_timestep, new_state, center);
          double score = start.score + 
            scorer_->score(new_timestep, new_state, center) + 
            scorer_->lambda(new_state, center); 
          if (!use_worst || heuristic + score < worst) {
            HMMState *hmm_state = 
              new HMMState(new_dictionary, new_timestep, new_state, center);
            children->push_back(Node<HMMState>(score, heuristic, hmm_state));
          }
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

typedef AStar<HMMState, Expander> AStarMemory;
typedef BeamSearch<HMMState, Expander> BeamMemory;

#endif
