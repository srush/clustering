#ifndef BEAM_SEARCH_H
#define BEAM_SEARCH_H

#include "astar.h"
 
template <class STATE>  
class Beam {
 public: 
 Beam() : k_(-1), upper_(1e10), finished_(false) {}
  //Beam(int k, int hash_size) : k_(k), upper_(1e10), hash_(hash_size, false), finished_(false) {}
 
  const Node<STATE> &element(int i ) const { return beam_[i]; }
    
  int elements() const { return beam_.size(); }

  bool Add(const Node<STATE> &node, int hash) {
    if (node.total() > upper_ + 0.01) {
      finished_ = true;
      return false;
    }
    if (hash_[abs(hash % (int)hash_.size())]) {
      return false;
    }
    hash_[abs(hash % (int)hash_.size())] = true;
    if (beam_.size() > 0) {
      assert(node.total() >= beam_.back().total());
    }
    beam_.push_back(node);
    return true;
  }

  bool InUpperRange(double score) {
    return score <= upper_;
  }

  bool HasHash(int hash) {
    return hash_[abs(hash % (int)hash_.size())];
  }
  
  bool Finished() {
    return elements() == k_ || finished_;
  }
  
  //int k() { return k_; }
  void set_k(int k) { k_ = k; }
  void set_upper(double upper) { upper_ = upper; }
  void set_hash(int hash) { hash_.resize(hash, false); }

 private:
  int k_; 
  double upper_;
  vector<Node<STATE> > beam_;
  vector<bool> hash_;
  bool finished_;
};

template <class STATE>
class BaseMerger {
 public:
  virtual void Initialize(Beam<STATE> *start) const = 0;
  virtual void MergeBeams(int time, int state, const Beam<STATE> &one, 
                  const Beam<STATE> &two, 
                  Beam<STATE> * new_beam) const = 0;
  virtual void show_stats() const = 0;
};

template <class STATE>  
class FastBeamSearch {
 public:
  FastBeamSearch(int num_states, int num_timesteps, int num_centers) 
    : num_states_(num_states),
    num_steps_(num_timesteps),
    num_centers_(num_centers),
    chart_(num_timesteps + 1) {
      for (int t = 0; t <= num_timesteps; ++t) {
        chart_[t].resize(num_states + 1);
      }
    }

  void Initialize() {
    //chart_[0][0].Add();
  }

  double Run(int k, const BaseMerger<STATE> &merger, STATE *state, double upper_bound, bool exact) {
    clock_t start = clock();
    for (int time = 0; time < num_steps_; ++time) {
      for (int state = 0; state < num_states_; ++state) {
        if (!exact) {
          chart_[time][state].set_k(k);
          chart_[time][state].set_upper(upper_bound + 10);
          chart_[time][state].set_hash(100000);
        } else { 
          //chart_[time][state].set_k(k);
          chart_[time][state].set_upper(upper_bound);
          chart_[time][state].set_hash(1000000);
        }

        if (time == 0) {
          if (state != 0) continue;
          Beam<STATE> temp;
          temp.set_hash(1000);
          Beam<STATE> empty; 
          merger.Initialize(&temp);
          merger.MergeBeams(time, 
                            state,
                            temp,
                            empty,
                            &chart_[time][state]);
        } else if (state == 0) {
          Beam<STATE> empty; 
          merger.MergeBeams(time, 
                            state,
                            empty,
                            chart_[time - 1][state], 
                            &chart_[time][state]);
        } else {
          merger.MergeBeams(time, 
                            state, 
                            chart_[time - 1][state - 1], 
                            chart_[time - 1][state], 
                            &chart_[time][state]);
        }
        if (chart_[time][state].elements() > 0) {
          /* cerr << time << "\t" << state << "\t" << */
          /*   chart_[time][state].element(0).total() << "\t" << */
          /*   chart_[time][state].element(0).score << "\t" << */
          /*   chart_[time][state].element(0).heuristic << "\t" <<  */
          /*   chart_[time][state].elements() << endl; */
          /* for (int i = 0; i < chart_[time][state].elements(); ++i) { */
          /*   chart_[time][state].element(i).state->to_string(); */
          /* } */
        }
      }
      cerr << time << endl; 
    } 

    cerr << "TIME: Beam search round: " << clock() - start  << " " << upper_bound << endl;
    merger.show_stats();
    *state = *chart_[num_steps_ - 1][num_states_ - 1].element(0).state;
    cerr << chart_[num_steps_ - 1][num_states_ - 1].element(0).heuristic;
    return chart_[num_steps_ - 1][num_states_ - 1].element(0).score;
  }

 private:
  int num_states_;
  int num_steps_;
  int num_centers_;
  vector<vector<Beam<STATE> > > chart_;
  
};

#endif
