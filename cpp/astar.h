#ifndef ASTAR_H
#define ASTAR_H

#include <vector>
#include <list>
#include <assert.h>
#include <queue>
#include <iostream>
#include <algorithm>
#define HASH_SIZE 10000000

using namespace std;



template<typename STATE>
struct Node {
Node(double _score, double _heuristic, STATE *_state) : 
  score(_score), heuristic(_heuristic), state(_state) {}
  Node() {} 
  Node(const Node<STATE> &node)
  : score(node.score),
    heuristic(node.heuristic),
    state(node.state) {}
  double total() const { return score + heuristic; }
  double score;
  double heuristic;
  STATE *state;

  void show() {
    state->to_string();
  }
  int operator<(const Node<STATE> &other) const {
    return (score + heuristic) > (other.score + other.heuristic);
  }
};

template<typename STATE, typename EXPANDER>
class Search {
 public:
  virtual double Run(STATE *state, int *rounds) = 0;
};

template<typename STATE, typename EXPANDER>
  class AStar : public Search<STATE, EXPANDER> {
 public:
 AStar(const EXPANDER *expander, double upper_bound)
   : expander_(expander), hash_(HASH_SIZE, false), hash_score_(HASH_SIZE), upper_bound_(upper_bound) {}

  virtual double Run(STATE *state, int *rounds) {
    Node<STATE> *start = expander_->start();
    queue_.push(*start);
    Node<STATE> node;
    while (RunRound(*rounds, &node)) {
      (*rounds)++;
    }
    cerr << "Found best " << node.score << " " << node.heuristic << "" << *rounds <<  endl;
    node.state->to_string();
    (*state) = (*node.state);
    return node.score;
  }

 private:
  bool RunRound(int round, Node<STATE> *node) {
    do {
      (*node) = queue_.top();
      queue_.pop();
      if (expander_->is_final(*node)) return false;
      int hash = abs(expander_->hash(*node) % HASH_SIZE);
      double score = node->heuristic + node->score;
      if (score <= hash_score_[hash]) break;
      else delete node->state;
    } while(true);
    int num = 100000;
    if (round % num == 0) {
      cerr << round << " " << node->score << " " << node->heuristic << " ";
      node->state->to_string();
      cerr << endl;
      cerr << "CUR SCORE: " << node->heuristic + node->score << " " << (clock() - time_) /  (float) num << " " << queue_.size() << endl; 
      time_ = clock();
    }
    vector<Node<STATE> > children;
    expander_->Expand(*node, &children, true, upper_bound_);
    for (unsigned int i = 0; i < children.size(); ++i) {
      Node<STATE> state = children[i];
      int child_hash = abs(expander_->hash(state) % HASH_SIZE);
      double score = state.heuristic + state.score;
      if (hash_[child_hash] == false || 
          score < hash_score_[child_hash] ) {
        hash_score_[child_hash] = score;
        hash_[child_hash] = true;
        queue_.push(state);
      }
    }
    delete node->state;
    return true;
  }

 private:
  priority_queue<Node<STATE> > queue_;
  const EXPANDER *expander_;
  vector<bool> hash_;
  vector<double> hash_score_;
  int time_; 
  double upper_bound_;
};

template<typename STATE, typename EXPANDER>
  class BeamSearch : public Search<STATE, EXPANDER> {
 public:
 BeamSearch(const EXPANDER *expander, int beam)
   : expander_(expander), queue_(expander->TotalOrder()), beam_(beam) {}

  virtual double Run(STATE *state, int *rounds) {
    rounds_ = 0;
    Node<STATE> *start = expander_->start();
    int order = expander_->order(*start);
    AddBeam(*start, &queue_[order]);
    Node<STATE> node;
    for (int order = 0; ;++order) {
      if (!RunRound(order, &node)) break;
    }
    
    cerr << "Found beam " << node.score << " " << node.heuristic << " " << rounds_ << " " << expander_->TotalOrder() << endl;
    node.state->to_string();
    (*state) = (*node.state);
    
    (*rounds) = rounds_;
    return node.score;    
  }

 private:
    /* // Binary search. */
    /* int bottom = 0; */
    /* int cur = beam->size() / 2; */
    /* int top = beam->size() - 1; */
    /* while (true) { */
    /*   double check = (*beam)[cur].total(); */
    /*   if (node.total() < check) { */
    /*     cur = (cur - bottom) / 2;  */
    /*     top = cur; */
    /*   } */
    /*   if (node.total() > check) { */
    /*     cur = (top - cur) / 2;  */
    /*     bottom = cur; */
    /*   } */
    /*   if (cur == top || cur == bottom) { */
    /*     break; */
    /*   }  */
    /* } */
  double WorstBeam(const list<Node<STATE> > &beam) {
    return beam.back().total();
  }

  double BestBeam(const list<Node<STATE> > &beam) {
    return beam.begin()->total();
  }

  double BeamFull(const list<Node<STATE> > &beam) {
    return (int)beam.size() == beam_;
  }

  void AddBeam(const Node<STATE> &node, list<Node<STATE> > *beam) {
    bool full = (int)beam->size() == beam_;
    if (beam->empty())  beam->push_back(node);
    if (node.total() > beam->back().total()) {
      if (full) {
        return;
      } else {
        beam->push_back(node);
      }
    }

    typename std::list< Node<STATE > >::iterator it;
    int i = 0;
    /* cerr << "start " << endl; */
    /* for (it = beam->begin();  */
    /*      it != beam->end(); */
    /*      ++it) { */
    /*   it->show(); */
    /* } */
    /* cerr << "end" << endl; */


    for (it = beam->begin(); 
         it != beam->end();
         ++it) {
      ++i;
      if (node.total() < it->total()) {
        beam->insert(it, node);
        if (full) beam->pop_back();
        break;
      }
    }
  }

  bool RunRound(int order, Node<STATE> *node) {
    typename std::list< Node<STATE > >::iterator it;
    /* cerr << "start" << endl; */
    /* for (it = queue_[order].begin();  */
    /*      it != queue_[order].end(); */
    /*      ++it) { */
    /*   it->show(); */
    /*   cerr << endl << endl; */
    /* } */
    /* cerr << "end" << endl; */
    for (it = queue_[order].begin(); 
         it != queue_[order].end();
         ++it) {
      (*node) = *it;
      if (expander_->is_final(*node)) return false;
      vector<Node<STATE> > children;
      expander_->order_next_state(*node);
      //int next_order = 
      bool use_worst = false;
      double worst = 0.0;
      /* if (!queue_[next_order].empty()) { */
      /*   if (BeamFull(queue_[next_order])) { */
      /*     worst = WorstBeam(queue_[next_order]); */
      /*     worst = min(BestBeam(queue_[next_order]) + 50, worst); */
      /*   } else { */
      /*     worst = BestBeam(queue_[next_order]) + 50; */
      /*   } */
      /*   worst = min(worst, 700.0); */
      /* } else { */
      /*   worst = 700.0; */
      /* } */

      expander_->Expand(*node, &children, use_worst, worst);
      rounds_++;
      for (unsigned int i = 0; i < children.size(); ++i) {
        Node<STATE> state = children[i];
        //double score = state.heuristic + state.score;
        int new_order = expander_->order(state);
        //cerr << i << " " << new_order << endl;
        if (state.total() < 700) {
          AddBeam(state, &queue_[new_order]);
        }
      }
      delete node->state;
    }
    queue_[order].clear();
    //delete node->state;
    return true;
  }

 private:
  const EXPANDER *expander_;
 
  int time_; 
 
  vector<list<Node<STATE> > > queue_;
 int beam_;
 int rounds_;
};

#endif
