#ifndef ASTAR_H
#define ASTAR_H

#include <vector>
#include <assert.h>
#include <queue>
#include <iostream>
#define HASH_SIZE 10000000

using namespace std;

template<class STATE>
struct Node {
Node(double _score, double _heuristic, STATE *_state) : 
  score(_score), heuristic(_heuristic), state(_state) {}
  Node() {} 
  Node(const Node<STATE> &node)
  : score(node.score),
    heuristic(node.heuristic),
    state(node.state) {}
  double score;
  double heuristic;
  STATE *state;

  
  int operator<(const Node<STATE> &other) const {
    return (score + heuristic) > (other.score + other.heuristic);
  }
};

template<class STATE, class EXPANDER>
class AStar {
 public:
  AStar(const EXPANDER *expander)
    : expander_(expander), hash_(HASH_SIZE, false), hash_score_(HASH_SIZE) {}

  double Run(STATE *state, int *rounds) {
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
    int num = 10000000;
    if (round % num == 0) {
      cerr << round << " " << node->score << " " << node->heuristic << " ";
      node->state->to_string();
      cerr << endl;
      cerr << "CUR SCORE: " << node->heuristic + node->score << " " << (clock() - time_) /  (float) num << " " << queue_.size() << endl; 
      time_ = clock();
    }
    vector<Node<STATE> > children;
    expander_->Expand(*node, &children);
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
};

#endif
