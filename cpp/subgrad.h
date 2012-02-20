#ifndef SUBGRAD_H
#define SUBGRAD_H

#include "data_point.h"
#include <vector>
using namespace std;

struct SubgradResult {
  // The dual value of the subgradient.
  double dual_value;

  // The primal value of the subgradient.
  double primal_value;

  // The computed subgradient as a vector.
  DataPoint subgradient;
};

struct SubgradInfo {
public:
SubgradInfo(int _round,
            const vector<double> &_duals, 
            const vector<double> &_primals
            ) : round(_round), duals(_duals), primals(_primals) {}
  
  // The current subgrad round.
  int round;

  // The preceeding dual values.
  const vector<double> &duals;

  // The preceeding primal values.
  const vector<double> &primals;
};

class SubgradProblem {
 public:
  virtual ~SubgradProblem() {}
  
  // Solve the problem with the current subgrad.
  virtual void Solve(const SubgradInfo &info, 
                     SubgradResult *result) = 0;

  // Update the weights with vector.
  virtual void Update(const DataPoint &data_point,
                      double alpha) = 0;
};

class SubgradRate {
 public:
  virtual ~SubgradRate() {}
  virtual double get_rate(const SubgradInfo&) = 0;
};

class ExponentialWaitRate : public SubgradRate {
 public:
 ExponentialWaitRate(double start_rate) 
   : start_rate_(start_rate) {}


  double get_rate(const SubgradInfo&);
 private:
  // The rate to start with.
  double start_rate_;

  double current_rate_;

  // Number of times dual has gone bad. 
  int bad_count_; 
};

class SubgradSolver {
 public: 
 SubgradSolver(SubgradProblem *problem) : 
  round_(1), 
  iterations_(1000),
  debug_(false),
  problem_(problem) {}

  // Run the subgradient algorithm.
  void Run();
  
  // Run the subgradient algorithm for one round.
  bool RunOneRound();

  // Set the max number of iterations.
  void set_iterations(int iterations) {
    iterations_ = iterations;
  }

  void set_rate(SubgradRate *rate) {
    rate_ = rate;
  }

  void set_debug(bool debug) {
    debug_ = debug;
  }

 private:  
  // The current round. 
  int round_;

  // Max number of iterations.
  int iterations_;
  
  // Debug mode.
  bool debug_;
  
  // Set a multiplier rate for the alpha
  SubgradRate *rate_;

  // The internal dual problem.
  SubgradProblem *problem_;

  // The running dual values.
  vector<double> dual_values_;
  vector<double> primal_values_;

  // The best seen dual and primal values.
  double best_dual_value_;
  double best_primal_value_;
};

#endif
