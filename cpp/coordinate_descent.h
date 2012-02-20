#ifndef COORDINATE_DESCENT_H
#define COORDINATE_DESCENT_H

class MPLPProblem {
 public:
  virtual ~SubgradProblem() {}
  
  // Solve the problem with the current subgrad.
  virtual void Solve(const SubgradInfo &info, 
                     SubgradResult *result) = 0;

  // Update the weights with vector.
  virtual void Update(const DataPoint &data_point,
                      double alpha) = 0;
};


class MPLPSolver {
 public: 
 MPLPSolver(SubgradProblem *problem) : 
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

  
  void set_debug(bool debug) {
    debug_ = debug;
  }
}

#endif
