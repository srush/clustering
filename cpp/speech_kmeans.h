#ifndef SPEECH_KMEANS_H
#define SPEECH_KMEANS_H

#include "distances.h"
#include "speech_problem.h"
#include "speech.h"


// Setup for the speech subgradient solver.
class SpeechKMeans {
 public:
  SpeechKMeans(const SpeechProblemSet &problem);

  // Run the k-means algorithm to convergence.
  double Run(int rounds);  

  // Random initialization of the centers.
  void InitializeCenters();

  // Set the centers to start EM.
  void SetCenters(const vector<vector<DataPoint> > &centers);
  
  void set_use_medians(bool use_medians) {
    use_medians_ = use_medians;
  }

  const vector<vector<DataPoint> > &centers() { return centers_; } 

  SpeechSolution *MakeSolution();

 private:

  // EM.
  double Expectation(int utterance_index,
                     int *correctness,
                     vector<vector<vector<DataPoint> > > *sets
                     );
  void Maximization(const vector<vector<vector<DataPoint> > > &sets);

  
  // The information of the underlying speech problem
  const SpeechProblemSet &problems_;

  // Precomputed terms for solvers.
  const ClusterSet &cluster_problems_;

  // The centers of the clusters.
  vector<vector<DataPoint> > centers_;

  // The number of types.
  int num_types_;
  int num_features_;

  // Force medians instead of means.
  bool use_medians_;

  vector<ThinDistanceHolder *> distances_;

  // The last path from the expectation state.
  vector<vector<int> > path_;
  vector<vector<int> > mode_centers_;
  
};

#endif
