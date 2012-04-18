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

  void set_use_gmm() {
    use_gmm_ = true;
  }

  void set_use_unsup() {
    use_unsupervised_ = true;
  }

  const vector<vector<DataPoint> > &centers() { return centers_; } 

  SpeechSolution *MakeSolution();

 private:
  vector<DataPoint> WeightedKMeans(vector<DataPoint> &points, 
                                   vector<double> &weights, int k);

  // EM.
  double Expectation(int utterance_index,
                     int *correctness,
                     vector<vector<vector<DataPoint> > > *sets
                     );
  void Maximization(const vector<vector<vector<DataPoint> > > &sets);

  void GMMMaximization(const vector<vector<DataPoint> > &estimators, 
                       const vector<vector<double> > &counts);

  double GMMExpectation(int utterance_index,
                        vector<vector<DataPoint> > &estimators, 
                        vector<vector<double> > &counts);
  void ClusterSegmentsExpectation(int utterance_index,
                                  vector<DataPoint> *points,
                                  vector<double> *weights);

  void ClusterSegmentsMaximization(vector<DataPoint> *points,
                                   vector<double> *weights);

  double UnsupExpectation(int utterance_index,
                          int *correctness,
                          vector<vector<vector<DataPoint> > > *sets);

  void UnsupMaximization(const vector<vector<vector<DataPoint> > > &sets);
  
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

  bool use_gmm_;

  bool use_unsupervised_;

  bool unsup_initialized_;
  vector<ThinDistanceHolder *> distances_;

  // The last path from the expectation state.
  vector<vector<int> > path_;
  vector<vector<int> > mode_centers_;
  vector<vector<int> > type_centers_;
  
};

#endif
