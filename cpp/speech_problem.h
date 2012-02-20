#ifndef SPEECH_PROBLEM_H
#define SPEECH_PROBLEM_H

#include "distances.h"
#include "speech.h"
#include "cluster_problem.h"
#include "speech_solution.h"

class SpeechProblem {

};

struct StateLocation {
StateLocation(int _problem, int _state): problem(_problem), state(_state) {}
  int problem;
  int state;
};


class SpeechProblemSet {
 public:
  SpeechProblemSet(const vector<Utterance *> &utterance,
                   vector<Center> *centers);

  // Precompute the distances of each start-end pair.
  DistanceHolder *MakeDistances(int problem) const;
  
  // Compute the underlying cluster problem.
  const ClusterSet &MakeClusterSet() const;

  // Construct the ball set for a distance epsilon.
  BallHolder *MakeBalls(vector<int> epsilon) const;

  // Read the speech problem from a protobuf file.
  static SpeechProblemSet *ReadFromFile(string, string, string);  

  /* const DataPoint &sequence(int i) const { */
  /*   return utterance_.sequence(i); */
  /* }  */

  const Utterance &utterance(int utterance_index) const {
    return *utterances_[utterance_index];
  }

  int utterance_size() const {
    return utterances_.size();
  }

  int num_features() const {
    //return utterance_.num_features();
    return num_features_;
  }

  int num_types() const {
    return utterances_[0]->num_phonemes();
  }

  int type_occurence_size(int type) const {
    return state_locations_[type].size();
  }

  const StateLocation &type_occurence(int type, int occurence) const {
    return state_locations_[type][occurence];
  }

  // Convert an alignment to a set of points in each phoneme cluster.
  void AlignmentClusterSet(int problem,
                           const vector<int> &alignment, 
                           vector<vector<DataPoint> > *cluster_sets) const;

  // Convert an alignment to a set of groups of points in each phoneme cluster.
  void AlignmentGroupClusterSet(int problem,
                                const vector<int> &alignment, 
                                vector<vector<vector<DataPoint> > > *cluster_sets) const;
  
  // Choose arbitrary centers that maximize an alignment cluster set.
  double MaximizeCenters(const vector<vector<DataPoint> > &cluster_sets,
                         vector<DataPoint> *centers) const;
  
  // Find the centroid of a start end combination.
  DataPoint Centroid(int problem, int start, int end) const;

  // Choose fixed centers that optimize an alignment cluster set.
  //double MaximizeMedians(const vector<vector<vector<DataPoint> > > &cluster_sets,
  //                     const BallHolder &ball_holder) const;

  double MaximizeMedians(const SpeechSolution &bad_speech_solution,
                         vector<DataPoint> *centroids) const;

  double MaximizeMediansHidden(const SpeechSolution &bad_speech_solution,
                               vector<DataPoint> *centroids) const;

  SpeechSolution *ApproxMaximizeMedians(
                 const SpeechSolution &bad_speech_solution, 
                 const BallHolder &ball_holder,
                 double *score) const;

  /* double ApproxMaximizeMedians(const vector<vector<vector<DataPoint> > > &cluster_sets,  */
  /*                              const BallHolder &ball_holder) const; */
 private:
  // Precompute the locations for each type.
  void CacheTypeOccurence();

  // The speech utterance to solve.
  const vector<Utterance *> &utterances_;

  ClusterSet *cluster_set_;

  // Possible centers.
  vector<Center> *centers_;

  int num_features_;

  // State locations for each type.
  vector<vector<StateLocation> > state_locations_;
};

#endif
