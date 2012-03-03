#ifndef ALIGNMENT_LP_H
#define ALIGNMENT_LP_H

#include "gurobi_c++.h"
#include "speech_problem.h"

class AlignmentLP {
 public:
 AlignmentLP(SpeechProblemSet &speech_problem) :
  speech_problem_(speech_problem), cs_(speech_problem_.MakeClusterSet()) {
    for (int u = 0; u < cs_.problems_size(); ++u) {
      distances_.push_back(speech_problem_.MakeDistances(u));
    }
  } 

  void ConstructLP();

 private:
    const SpeechProblemSet &speech_problem_;
    ClusterSet cs_;
    vector<ThinDistanceHolder *> distances_;
    
    vector< vector < vector <vector <vector <GRBVar > > > > > x; 
    vector <vector <vector <GRBVar > > >  y; 
    vector <vector <GRBVar > >  z; 
};

#endif
