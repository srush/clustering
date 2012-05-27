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
    
    vector< vector< vector < vector <vector <GRBVar > > > > > s; 
    vector< vector< vector < GRBVar > > > s2; 
    vector <vector <GRBVar > >  r; 
    vector <vector <vector <GRBVar > > > t; 

    vector <vector <vector <vector <GRBVar > > > > position_var; 
};

#endif
