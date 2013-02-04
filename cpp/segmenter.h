#ifndef SEGMENTER_H
#define SEGMENTER_H
#include "speech.h"
#include "cluster_problem.h"

class Segmenter {
 public:
  Segmenter() {}

  Utterance *Run(const ClusterProblem &problem, 
                 const Utterance &utterance) const;

  void MakeSequence(const Utterance &utterance, 
                    vector<int> order,  
                    vector<vector<const DataPoint *> > *seg) const;

};

#endif
