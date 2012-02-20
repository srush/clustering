#include "cluster_subgrad.h"
#include <sstream>
using namespace std;
// DataPoint *ClusterSubgrad::ToVector() const {
//   DataPoint *vec = 
//     new DataPoint(structure_.size());
//   for (uint i = 0; i < structure_.size(); ++i) {
//     (*vec)[i] = structure_[i];
//   }
//   return vec;
// }

ClusterSubgrad *ClusterSubgrad::FromVector(const ClusterSet &cs,
                                           int num_balls,
                                           const DataPoint &vector) {
  ClusterSubgrad *subgrad = new ClusterSubgrad(cs, num_balls);
  for (DataPoint::const_iterator pos = vector.begin(); 
       pos != vector.end();
       ++pos) { 
    int ind = pos.index();
    double score = *pos;
    subgrad->structure_[ind] = score;
  }
  return subgrad;
}

void ClusterSubgrad::Align(const ClusterSubgrad &s1, 
                           const ClusterSubgrad &s2) {
  for (int b = 0; b < s1.num_balls_; ++b) {
    vector<vector<string> > clusters(s1.cs_.num_types());
    for (int u = 0; u < s1.cs_.problems_size(); ++u) {
      const ClusterProblem &problem = s1.cs_.problem(u);
      for (int i = 0; i < s1.cs_.max_states(); ++i) {
        int q1 = -1, q2 = -1; 
        for (int q = 0; q < s1.cs_.num_hidden(); ++q) {
          ClusterIndex cluster_index(b, u, i, q);
          if (s1.score(cluster_index) != 0) {
            q1 = q;
          }
          if (s2.score(cluster_index) != 0) {
            q2 = q;
          }
        }
        if (q1 != -1) { 
          int type = problem.MapState(i);
          stringstream buf; 
          buf << u << " " << i << " " << q1 << " " << q2;
          clusters[type].push_back(buf.str());
        }
        // if (q1 != q2) { 
        //   cerr << u << " " << i << " "<<  problem.MapState(i)
        //        << " " << q1 << " " << q2 << endl;
        // }
      }
    }
    for (int t = 0; t < s1.cs_.num_types(); ++t) {
      for (uint i = 0; i < clusters[t].size(); ++i) {
        cerr << t << " " << clusters[t][i] << endl;
      }
    }
  }
}

std::ostream &operator<<(std::ostream &os, const ClusterSubgrad &subgrad) {
  cerr << "Ball Problem State Hidden" << endl; 
  for (int b = 0; b < subgrad.num_balls_; ++b) {
    for (int u = 0; u < subgrad.cs_.problems_size(); ++u) {
      for (int i = 0; i < subgrad.cs_.max_states(); ++i) {
        for (int q = 0; q < subgrad.cs_.num_hidden(); ++q) {
          ClusterIndex cluster_index(b, u, i, q);
          int index = subgrad.to_index(cluster_index);
          if (subgrad.structure_[index] != 0) {
            os << b << " " << u << " " <<  i << " " << q << " " 
               << subgrad.structure_[index] << endl;
          }
        }
      }
    }
  }
  return os;
}

