#include <iostream>
#include <fstream>
#include <assert.h>

#include "distances.h"
#include "hidden_solver.h"
#include "speech.h"
#include "speech_problem.h"
#include "speech_problem.h"
#include "build/speech.pb.h"
using namespace std;


SpeechProblemSet::SpeechProblemSet(const vector<Utterance *> &utterances, 
                                   vector<Center> *centers) 
  : utterances_(utterances), centers_(centers), holder_(utterances_.size()) {
  vector<ClusterProblem *> cluster_problems;
  int num_hidden = 0;
  int num_types = 0;
  int width_limit = 0;

  for (uint u = 0; u < utterances_.size(); ++u) {
    holder_[u] = NULL;
    num_features_ = utterances_[u]->num_features();
    int num_steps = utterances_[u]->sequence_size();
    width_limit = utterances_[u]->max_phone_length();
    int num_states = utterances_[u]->phones_size();
    num_types = utterances_[u]->phoneme_set().phonemes_size();
  
    // For now all time steps are available.
    //num_hidden += utterances_[u]->sequence_size();
    vector<int> state_to_type(num_states);  

    // Map the states to their types.
    for (int i = 0; i < num_states; ++i) {
      state_to_type[i] = utterances_[u]->phones(i).id();
    }
    ClusterProblem *problem = new ClusterProblem(num_steps,
                                                  num_states,
                                                 state_to_type);
    
    problem->set_gold_segmentation(utterances_[u]->get_gold());
    cluster_problems.push_back(problem);
  }

  num_hidden = centers_->size();
  cluster_set_ = new ClusterSet(cluster_problems, 
                                num_hidden, 
                                num_types, 
                                0, //TODO, 
                                width_limit);
  for (uint u = 0; u < utterances_.size(); ++u) {
    cluster_problems[u]->cluster_set_ = cluster_set_;
  }

  // Allow all hidden to begin with.
  for (int p = 0; p < num_types; ++p) {
    for (int h = 0; h < num_hidden; ++h) {
      cluster_set_->add_type_hidden(p, h);
    }
  }

  CacheTypeOccurence();
}

void SpeechProblemSet::CacheTypeOccurence() {
  state_locations_.resize(num_types());
  for (int u = 0; u < cluster_set_->problems_size(); ++u) {
    const ClusterProblem &problem = cluster_set_->problem(u);
    for (int s = 0; s < problem.num_states; ++s) {
      int p = problem.MapState(s);
      state_locations_[p].push_back(StateLocation(u, s));
    }
  }
}

ThinDistanceHolder *SpeechProblemSet::MakeDistances(int problem) const {
  if (holder_[problem] == NULL) {
    vector<const DataPoint * > *time_steps = new vector<const DataPoint*>();
    for (int i = 0; i < utterances_[problem]->sequence_size(); ++i) {
      time_steps->push_back(&utterances_[problem]->sequence(i));
    }
    // holder_[problem] = new DistanceHolder(*centers_, 
    //                                       *time_steps, 
    //                                       utterances_[problem]->max_phone_length());
    holder_[problem] = new ThinDistanceHolder(*centers_, 
                                              *time_steps);
    holder_[problem]->Initialize();
    //holder_[problem]->ComputeDistances();
  }
  return holder_[problem];
}


BallHolder *SpeechProblemSet::MakeBalls(vector<int> epsilons) const {
  BallHolder *holder = new BallHolder(epsilons, *centers_);
  holder->ConstructRelations();
  //cerr << *holder << endl; 
  return holder;
}

const ClusterSet &SpeechProblemSet::MakeClusterSet() const {
  return *cluster_set_;
}

void SpeechProblemSet::AlignmentClusterSet(int problem,
                                           const vector<int> &alignment,  
                                           vector<vector<DataPoint> > *cluster_sets) const {
  cluster_sets->resize(cluster_set_->num_types());
  const ClusterProblem &cluster_problem = cluster_set_->problem(problem);
  for (uint i = 0; i < alignment.size() - 1; ++i) {
    int start = alignment[i];
    int end = alignment[i + 1];
    int type = cluster_problem.MapState(i);
    for (int j = start; j < end; ++j) {
      (*cluster_sets)[type].push_back(utterance(problem).sequence(j));
    }
  }
  assert(alignment.size() - 1 == (uint)cluster_problem.num_states);
}

void SpeechProblemSet::AlignmentGroupClusterSet(int problem,
                                                const vector<int> &alignment,  
                                                vector<vector<vector<DataPoint> > > *cluster_sets) const {
  cluster_sets->resize(cluster_set_->num_types());
  const ClusterProblem &cluster_problem = cluster_set_->problem(problem);
  for (uint i = 0; i < alignment.size() - 1; ++i) {
    int type = cluster_problem.MapState(i);    
    (*cluster_sets)[type].push_back(vector<DataPoint>());
    int last = (*cluster_sets)[type].size() - 1;
    int start = alignment[i];
    int end = alignment[i + 1];
    for (int j = start; j < end; ++j) {
      (*cluster_sets)[type][last].push_back(utterance(problem).sequence(j));
    }
  }
  assert(alignment.size() - 1 == (uint)cluster_problem.num_states);
}


double SpeechProblemSet::MaximizeCenters(const vector<vector<DataPoint> > &cluster_sets,
                                         vector<DataPoint> *centers) const {
  centers->clear();
  double score = 0.0;
  //assert(cluster_sets.size() == (uint)cluster_problem_->num_types);
  for (int p = 0; p < cluster_set_->num_types(); ++p) {
    DataPoint total(num_features());
    for (uint i = 0; i < cluster_sets[p].size(); ++i) {
      total += cluster_sets[p][i];
    }
    total = total / float(cluster_sets[p].size());
    centers->push_back(total);
    for (uint i = 0; i < cluster_sets[p].size(); ++i) {
      score += dist(cluster_sets[p][i], total);
    }
  }
  return score;
}

DataPoint SpeechProblemSet::Centroid(int problem, int start, int end) const {
  DataPoint query(num_features());
  for (int i = start; i <= end; ++i) {
    query += utterances_[problem]->sequence(i);
  }
  return query / float(end - start + 1);
}

double SpeechProblemSet::MaximizeMedians(const SpeechSolution &bad_speech_solution,
                                         vector<DataPoint> *centroids) const {
  double total = 0.0;
  centroids->clear();

  // For each type and state, find the centroid.
  for (int p = 0; p < num_types(); ++p) {
    int count = 0;
    DataPoint query(num_features());
    for (int i = 0; i < type_occurence_size(p); ++i) {
      StateLocation location = type_occurence(p, i);
      int s,e;
      bad_speech_solution.alignment(location.problem).StateAlign(location.state, &s, &e);
      for (int i = s; i <= e; ++i) {
        query += utterances_[location.problem]->sequence(i);
        ++count;
      }
    }
    DataPoint centroid = query / (float)count;
    DataPoint closest(num_features());
    double best = 100000;
    for (int i = 0; i < type_occurence_size(p); ++i) {
      StateLocation location = type_occurence(p, i);
      int s,e;
      bad_speech_solution.alignment(location.problem).StateAlign(location.state, &s, &e);
      for (int q = 0; q < centers_size(); ++q) {
        const DataPoint &possible_center = center(q).point();
        double trial = dist(possible_center, centroid) ;
        if (trial < best) {
          closest = possible_center;
          best = trial;
        }
      }
      // for (int i = s; i <= e; ++i) {
      //   double trial = dist(utterances_[location.problem]->sequence(i), centroid) ;
      //   if (trial < best) {
      //     closest = utterances_[location.problem]->sequence(i);
      //     best_hidden = i;
      //     best = trial;
      //   }
      // }
      // for () {

      // }
    }
    centroids->push_back(closest);
    for (int i = 0; i < type_occurence_size(p); ++i) {
      StateLocation location = type_occurence(p, i);
      int s,e;
      bad_speech_solution.alignment(location.problem).StateAlign(location.state, &s, &e);
      for (int i = s; i <= e; ++i) {
        total += dist(utterances_[location.problem]->sequence(i), closest);
      }
    }
  }

  return total;
}

double SpeechProblemSet::MaximizeMediansHidden(const SpeechSolution &bad_speech_solution,
                                               vector<DataPoint> *centroids) const {
  double total = 0.0;
  centroids->clear();

  // For each type and state, find the centroid.
  for (int p = 0; p < num_types(); ++p) {
    int count = 0;
    DataPoint query(num_features());
    for (int i = 0; i < type_occurence_size(p); ++i) {
      StateLocation location = type_occurence(p, i);
      int hidden = bad_speech_solution.alignment(location.problem).HiddenAlign(location.state);
      query += (*centers_)[hidden].point();
      ++count;
    }
    DataPoint centroid = query / (float)count;
    DataPoint closest(num_features());
    double best = 100000;
    for (int i = 0; i < type_occurence_size(p); ++i) {
      StateLocation location = type_occurence(p, i);
      int hidden = bad_speech_solution.alignment(location.problem).HiddenAlign(location.state);
      double trial = dist((*centers_)[hidden].point(),
                          centroid);
      if (trial < best) {
        closest = (*centers_)[hidden].point();
        best = trial;
      }
    }
    centroids->push_back(closest);

    for (int i = 0; i < type_occurence_size(p); ++i) {
      StateLocation location = type_occurence(p, i);
      int hidden = bad_speech_solution.alignment(location.problem).HiddenAlign(location.state);
      total += dist((*centers_)[hidden].point(), closest);
    }
  }

  return total;
}

SpeechSolution *SpeechProblemSet::ApproxMaximizeMedians(
                 const SpeechSolution &bad_speech_solution, 
                 const BallHolder &ball_holder,
                 double *score) const {
  *score = 0.0;
  SpeechSolution *proposed_primal = 
    new SpeechSolution(bad_speech_solution);

  // For each type and state, find the centroid
  vector<vector<DataPoint> > centroids(num_types());
  for (int p = 0; p < num_types(); ++p) {
    for (int i = 0; i < type_occurence_size(p); ++i) {
      StateLocation location = type_occurence(p, i);
      int s,e;
      bad_speech_solution.alignment(location.problem).StateAlign(location.state, &s, &e);
      
      centroids[p].push_back(Centroid(location.problem, s, e));
    }
  }

  for (int p = 0; p < num_types(); ++p) {
    DataPoint query(num_features());
    int query_points = 0;

    for (int i = 0; i < type_occurence_size(p); ++i) {
      StateLocation location = type_occurence(p, i);
      int s,e;
      bad_speech_solution.alignment(location.problem).StateAlign(location.state, &s, &e);

      // Construct the query point for the group.
      for (int i = s; i <= e; ++i) {
        query += utterances_[location.problem]->sequence(i);
        query_points++;
      }
    }
    query = query / float(query_points);

    double best_center_score = 1e20;
    int best_center = -1;
    for (uint t = 0; t < centers_->size(); ++t) {
      double trial =  dist((*centers_)[t].point(), query);
      if (trial < best_center_score) {
        best_center_score = trial;
        best_center = t;
      }
    }
    assert(best_center != -1);

    proposed_primal->set_type_to_hidden(p, best_center);
    
    
      int partition = ball_holder.partition_for_center(0, best_center);
      int nearby_size = 
        ball_holder.nearby_size(0, partition);
      
      for (int i = 0; i < type_occurence_size(p); ++i) {
        StateLocation location = type_occurence(p, i);
        int s,e;
        bad_speech_solution.alignment(location.problem).StateAlign(location.state, &s, &e);
        
        // Find closest point in ball to centroid of group.
        const DataPoint &centroid = centroids[p][i];
        int closest = -1;
        double closest_to_centroid_score = 1e20;
        
        for (int j = 0; j < nearby_size; ++j) {
          int q = ball_holder.nearby(0, partition, j);
          double d = dist((*centers_)[q].point(), centroid);
          if (d < closest_to_centroid_score) {
            closest = q;
            closest_to_centroid_score = d;
          }
        }
        assert(closest != -1);

        // Set the proposed primal value.
        SpeechAlignment *alignment = 
          proposed_primal->mutable_alignment(location.problem);
        alignment->set_hidden_alignment(location.state, closest);
        
        // Compute the cost to the closest;
        for (int point = s; point <= e; ++point) {
          *score += dist((*centers_)[closest].point(), 
                         utterances_[location.problem]->sequence(point));
        }
      }
    
  }
  return proposed_primal;
}


SpeechProblemSet *SpeechProblemSet::ReadFromFile(string phoneme_set_file,
                                                 string utterance_file,
                                                 string center_file) {
  // Read the phoneme type set.
  fstream input(phoneme_set_file.c_str(), ios::in | ios::binary);
  speech::PhonemeSet phoneme_set_buf;
  speech::UtteranceSet utterance_buf;
  speech::CenterSet center_buf;

  cerr << "Reading phoneme set." << endl;
  phoneme_set_buf.ParseFromIstream(&input);
  PhonemeSet *phoneme_set = PhonemeSet::FromProtobuf(phoneme_set_buf);
  cerr << "Phoneme set has " << phoneme_set->phonemes_size()
       << " phonemes."<< endl;

  // Read in the utterances. 
  vector<Utterance *> *utterances = new vector<Utterance *>();
  fstream input2(utterance_file.c_str(), ios::in | ios::binary);
  cerr << "Reading utterance." << endl;
  utterance_buf.ParseFromIstream(&input2);  
  for (int i = 0; i < utterance_buf.utterances_size(); ++i) {
    Utterance *utterance = Utterance::FromProtobuf(*phoneme_set, 
                                                   utterance_buf.utterances(i));
    utterances->push_back(utterance);
  }

  fstream input3(center_file.c_str(), ios::in | ios::binary);
  cerr << "Reading center set." << endl;
  center_buf.ParseFromIstream(&input3);  
  vector<Center> *centers = new vector<Center>(center_buf.centers_size());
  for (int i = 0; i < center_buf.centers_size(); ++i) {
    DataPoint *point = DataPointFromProtobuf(center_buf.centers(i));
    (*centers)[i].point_ = point;
    //}

  //vector<vector<bool> > *addresses = new vector<vector<bool> >();
  //for (int i = 0; i < center_buf.segments_size(); ++i) {
    for (int j = 0; j < center_buf.segments(i).address_size(); ++j) {
      (*centers)[i].address.push_back(center_buf.segments(i).address(j));
    }
  }

  SpeechProblemSet *problem_set = new SpeechProblemSet(*utterances, centers);
  input.close();
  input2.close();
  input3.close();

  return problem_set;
}
