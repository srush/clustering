#include "speech_kmeans.h"
#include "semimarkov.h"
#include "viterbi.h"
#include "kmlocal-1.7.2/src/KMlocal.h"

#define CENTERS_RATIO 1

SpeechKMeans::SpeechKMeans(const SpeechProblemSet &problem_set): 
  problems_(problem_set), cluster_problems_(problem_set.MakeClusterSet()),
  path_(cluster_problems_.problems_size()),
  mode_centers_(cluster_problems_.problems_size()),
  type_centers_(cluster_problems_.problems_size()),
  use_medians_(false),
  use_gmm_(false),
  use_unsupervised_(false),
  use_isotropic_(false),
  unsup_initialized_ (false) { 
  num_types_ = cluster_problems_.num_types();
  num_features_ = problem_set.num_features();
  distances_.resize(cluster_problems_.problems_size());
  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    distances_[u] = problems_.MakeDistances(u);
  }
}

double SpeechKMeans::Run(int rounds) {
  vector<vector<vector<DataPoint> > > phoneme_states(num_types_);
  int num_modes = cluster_problems_.num_modes();
  vector<vector<DataPoint> > center_estimators(num_modes);  
  vector<vector<double> > center_counts(num_modes);  

  if (use_unsupervised_ && !unsup_initialized_) {
    vector<double> weights;
    vector<DataPoint> points;
    for (int utterance_index = 0; 
         utterance_index < problems_.utterance_size(); 
         ++utterance_index) {
      ClusterSegmentsExpectation(utterance_index, &points, &weights); 
    }
    ClusterSegmentsMaximization(&points, &weights);
    unsup_initialized_ = true;
  }

  double round_score = 0.0;
  for (int round = 0; round < rounds; ++round) {
    if (use_gmm_) {
      for (int mode = 0; mode < num_modes; ++mode) {
        center_estimators[mode].resize(num_types_);
        center_counts[mode].resize(num_types_);
        for (int type = 0; type < num_types_; ++type) {
          center_estimators[mode][type].resize(problems_.num_features(), 0.0);
          for (int feat = 0; feat < problems_.num_features(); ++feat) {
            center_estimators[mode][type][feat] = 0.0;
          }
          center_counts[mode][type] = 0.0;
        }
      }
    }

    round_score = 0.0;
    int total_correctness = 0;
    for (int utterance_index = 0; 
         utterance_index < problems_.utterance_size(); 
         ++utterance_index) {
      int correctness; 
      if (use_unsupervised_) {
        round_score += UnsupExpectation(utterance_index, &correctness, &phoneme_states);
        total_correctness += correctness;
      } else if (!use_gmm_) {
        round_score += Expectation(utterance_index, &correctness, &phoneme_states);
        total_correctness += correctness;
      } else {
        round_score += GMMExpectation(utterance_index, center_estimators, center_counts);
        Expectation(utterance_index, &correctness, &phoneme_states);
        total_correctness += correctness;
      }
    }
    if (use_unsupervised_) {
      UnsupMaximization(phoneme_states);
    } else if (!use_gmm_) {
      Maximization(phoneme_states);
    } else {
      GMMMaximization(center_estimators, center_counts);
    }
    cerr << "SCORE: Round score: " << round << " " <<  round_score << " " << total_correctness << endl;
  }
  return round_score;
}

double SpeechKMeans::Expectation(int utterance_index,
                                 int *correctness,
                                 vector<vector<vector<DataPoint> > > *sets) {
  int u = utterance_index;
  const ClusterProblem &cluster_problem = 
    cluster_problems_.problem(utterance_index);
  const Utterance &utterance = problems_.utterance(utterance_index);

  // Duplicate each state for every mode.
  Viterbi viterbi(cluster_problem.num_states,
                  cluster_problem.num_steps, 
                  cluster_problems_.num_modes(), 
                  1);
  viterbi.Initialize();


   // Set the weights based on the current centers.
  clock_t start = clock();

  for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
    for (int i = 0; i < cluster_problem.num_states; ++i) {
      
      // const Gaussian &gaussian = gmms_[mode][cluster_problem.MapState(i)];
      // for (int s = 0; s < cluster_problem.num_steps; ++s) {
      //   double score = -gaussian.LogLikelihood(utterance.sequence(s));
      //   viterbi.set_transition_score(s, i, mode, score);
      // }
      const DataPoint &center = centers_[mode][cluster_problem.MapState(i)];
      for (int s = 0; s < cluster_problem.num_steps; ++s) {
        double score = 0.0;
        for (int j = 0; j < utterance.sequence_points(s); ++j) {
          score += dist(center, utterance.sequence(s, j));
        }
        viterbi.set_transition_score(s, i, mode, score);
      }
    }
  }
  cerr << "TIME: score setting " << clock() - start << endl;  
  
  // Run viterbi algorithm.
  viterbi.ForwardScores();
  double score = viterbi.GetBestPath(&path_[u], &mode_centers_[u]);
  (*correctness) = utterance.ScoreAlignment(path_[u]);
  cerr << "SCORE: Correctness: " << *correctness << endl; 

  // Make the cluster sets.
  sets->resize(cluster_problems_.num_modes());
  for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
    (*sets)[mode].resize(num_types_);
  }
  problems_.AlignmentClusterSet(utterance_index, path_[u], mode_centers_[u], sets);

  // collapse to modes. 
  assert(path_[u].size() - 1 == (uint)cluster_problem.num_states);
  cerr << endl;
  return score;
}

void SpeechKMeans::Maximization(const vector<vector<vector<DataPoint> > > &sets) {
  centers_.clear();
  gmms_.clear();
  int num_modes = cluster_problems_.num_modes();
  gmms_.resize(num_modes);
  centers_.resize(num_modes);
  if (!use_medians_) {
    for (int mode = 0; mode < num_modes; ++mode) {
      for (int p = 0; p < num_types_; ++p) {
        Gaussian total(num_features_);
        total.set_isotropic(use_isotropic_);
        total.EstimateFromData(sets[mode][p]);
        // for (uint i = 0; i < sets[mode][p].size(); ++i) {
        //   total += sets[mode][p][i];
        // }
        // total = total / float(sets[mode][p].size());
        gmms_[mode].push_back(total);
      }
    }
  } else {
    for (int mode = 0; mode < num_modes; ++mode) {
      for (int p = 0; p < num_types_; ++p) {
        DataPoint total(num_features_);
        for (uint i = 0; i < sets[mode][p].size(); ++i) {
          total += sets[mode][p][i];
        }
        total = total / float(sets[mode][p].size());
        double best = 1e10;
        // DataPoint median(num_features_);
        // for (uint i = 0; i < sets[mode][p].size(); ++i) {
        //   double trial = dist(sets[mode][p][i], total);
        //   if (trial < best) {
        //     best = trial;
        //     median = sets[mode][p][i];
        //   }
        // }
        DataPoint median(num_features_);
        for (int i = 0; i < problems_.centers_size(); ++i) {
          double trial = dist(problems_.center(i).point(), total);
          if (trial < best) {
            best = trial;
            median = problems_.center(i).point();
          }
        }
        centers_[mode].push_back(median);
      }
    }
  }
}

/*
double SpeechKMeans::Expectation(int utterance_index,
                                 int *correctness,
                                 vector<vector<DataPoint> > *sets) {
  int u = utterance_index;
  const ClusterProblem &cluster_problem = 
    cluster_problems_.problem(utterance_index);
  const Utterance &utterance = problems_.utterance(utterance_index);

  int width_limit = cluster_problems_.width_limit();
  SemiMarkov semimarkov(cluster_problem.num_states, 
                        cluster_problem.num_steps,
                        width_limit);
  semimarkov.Initialize();


   // Set the weights based on the current centers.
  clock_t start = clock();
  int count = 0;

  for (int s = 0; s < cluster_problem.num_steps; ++s) {
    for (int o = 0; o < width_limit; ++o) {
      if (s + o >= cluster_problem.num_steps) {
        continue;
      }
      if (distances_[u]->is_pruned(s, o)) {
        semimarkov.set_pruned(s, s + o);
      }
    }
  }


  for (int i = 0; i < cluster_problem.num_states; ++i) {
    const DataPoint &center = centers_[cluster_problem.MapState(i)];
    vector<double> dists;
    for (int s = 0; s < cluster_problem.num_steps; ++s) {
      dists.push_back(dist(center,
                           utterance.sequence(s)));
    }
    for (int s = 0; s < cluster_problem.num_steps; ++s) {
      double total = 0.0;
      for (int o = 0; o < width_limit; ++o) {
        total += dists[s + o];
        if (s + o >= cluster_problem.num_steps) {
          break;
        }
        if (!distances_[u]->is_pruned(s, o)) { 
          semimarkov.set_score(s, s + o, i, total);
        }
        ++count;
      }
    }
  }
  cerr << "score setting " << clock() - start << " " << count << endl;  
  
  // Run semimarkov algorithm.
  //vector<int> path;
  semimarkov.ViterbiForward();
  double score = semimarkov.GetBestPath(&path_[u]);
  (*correctness) = utterance.ScoreAlignment(path_[u]);
  cerr << "Correctness: " << *correctness << endl; 

  // Make the cluster sets.
  problems_.AlignmentClusterSet(utterance_index, path_[u], sets);

  assert(path_[u].size() - 1 == (uint)cluster_problem.num_states);
  cerr << endl;
  return score;
}
*/


void SpeechKMeans::ClusterSegmentsExpectation(int utterance_index,
                                              vector<DataPoint> *points,
                                              vector<double> *weights) {
  const ClusterProblem &cluster_problem = 
    cluster_problems_.problem(utterance_index);

  // Duplicate each state for every mode.
  Viterbi viterbi(cluster_problem.num_states,
                  cluster_problem.num_steps, 
                  cluster_problem.num_hidden(0), 1);
  viterbi.Initialize();


   // Set the weights based on the current centers.
  clock_t start = clock();

  for (int s = 0; s < cluster_problem.num_steps; ++s) {
    for (int c = 0; c < cluster_problems_.num_hidden(0); ++c) {
      double score = distances_[utterance_index]->get_distance(s, c);
      viterbi.set_score(s, c, score);
    }
  }
  cerr << "TIME: score setting " << clock() - start << endl;  
  
  // Run semimarkov algorithm.
  viterbi.ForwardScores();
  vector<int> path;
  vector<int> centers;
  viterbi.GetBestPath(&path, &centers);
  //double weight = 0.0;
  int state = 0;
  for (int s = 0; s < cluster_problem.num_steps; ++s) {
    points->push_back(problems_.center(centers[state]).point());
    // weight += distances_[utterance_index]->get_distance(s, 
    //                                                     centers[state]);
    if (s >= path[state + 1]) {
      //weights->push_back(weight);
      
      points->push_back(problems_.center(centers[state]).point());
      ++state;
      //weight = 0.0;
    }
  }
}

void SpeechKMeans::ClusterSegmentsMaximization(vector<DataPoint> *points,
                                               vector<double> *weights) {
  vector<vector<DataPoint> > types(1);
  types[0] = 
    WeightedKMeans(*points, *weights, 
                   min(cluster_problems_.num_types() / CENTERS_RATIO, (int)points->size()));
  types[0].resize(cluster_problems_.num_types());
  int dim = problems_.num_features(); // dimension
  for (int extra = points->size(); extra < cluster_problems_.num_types(); ++extra) {
    types[0][extra].resize(dim);
    for (int i = 0; i < dim; ++i) {
      types[0][extra][i] = 0.0;
    }
  }
  SetCenters(types);
  cerr << "Done maximization" << endl;
}

vector<DataPoint> SpeechKMeans::WeightedKMeans(vector<DataPoint> &points, 
                                               vector<double> &weights, int k) {

  
  KMterm term(100, 0, 0, 0, // run for 100 stages
              0.10, 0.10, 3, // other typical parameter values
              0.50, 10, 0.95);
  int dim = problems_.num_features(); // dimension
  int nPts = points.size(); // number of data points
  KMdata dataPts(dim, nPts); // allocate data storage
  for (int p = 0; p < nPts; ++p) {
    dataPts[p] = new double[dim];
    for (int i = 0; i < dim; ++i) {
      dataPts[p][i] = points[p][i];
    }
  }
  //kmUniformPts(dataPts.getPts(), nPts, dim); // generate random points
  dataPts.buildKcTree(); // build filtering structure
  KMfilterCenters ctrs(k, dataPts); // allocate centers
  // run the algorithm
  //KMlocalLloyds kmAlg(ctrs, term); // repeated Lloyd's
  // KMlocalSwap kmAlg(ctrs, term); // Swap heuristic
  // KMlocalEZ_Hybrid kmAlg(ctrs, term); // EZ-Hybrid heuristic
  KMlocalHybrid kmAlg(ctrs, term); // Hybrid heuristic
  ctrs = kmAlg.execute(); // execute
  // print number of stages
  cout << "Number of stages: " << kmAlg.getTotalStages() << "\n";
  // print average distortion
  cout << "Average distortion: " << ctrs.getDist(false)/nPts << "\n";
  ctrs.print(); // print final centers

  cerr << "copying points" << endl;
  vector<DataPoint> results(k); 
  for (int j = 0; j < k; ++j) {
    results[j].resize(dim);
    for (int i = 0; i < dim; ++i) {
      results[j][i] = ctrs.getCtrPts()[j][i];
    }
  }
  cerr << "done copying points" << endl;
  return results;
}


double SpeechKMeans::UnsupExpectation(int utterance_index,
                                      int *correctness,
                                      vector<vector<vector<DataPoint> > > *sets) {
  cerr << "Unsupervised maximization" << endl;
  int u = utterance_index;
  const ClusterProblem &cluster_problem = 
    cluster_problems_.problem(utterance_index);
  const Utterance &utterance = problems_.utterance(utterance_index);

  // Duplicate each state for every mode.
  Viterbi viterbi(cluster_problem.num_states,
                  cluster_problem.num_steps, 
                  cluster_problems_.num_types(), 
                  1);
  viterbi.Initialize();


   // Set the weights based on the current centers.
  clock_t start = clock();

  for (int type = 0; type < cluster_problems_.num_types(); ++type) {
    const DataPoint &center = centers_[0][type];
    for (int s = 0; s < cluster_problem.num_steps; ++s) {
      double score = dist(center,
                          utterance.sequence(s, 0));
      for (int i = 0; i < cluster_problem.num_states; ++i) {
        viterbi.set_transition_score(s, i, type, score);
      }
    }
  }
  cerr << "TIME: score setting " << clock() - start << endl;  
  
  // Run semimarkov algorithm.
  viterbi.ForwardScores();
  double score = viterbi.GetBestPath(&path_[u], &type_centers_[u]);
  (*correctness) = utterance.ScoreAlignment(path_[u]);
  cerr << "SCORE: Correctness: " << *correctness << endl; 

  // Make the cluster sets.
  sets->resize(cluster_problems_.num_modes());
  for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
    (*sets)[mode].resize(num_types_);
  }
  problems_.AlignmentClusterSetUnsup(utterance_index, path_[u], 
                                     type_centers_[u], sets);

  // collapse to modes. 
  cerr << endl;
  return score;
}

double SpeechKMeans::GMMExpectation(int utterance_index,
                                    vector<vector<DataPoint> > &estimators, 
                                    vector<vector<double> > &counts) {
  const ClusterProblem &cluster_problem = 
    cluster_problems_.problem(utterance_index);
  const Utterance &utterance = problems_.utterance(utterance_index);

  // Duplicate each state for every mode.
  Viterbi viterbi(cluster_problem.num_states,
                  cluster_problem.num_steps, cluster_problems_.num_modes(), 3);
  viterbi.Initialize();


   // Set the weights based on the current centers.
  clock_t start = clock();

  for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
    for (int i = 0; i < cluster_problem.num_states; ++i) {
      const DataPoint &center = centers_[mode][cluster_problem.MapState(i)];
      for (int s = 0; s < cluster_problem.num_steps; ++s) {
        double score = dist(center,
                            utterance.sequence(s, 0));
        viterbi.set_transition_score(s, i, mode, score);
      }
    }
  }
  cerr << "score setting " << clock() - start << endl;  
  
  // Run semimarkov algorithm.
  //vector<int> path;
  viterbi.set_use_sum();
  viterbi.ForwardScores();
  double score = viterbi.GetBestScore();
  viterbi.BackwardScores();
  vector<vector<vector<double> > > marginals;
  viterbi.Marginals(&marginals);
  for (int s = 0; s < cluster_problem.num_steps; ++s) {
    for (int i = 0; i < cluster_problem.num_states; ++i) {
      int type = cluster_problem.MapState(i);
      for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
        double p = marginals[s][i][mode];
        //cerr << p << endl;
        //assert(p <= 1.0 + 1e-4);
        //assert(p >= 0.0);
        
        if (p > 1.0) p = 1.0;
        //if (p > 1e-4) {
        estimators[mode][type] += p * utterance.sequence(s, 0);
        counts[mode][type] += p;
        //}
      }
    }
  }
  return score;
}


void SpeechKMeans::GMMMaximization(const vector<vector<DataPoint> > &estimators, 
                                   const vector<vector<double> > &counts) {  
  centers_.clear();
  int num_modes = cluster_problems_.num_modes();
  centers_.resize(num_modes);
  for (int mode = 0; mode < num_modes; ++mode) {
    for (int type = 0; type < cluster_problems_.num_types(); ++type) {
      centers_[mode].push_back(estimators[mode][type] / counts[mode][type]);
    }
  }
}

void SpeechKMeans::UnsupMaximization(const vector<vector<vector<DataPoint> > > &sets) {
  centers_.clear();
  int num_modes = cluster_problems_.num_modes();
  centers_.resize(num_modes);

  int dim = problems_.num_features(); // dimension
  DataPoint zero(dim);
  for (int i = 0; i < dim; ++i) {
    zero[i] = 0.0;
  }

  for (int mode = 0; mode < num_modes; ++mode) {
    for (int p = 0; p < num_types_ ; ++p) {
      if (p < num_types_ / CENTERS_RATIO) {
        DataPoint total(num_features_);
        
        for (uint i = 0; i < sets[mode][p].size(); ++i) {
          total += sets[mode][p][i];
        }
        total = total / float(sets[mode][p].size());
        centers_[mode].push_back(total);
      } else {
        centers_[mode].push_back(zero);
      }
    }
  }
}

void SpeechKMeans::InitializeCenters() {
  srand(13);
  centers_.clear();
  gmms_.clear();
  centers_.resize(cluster_problems_.num_modes());
  gmms_.resize(cluster_problems_.num_modes());
  for (int p = 0; p < num_types_; ++p) {
    for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
      DataPoint point(num_features_);
      for (int feat = 0; feat < num_features_; ++feat) {
        point[feat] = rand() / (float) RAND_MAX;
      }
      centers_[mode].push_back(point);
      Gaussian gaussian(point.size());
      gaussian.center = point;
      gaussian.set_isotropic(use_isotropic_);
      gmms_[mode].push_back(gaussian);
    }
  }
}

void SpeechKMeans::SetCenters(const vector<vector<DataPoint> > &centers) {
  centers_ = centers;
}

SpeechSolution *SpeechKMeans::MakeSolution() {
  const ClusterSet &cluster_set = problems_.MakeClusterSet();
  SpeechSolution *solution = new SpeechSolution(cluster_set);
  for (int p = 0; p < cluster_problems_.num_types(); ++p) {
    for (int mode = 0; mode < cluster_problems_.num_modes(); ++mode) {
      solution->set_type_to_special(p, mode, gmms_[mode][p].center);
    }
  }

  for (int u = 0; u < cluster_problems_.problems_size(); ++u) {
    const ClusterProblem &problem = cluster_problems_.problem(u);
    SpeechAlignment *align = solution->mutable_alignment(u);
    vector<int> *solution = align->mutable_alignment();
    vector<int> *modes = align->mutable_mode_align();
    modes->resize(problem.num_states + 1);
    solution->resize(problem.num_states + 1);
    if (!use_unsupervised_) {
      for (int i = 0; i < problem.num_states + 1; ++i) {
        (*solution)[i] = path_[u][i];
        if (i < problem.num_states) {
          (*modes)[i] = mode_centers_[u][i];
        } 
      }
    } else {
      for (int i = 0; i < problem.num_states + 1; ++i) {
        (*solution)[i] = path_[u][i];
        if (i < problem.num_states) {
          (*modes)[i] = type_centers_[u][i];
        } 
      }      
    }
  }
  return solution;
}
