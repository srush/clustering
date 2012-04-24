#include "speech_kmeans.h"
#include "speech_problem.h"
#include "speech_subgrad.h"
#include "subgrad.h"

#include "alignment_lp.h"

#include <google/gflags.h>
using namespace google;


DEFINE_string(problem_name, "", "The name of the problem to run.");

DEFINE_string(base_directory, "../data/problems/", "The name of the base directory. .");

DEFINE_string(algorithm, "lr", "The algorithm to run.");

DEFINE_string(output_name, "", "The name of the output.");

DEFINE_string(starting_model, "", "The name of a starting model.");

DEFINE_bool(pc_use_medians, false, "Restrict primal coordinate descent to use medians.");

DEFINE_bool(pc_use_gmm, false, "Run primal coordinate descent with GMMs.");

DEFINE_bool(pc_unsupervised, false, "Run primal coordinate descent unsupervised (no phonemes).");

static bool ValidateSet(const char *flagname, const string &value) {
  if (value != "") {
    return true;
  }
  printf("Invalid value for --%s: %s\n", flagname, value.c_str());
  return false;
}

static const bool port_dummy = 
  RegisterFlagValidator(&FLAGS_problem_name, &ValidateSet);

int main(int argc, char **argv) {
  ParseCommandLineFlags(&argc, &argv, true);
  
  SpeechProblemSet *speech_problems = 
    SpeechProblemSet::ReadFromFile(FLAGS_base_directory + FLAGS_problem_name + "/pho", 
                                   FLAGS_base_directory + FLAGS_problem_name + "/utt", 
                                   FLAGS_base_directory + FLAGS_problem_name + "/cent"); 
  const ClusterSet &cluster_problems = speech_problems->MakeClusterSet();
  
  // Primal coordinate descent.
  if (FLAGS_algorithm == "pc") {

    SpeechKMeans kmeans(*speech_problems);

    if (FLAGS_starting_model != "") {
      // Load in a model to start from. 
      SpeechSolution solution(cluster_problems);
      speech::SpeechSolution solution_buf;
      fstream input(FLAGS_starting_model.c_str(), ios::in | ios::binary);
      solution_buf.ParseFromIstream(&input);  
      solution.FromProtobuf(solution_buf);
      vector<vector <DataPoint> > centers;
      centers.resize(cluster_problems.num_modes());
      for (int mode = 0; mode < cluster_problems.num_modes(); ++mode) {
        centers[mode].resize(cluster_problems.num_types());
        for (int type = 0; type < cluster_problems.num_types(); ++type) {
          centers[mode][type] = 
            solution.TypeToSpecial(type, mode);
        }
      }
      kmeans.SetCenters(centers);
      input.close();
    } else { 
      kmeans.InitializeCenters();
    }

    kmeans.set_use_medians(FLAGS_pc_use_medians);
    for (int i = 0; i < 100; ++i) {

      if (FLAGS_pc_use_gmm) { 
        kmeans.set_use_gmm();
      }
      if (FLAGS_pc_unsupervised) { 
        kmeans.set_use_unsup();
      }
      kmeans.Run(10);
      
      stringstream buf;
      if (FLAGS_starting_model == "") {
        buf << "results/" << FLAGS_problem_name << "_pc_solution_"<< "_" 
            << FLAGS_output_name << "_" << i*10 + 10;
      } else {
        buf << "results/" << FLAGS_problem_name << "_pc_solution_starting_" 
            << FLAGS_output_name << "_"  << i*10 + 10;
      }

      fstream output(buf.str().c_str(), ios::out | ios::binary);
      SpeechSolution *solution = kmeans.MakeSolution();

      speech::SpeechSolution sol;
      solution->ToProtobuf(sol, *speech_problems);
      sol.SerializeToOstream(&output);
      output.close();
    }
  } else if (FLAGS_algorithm == "lr") {
    SpeechSubgradient *speech_subgrad = new SpeechSubgradient(*speech_problems);
    SpeechKMeans kmeans(*speech_problems);

    for (uint i = 0; i < 5000; ++i) {
      speech_subgrad->MPLPRound(i);
      cerr << "INFO: Round: " << i << endl;

      if (i % 10 == 5) {
        cerr << "INFO: Running kmeans" << endl;
        kmeans.SetCenters(speech_subgrad->centers());
        kmeans.set_use_medians(true);
        kmeans.Run(2);
        
        // Write out kmeans solution
        stringstream buf;
        buf << "results/" << FLAGS_problem_name << "_lr_solution_" << FLAGS_output_name << " " << i;
        fstream output(buf.str().c_str(), ios::out | ios::binary);
        SpeechSolution *solution = kmeans.MakeSolution();
        speech::SpeechSolution sol;
        solution->ToProtobuf(sol, *speech_problems);
        sol.SerializeToOstream(&output);
        output.close();
      }
    }
  } else if (FLAGS_algorithm == "lp") {
    AlignmentLP alignment_lp(*speech_problems);
    alignment_lp.ConstructLP();
  }
};
