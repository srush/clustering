#include "speech_kmeans.h"
#include "speech_problem.h"
#include "speech_subgrad.h"
#include "subgrad.h"


int main() {
  SpeechProblemSet *speech_problems = 
    SpeechProblemSet::ReadFromFile("/tmp/out_pho", 
                                   "/tmp/out_utt", 
                                   "/tmp/out_cent"); 
  SpeechSubgradient *speech_subgrad = new SpeechSubgradient(*speech_problems);

  //double epsilons[3] = {5.0, 1.0, 0.0001};
  //int rounds = 10;
  vector<vector<int> > ball_epsilons;
  vector<int> ball_ep;
  // div = 1;

  //forint i = 0; i < rounds; i += 1) {
  //ball_ep.insert(ball_ep.begin(), i);
  //ball_epsilons.push_back(ball_ep);
    //div *= 2;
  //}
  
  // ball_epsilons.push_back(ball_ep);
  ball_ep.push_back(8);
  BallHolder *holder = speech_problems->MakeBalls(ball_ep);
  speech_subgrad->set_ball_holder(holder);
  
  // 
  //for (uint i = 0; i < rounds; ++i) {
  SubgradSolver solver(speech_subgrad);
    
  ExponentialWaitRate rate(25.0);
  solver.set_rate(&rate);
  solver.set_iterations(1000);
  solver.Run();
  

  // K-Means.
  // SpeechKMeans kmeans(*speech_problems);
  // kmeans.SetCenters(speech_subgrad->centers());
  // kmeans.Run();
};
