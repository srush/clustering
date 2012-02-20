#include "speech_problem.h"
#include "speech_kmeans.h"
#include "subgrad.h"

int main() {

  SpeechProblemSet *speech_problems = 
    SpeechProblemSet::ReadFromFile("/tmp/out_pho",
                                   "/tmp/out_utt", 
                                   "/tmp/out_cent");

  SpeechKMeans *speech_kmeans = new SpeechKMeans(*speech_problems);

  speech_kmeans->InitializeCenters();
  speech_kmeans->Run();
};
