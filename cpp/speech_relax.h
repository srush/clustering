#ifndef SPEECH_RELAX_H
#define SPEECH_RELAX_H

#include "semimarkov.h"

class SpeechRelax : AlignmentSolver {
  SpeechRelax() {}

  AlignmentSolution SolveAlignment(const AlignmentProblem &problem) {
    int num_states = problem.utterance().phones_size();
    int num_timesteps = problem.time_sequence().sequence_length();
    SemiMarkov semimarkov(num_states, num_timesteps);

  }
};

#endif
