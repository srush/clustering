#include "speech_solution.h"


SpeechAlignment::SpeechAlignment(const SpeechAlignment &alignment) {
  problem_ = alignment.problem_;
  alignment_ = alignment.alignment_;
  hidden_align_ = alignment.hidden_align_;
}


SpeechSolution::SpeechSolution(const SpeechSolution &solution) :
  cluster_set_(solution.cluster_set_)
{
  speech_alignments_ = solution.speech_alignments_;
  type_to_hidden_ = solution.type_to_hidden_;
  
}
