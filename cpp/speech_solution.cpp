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


void SpeechSolution::ToProtobuf(speech::SpeechSolution &solution, 
                                const SpeechProblemSet &speech_problem) const {
  for (int u = 0; u < cluster_set_.problems_size(); ++u) {
    const SpeechAlignment &align = alignment(u);
    speech::UtteranceAlignment *align_buf= solution.add_alignments();
    align.ToProtobuf(*align_buf);
  }
  for (int p = 0; p < cluster_set_.num_types(); ++p) {
    speech::Vector *vector = solution.add_phoneme_centers();

    if (!use_special_) {
      const Center &center = speech_problem.center(TypeToHidden(p));
      const DataPoint &center_point = center.point();
      for (int feature = 0; feature < speech_problem.num_features(); ++feature) {
        vector->add_dim(center_point[feature]);
      }
    } else {
      DataPoint center_point = type_to_special_[p];
      for (int feature = 0; feature < speech_problem.num_features(); ++feature) {
        vector->add_dim(center_point[feature]);
      }
    }
  }
}
