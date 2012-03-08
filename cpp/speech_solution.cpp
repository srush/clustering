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

void SpeechSolution::FromProtobuf(const speech::SpeechSolution &solution) {
  for (int p = 0; p < solution.phoneme_multi_centers_size(); ++p) {
    const speech::MultiCenter &multi = solution.phoneme_multi_centers(p);
    for (int mode = 0; mode < multi.centers_size(); ++mode) {
      DataPoint center_point = type_to_special_[mode][p];
      const speech::Vector &vector = multi.centers(mode);
      DataPoint *point = DataPointFromProtobuf(vector);
      type_to_special_[mode][p] = *point;
    }
  }
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
    speech::MultiCenter *multi = solution.add_phoneme_multi_centers();

    if (!use_special_) {
      // Old style. 
      const Center &center = speech_problem.center(TypeToHidden(p, 0));
      const DataPoint &center_point = center.point();
      for (int feature = 0; feature < speech_problem.num_features(); ++feature) {
        vector->add_dim(center_point[feature]);
      }
      // New style.
      for (int mode = 0; mode < cluster_set_.num_modes(); ++mode) {
        speech::Vector *vector = multi->add_centers();
        const Center &center = speech_problem.center(TypeToHidden(p, mode));
        const DataPoint &center_point = center.point();
        for (int feature = 0; feature < speech_problem.num_features(); ++feature) {
          vector->add_dim(center_point[feature]);
        }
      }
    } else {
      DataPoint center_point = type_to_special_[0][p];
      for (int feature = 0; feature < speech_problem.num_features(); ++feature) {
        vector->add_dim(center_point[feature]);
      }
      // New style.
      for (int mode = 0; mode < cluster_set_.num_modes(); ++mode) {
        DataPoint center_point = type_to_special_[mode][p];
        speech::Vector *vector = multi->add_centers();
        for (int feature = 0; feature < speech_problem.num_features(); ++feature) {
          vector->add_dim(center_point[feature]);
        }
      }

    }
  }
}
