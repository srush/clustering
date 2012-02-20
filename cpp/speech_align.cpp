
#include "speech_align.h"


AlignmentSolution::AlignmentSolution(const AlignmentProblem &problem,
                                     vector<int> change_points) {  
  for (int change_index = 0; 
       change_index < change_points.size() - 1; 
       ++change_index) {
    phone_start.push_back(change_points[change_index]);
    phone_end.push_back(change_points[change_index + 1]);
  }    
  assert(phone_start.size() == utterance.phones_size());
  assert(phone_end.size() == utterance.phones_size());
}

void AlignmentSolution::Print() {
  const Utterance &utterance = problem_.utterance();
  for (int i = 0; utterance.phones_size(); ++i) {
    cout << utterance.phone(i).phoneme() << " " << 
      phone_start_[i] << " " << phone_end_[i];
  }
}


