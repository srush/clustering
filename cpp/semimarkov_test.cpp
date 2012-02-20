#include "gtest/gtest.h"
#include "semimarkov.h"

class SemiMarkovTest : public testing::Test {
protected:  
  virtual void SetUp() {
    width_limit_ = 10;
    semi_markov_ = new SemiMarkov(4, 20, width_limit_);
    semi_markov_->Initialize();
    BuildScoring();
  }

  void BuildScoring() {
    int m = semi_markov_->num_states();
    int n = semi_markov_->num_timesteps();
    for (int i = 0; i < m; i++) { 
      for (int s = 0; s < n; s++) {
        for (int e = s; e < n; e++) {
          if (e - s >= width_limit_) continue;
          if (e - s == 4) {
            semi_markov_->set_score(s, e, i, -10);
            assert(semi_markov_->score(s, e, i) == -10);
          } else {
            semi_markov_->set_score(s, e, i, 0);
            assert(semi_markov_->score(s, e, i) == 0);
          }
        }
      }
    }
  }
  int width_limit_;
  SemiMarkov *semi_markov_;
};

TEST_F(SemiMarkovTest, RunTest) {
  vector<int> path; 
  semi_markov_->ViterbiForward();
  double score = semi_markov_->GetBestPath(&path);
  EXPECT_EQ(score, -semi_markov_->num_states() * 10);
  for (int i = 0; i < semi_markov_->num_states(); ++i) {
    EXPECT_EQ(path[i], (i) * 5);
  }
  
  vector<int> back_path; 
  semi_markov_->ViterbiBackward();
  double back_score = semi_markov_->GetBestBackPath(&back_path);
  EXPECT_EQ(back_score, score);
  EXPECT_EQ(back_path.size(), path.size());
  for (uint i = 0; i < path.size(); ++i) {
    cerr << path[i] << " " << back_path[i] << endl;
    EXPECT_EQ(path[i], back_path[i]);
  }

  // Check that the marginals add up.
  for (uint i = 0; i < path.size(); ++i) {
    cerr << i << " " << path[i] << endl;
    cerr << " " << semi_markov_->GetForwardScore(i, path[i]);
    cerr << " " << semi_markov_->GetBackwardScore(i, path[i]) << endl;
    EXPECT_EQ(score, 
              semi_markov_->GetForwardScore(i, path[i]) + 
              semi_markov_->GetBackwardScore(i, path[i]));
  }

  for (int i = 0; i < semi_markov_->num_states(); ++i) {
    for (int t = 0; t < semi_markov_->num_timesteps(); ++t) {
      EXPECT_GE(semi_markov_->GetForwardScore(i, t) + 
                semi_markov_->GetBackwardScore(i, t), 
                score);
    }
  }
}

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
