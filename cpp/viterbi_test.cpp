#include "gtest/gtest.h"
#include "viterbi.h"
#include "math.h"
#define INF 1e10 

class ViterbiTest : public testing::Test {
protected:  
  virtual void SetUp() {

  }

  void CheckValidMinMarginals(double score, Viterbi *viterbi) {
    vector<vector<double> > min_marginals;
    viterbi->MinMarginals(&min_marginals);
    int N = viterbi->num_states();
    int O = viterbi->num_centers();
    for (int n = 0; n < N; ++n) {
      for (int o = 0; o < O; ++o) {
        if (n != o) {
          EXPECT_GT(min_marginals[n][o], score);
        } else {
          EXPECT_GE(min_marginals[n][o], score);
        }
      }
    }
  }

  void BuildScoring(Viterbi *viterbi) {
    int M = viterbi->num_timesteps();
    int N = viterbi->num_states();
    int O = viterbi->num_centers();
    for (int m = 0; m < M; m++) { 
      for (int n = 0; n < N; n++) {
        for (int o = 0; o < O; o++) {
          if (m + 1 == (n + 1) * 5) {
            viterbi->set_transition_score(m, n, o, -10);
            EXPECT_EQ(viterbi->transition_score(m, n, o), -10);
          } else {
            viterbi->set_transition_score(m, n, o, 0);
          }
          if (o == n) {
            viterbi->set_lambda(n, o, -5);
          } else {
            viterbi->set_lambda(n, o, 0.0);
          }
        }
      }
    }
  }

  void BuildScoringLog(Viterbi *viterbi) {
    int M = viterbi->num_timesteps();
    int N = viterbi->num_states();
    int O = viterbi->num_centers();
    for (int m = 0; m < M; m++) { 
      for (int n = 0; n < N; n++) {
        for (int o = 0; o < O; o++) {
          if (m + 1 == (n + 1) * 5) {
            viterbi->set_transition_score(m, n, o, -log(0.5));
            EXPECT_EQ(viterbi->transition_score(m, n, o), -log(0.5));
          } else {
            viterbi->set_transition_score(m, n, o, INF);
          }
          if (o == n) {
            viterbi->set_lambda(n, o, -log(0.4));
          } else {
            viterbi->set_lambda(n, o, INF);
          }
        }
      }
    }
  }

};

TEST_F(ViterbiTest, RunLogAdd) {
  // EXPECT_EQ(exp(-LogAdd(-log(0.1), -log(0.5))), 0.6);
  // EXPECT_EQ(exp(-LogAdd(-log(0.001), -log(0.000000000001))), 0.001);
  // EXPECT_EQ(exp(-LogAdd(-log(0.1), -log(0.5))), 0.6);
}

TEST_F(ViterbiTest, RunTest) {
  Viterbi *viterbi = new Viterbi(4, 20, 5, 1);
  viterbi->Initialize();
  BuildScoring(viterbi);

  vector<int> path, centers; 
  viterbi->ForwardScores();
  double score = viterbi->GetBestPath(&path, &centers);
  EXPECT_EQ(score, -viterbi->num_states() * 10 + -viterbi->num_states() * 5);
  for (int i = 0; i < viterbi->num_states(); ++i) {
    EXPECT_EQ(path[i], i * 5);
    EXPECT_EQ(centers[i], i);
  }
  viterbi->BackwardScores();
  vector<vector<double> > min_marginals;
  viterbi->MinMarginals(&min_marginals);

  CheckValidMinMarginals(score, viterbi);
  delete viterbi;
}


TEST_F(ViterbiTest, RunTestLimit) {
  Viterbi *viterbi = new Viterbi(4, 40, 5, 8);
  viterbi->Initialize();
  BuildScoring(viterbi);

  vector<int> path, centers; 
  viterbi->ForwardScores();
  double score = viterbi->GetBestPath(&path, &centers);
  EXPECT_EQ(score, -20 + -viterbi->num_states() * 5);
  for (int i = 0; i < viterbi->num_states(); ++i) {
    EXPECT_EQ(centers[i], i);
  }
  viterbi->BackwardScores();
  CheckValidMinMarginals(score, viterbi);
}

TEST_F(ViterbiTest, RunTestSum) {
  Viterbi *viterbi = new Viterbi(4, 40, 5, 8);
  viterbi->Initialize();
  viterbi->set_use_sum();
  BuildScoringLog(viterbi);

  vector<int> path, centers; 
  viterbi->ForwardScores();
  viterbi->BackwardScores();
  vector<vector<vector<double> > > marginals;
  viterbi->Marginals(&marginals);
}


int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
