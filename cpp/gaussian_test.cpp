#include "gtest/gtest.h"
#include "gaussian.h"

class GaussianTest : public testing::Test {
protected:  
  virtual void SetUp() {
    
  }  
};

TEST_F(GaussianTest, RunTest) {
  Gaussian gaussian(3);
  vector<DataPoint> points(3);
  points[0].resize(3);
  points[0][0] = 0.0;
  points[0][1] = 1.0;
  points[0][2] = -1.0;

  points[1].resize(3);
  points[1][0] = 1.0;
  points[1][1] = -1.0;
  points[1][2] = 0.0;

  points[2].resize(3);
  points[2][0] = -1.0;
  points[2][1] = 0.0;
  points[2][2] = 1.0;
  
  gaussian.EstimateFromData(points);
  
  DataPoint center_point(3);
  center_point[0] = 0.0;
  center_point[1] = 0.0;
  center_point[2] = 0.0;

  cout << exp(gaussian.LogLikelihood(center_point)) << endl; 

  center_point[0] = 1.0;
  center_point[1] = 1.0;
  center_point[2] = 1.0;
  cout << exp(gaussian.LogLikelihood(center_point)) << endl; 
}

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
