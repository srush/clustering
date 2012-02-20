#include "gtest/gtest.h"
#include "distances.h"

class DistanceTest : public testing::Test {
protected:  
  virtual void SetUp() {

    DataPoint *v = new DataPoint(1, 1);
    (*v)(0) = 0;
    centers_.push_back(*v);
    timesteps_.push_back(*v);

    v = new DataPoint(1, 1);
    (*v)(0) = 10;
    centers_.push_back(*v);
    timesteps_.push_back(*v);

    v = new DataPoint(1, 1);
    (*v)(0) = 5;
    centers_.push_back(*v);
    timesteps_.push_back(*v);

    v = new DataPoint(1, 1);
    (*v)(0) = 20;
    centers_.push_back(*v);
    timesteps_.push_back(*v);

    distance_holder_ = new DistanceHolder(centers_, timesteps_, 4);
  }  
  DistanceHolder *distance_holder_;
  vector<Center> centers_;
  vector<DataPoint> timesteps_;
};

TEST_F(DistanceTest, RunTest) {
  distance_holder_->Initialize();
  distance_holder_->ComputeDistances();
  for (uint i = 0; i < centers_.size(); ++i) {
    EXPECT_EQ(distance_holder_->get_distance(i, 0, i),
              0.0);
  }

  double a = distance_holder_->get_distance(0, 1, 0);
  double b = distance_holder_->get_distance(1, 0, 0);
  EXPECT_EQ(a, b);

  a = distance_holder_->get_distance(0, 2, 0);
  b = distance_holder_->get_distance(1, 1, 0);
  EXPECT_EQ(a, b);

  a = distance_holder_->get_distance(0, 2, 0);
  b = 5* 5 + 10 * 10;
  EXPECT_EQ(a, b);
}

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
