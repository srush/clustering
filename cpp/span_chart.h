#ifndef SPAN_CHART_H
#define SPAN_CHART_H

#include <vector>
using namespace std;

// Chart specialized for start-end access. 
template <class Holder>
class SpanChart {
 public:
  // Create a span chart over total_size elements with end - start < width_limit. 
 SpanChart(int total_size, 
           int width_limit): 
  total_size_(total_size), 
    width_limit_(width_limit) {
      chart_.resize(total_size_);
      for (int s = 0; s < total_size_; ++s) {
        chart_[s].resize(width_limit_);
      }
    }

  const Holder &get(int s, int o) const {
    //assert(s < total_size_ && o < width_limit_);
    return chart_[s][o];
  }

  Holder get_const(int s, int o) const {
    //assert(s < total_size_ && o < width_limit_);
    return chart_[s][o];
  }

  // Get an element by the start index and end offset.
  Holder &get(int s, int o) {
    //assert(s < total_size_ && o < width_limit_);
    return chart_[s][o];
  }

  void set(int s, int o, Holder h) {
    //assert(s < total_size_ && o < width_limit_);
    chart_[s][o] = h;
  }

  int total_size_;

  // The maximum width of a span.
  int width_limit_;
 
 private:
  // The chart buffer.
  vector<vector <Holder> > chart_;
  
};

#endif
