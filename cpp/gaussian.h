#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "distances.h"
#include <vector>

typedef boost::numeric::ublas::mapped_vector<double> DataPoint;
// Parameters of diagonal multivariate gaussian
class Gaussian {
 public:
 Gaussian(int dims): 
  center(dims),
  dims_(dims), 
    inverse_variances(dims),
    isotropic_(false)
    {
      for (int d = 0; d < dims_; ++d) {
        center[d] = 0.0;
        inverse_variances(d) = 1.0;
      }
      log_determinant = 0.0;
    }
  
  void set_isotropic(bool isotropic) {
    isotropic_ = isotropic;
  }

  void EstimateFromData(const vector<DataPoint> &points);

  double LogLikelihood(const DataPoint &point) const;

  DataPoint center;
  double log_determinant;

 private:
  int dims_;
  DataPoint inverse_variances;
  
  bool isotropic_;
};

#endif
