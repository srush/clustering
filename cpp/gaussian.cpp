#include "gaussian.h"
#include <vector>
using namespace std;

void Gaussian::EstimateFromData(const vector<DataPoint> &points) {
  DataPoint total(dims_);
  DataPoint var_total(dims_);
  for (uint i = 0; i < points.size(); ++i ) {
    total += points[i];
  }
  center = total / (float) points.size();
  for (uint i = 0; i < points.size(); ++i ) {
    var_total += element_prod((points[i] - center), (points[i] - center));
  }

  log_determinant = 0.0;
  if (!isotropic_ && points.size() > 1) {
    var_total = var_total / (float) (points.size() - 1.0);
    for (int d = 0; d < dims_; ++d) {
      inverse_variances[d] = 1.0 / (var_total[d]);
      log_determinant += log(var_total[d]);
    }
  } else {
    for (int d = 0; d < dims_; ++d) {
      inverse_variances[d] = 1.0;
    }
  }
}

double Gaussian::LogLikelihood(const DataPoint &point) const {
  // Mahanobolis distance
  double result =  
    - 0.5 * (dims_ * log(2.0 * M_PI)  +
             log_determinant +
             inner_prod(element_prod((center - point), inverse_variances), 
                        (center - point)));
  return result;

}
