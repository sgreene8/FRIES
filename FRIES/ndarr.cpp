/*! \file C++ definitions of classes for variable-length multidimensional arrays
* Largely copied from https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
*/

#include "ndarr.hpp"

inline
double& FourDArr::operator() (size_t i1, size_t i2, size_t i3, size_t i4)
{
  return data_[i1 * len2_ * len3_ * len4_ + i2 * len3_ * len4_ + i3 * len4_ + i4];
}


double  FourDArr::operator() (size_t i1, size_t i2, size_t i3, size_t i4) const {
    return data_[i1 * len2_ * len3_ * len4_ + i2 * len3_ * len4_ + i3 * len4_ + i4];
}
