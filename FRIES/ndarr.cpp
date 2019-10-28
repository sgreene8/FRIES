// Copied from https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op

#include "ndarr.hpp"

//inline
//FourDArr::FourDArr(size_t len1, size_t len2, size_t len3, size_t len4)
//  : len1_(len1)
//  , len2_(len2)
//  , len3_(len3)
//  , len4_(len4)
//{
//  data_ = new double[len1 * len2 * len3 * len4];
//}
//inline
//FourDArr::~FourDArr()
//{
//  delete[] data_;
//}
inline
double& FourDArr::operator() (size_t i1, size_t i2, size_t i3, size_t i4)
{
  return data_[i1 * len2_ * len3_ * len4_ + i2 * len3_ * len4_ + i3 * len4_ + i4];
}


//inline
//double FourDArr::operator() (size_t i1, size_t i2, size_t i3, size_t i4) const
//{
//  return data_[i1 * len2_ * len3_ * len4_ + i2 * len3_ * len4_ + i3 * len4_ + i4];
//}
