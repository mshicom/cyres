#include "ceres/ceres.h"
#include "Eigen/Eigen"
#define SOPHUS_CERES_FOUND
#include "sophus/se3.hpp"

// ideal pose from master's perspective, wTb[i] = wTa[i] * aTb
// slave's actual pose,  wTb[i] = (wTa[0]*aTb*b[0]Tv)vTb[i-1] * b[i-1]Tb[i]
struct SimilarityCost {
  SimilarityCost() {}

  template <typename T>
  bool operator()(const T* const aTb_s,  // aTb
                  const T* const wTa_s,  // wTa
                  const T* const vTb_s,  // vTb
                  const T* const scale, T* sResiduals) const {
    const Eigen::Map<const Sophus::SE3Group<T>> aTb(aTb_s);
    const Eigen::Map<const Sophus::SE3Group<T>> wTa(wTa_s);
    const Eigen::Map<const Sophus::SE3Group<T>> vTb(vTb_s);

    const Sophus::SE3Group<T> wTb_ = wTa * aTb;
    const Sophus::SE3Group<T> vTb_scaled(vTb.so3(), scale[0] * vTb.translation());

    const Sophus::SE3Group<T> wTb = aTb * vTb_scaled;

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(sResiduals);

    residuals = (wTb_.inverse() * wTb).log();
    return true;
  }

  static ceres::CostFunction* create() {
    return new ceres::AutoDiffCostFunction<SimilarityCost, 6, 7, 7, 7, 1>(new SimilarityCost());
  }
};


