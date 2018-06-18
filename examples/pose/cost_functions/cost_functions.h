#include "ceres/ceres.h"
#include "Eigen/Eigen"
#ifndef SOPHUS_CERES_FOUND
    #define SOPHUS_CERES_FOUND
#endif
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

#include "EigenUtils.h"
struct CameraOdometerError {
  CameraOdometerError(const Eigen::Matrix4d& x, const Eigen::Matrix4d& y)
        : Rx(x.block<3, 3>(0, 0)), Ry(y.block<3, 3>(0, 0)),
          qx(Rx), qy(Ry),
          tx(x.block<3, 1>(0, 3)), ty(y.block<3, 1>(0, 3)) {}

  template <typename T>
  bool operator()(const T* const q4x1, const T* const t3x1, T* residuals) const {
    Eigen::Quaternion<T> q(q4x1[0], q4x1[1], q4x1[2], q4x1[3]);
    Eigen::Matrix<T, 3, 1> t;
    t << t3x1[0], t3x1[1], T(0);

    Eigen::Matrix<T, 3, 1> t_err = (Rx.cast<T>() - Eigen::Matrix<T, 3, 3>::Identity()) * t - q._transformVector(ty.cast<T>()) + tx.cast<T>();

    Eigen::Quaternion<T> q_err = q.conjugate() * qx.cast<T>() * q * qy.conjugate().cast<T>();
    Eigen::Matrix<T, 3, 3> R_err = q_err.toRotationMatrix();

    T roll, pitch, yaw;
    mat2RPY(R_err, roll, pitch, yaw);

    residuals[0] = t_err(0);
    residuals[1] = t_err(1);
    residuals[2] = t_err(2);
    residuals[3] = roll;
    residuals[4] = pitch;
    residuals[5] = yaw;

    return true;
  }

  static ceres::CostFunction* create(const Eigen::Matrix4d& x, const Eigen::Matrix4d& y) {
    return new ceres::AutoDiffCostFunction<CameraOdometerError, 6, 4, 3>(new CameraOdometerError(x, y));
  }

  protected:
  Eigen::Matrix3d Rx, Ry;
  Eigen::Quaterniond qx,qy;
  Eigen::Vector3d tx,ty;


};

