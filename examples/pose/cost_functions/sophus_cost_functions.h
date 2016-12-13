#include "ceres/ceres.h"
#include "Eigen/Eigen"
#ifndef SOPHUS_CERES_FOUND
    #define SOPHUS_CERES_FOUND
#endif
#include "sophus/se3.hpp"

class LocalParameterizationSE3 : public ceres::LocalParameterization {
 public:
  virtual ~LocalParameterizationSE3() {}

  /**
   * \brief SE3 plus operation for Ceres
   *
   * \f$ T\cdot\exp(\widehat{\delta}) \f$
   */
  virtual bool Plus(const double* T_raw, const double* delta_raw, double* T_plus_delta_raw) const {
    const Eigen::Map<const Sophus::SE3d> T(T_raw);
    const Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta(delta_raw);
    Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = T * Sophus::SE3d::exp(delta);
    return true;
  }

  /**
   * \brief Jacobian of SE3 plus operation for Ceres
   *
   * \f$ \frac{\partial}{\partial
   * \delta}T\cdot\exp(\widehat{\delta})|_{\delta=0} \f$
   */
  virtual bool ComputeJacobian(const double* T_raw, double* jacobian_raw) const {
    const Eigen::Map<const Sophus::SE3d> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_raw);
    jacobian = T.internalJacobian().transpose();
    return true;
  }

  virtual int GlobalSize() const { return Sophus::SE3d::num_parameters; }

  virtual int LocalSize() const { return Sophus::SE3d::DoF; }
};

struct TestCostFunctor {
  TestCostFunctor(Sophus::SE3d T_aw) : T_aw(T_aw) {}

  template <typename T>
  bool operator()(const T* const sT_wa, T* sResiduals) const {
    const Eigen::Map<const Sophus::SE3Group<T> > T_wa(sT_wa);
    Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(sResiduals);

    residuals = (T_aw.cast<T>() * T_wa).log();
    return true;
  }

  Sophus::SE3d T_aw;

  static ceres::CostFunction* create(Sophus::SE3d T_aw) {
    return new ceres::AutoDiffCostFunction<TestCostFunctor, 6, 7>(new TestCostFunctor(T_aw));
  }

};

template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 6, 1> log_decoupled(
    const Sophus::SE3Group<Scalar>& a, const Sophus::SE3Group<Scalar>& b) {
  Eigen::Matrix<Scalar, 6, 1> res;
  res.template head<3>() = a.translation() - b.translation();
  res.template tail<3>() = (a.so3() * b.so3().inverse()).log();
  return res;
}

struct AdjointMotionCost {
  AdjointMotionCost(const Sophus::SE3d& x, const Sophus::SE3d& y)
      : x_(x), y_(y)  {}

  template <typename T>
  bool operator()(const T* const xTy_s, T* sResiduals) const {
    const Eigen::Map<const Sophus::SE3Group<T>> xTy(xTy_s);
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(sResiduals);

    const Sophus::SE3Group<T> x_est = xTy * y_.cast<T>() * xTy.inverse();

//    residuals = (x_inverse_.cast<T>() * x_est).log();
    residuals = log_decoupled(x_est, x_.cast<T>());
    return true;
  }

  static ceres::CostFunction* create(const Sophus::SE3d& x, const Sophus::SE3d& y) {
    return new ceres::AutoDiffCostFunction<AdjointMotionCost, 6, 7>(new AdjointMotionCost(x, y));
  }
  protected:
    Sophus::SE3d x_;
    Sophus::SE3d y_;
};
