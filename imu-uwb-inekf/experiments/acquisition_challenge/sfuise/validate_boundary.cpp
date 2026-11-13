// Check sole-endpoint startup evaluation against a valid duplicate-endpoint spline.
#include <iostream>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include "SplineState.h"

// Compare only the first integration time used by the unmodified author startup.
int main() {
#ifdef _WIN32
    // Keep deliberate strict-source bounds diagnostics from opening error dialogs.
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    // The author starts with a 0.1 s knot interval and a zero local pose and bias.
    constexpr int64_t dt_ns = 100000000;
    SplineState one, two;
    one.init(dt_ns, 0, 0);
    two.init(dt_ns, 0, 0);
    Eigen::Quaterniond identity = Eigen::Quaterniond::Identity();
    Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 6, 1> bias = Eigen::Matrix<double, 6, 1>::Zero();
    one.addOneStateKnot(identity, zero, bias);
    two.addOneStateKnot(identity, zero, bias);
    two.addOneStateKnot(identity, zero, bias);
    Eigen::Quaterniond q_one, q_two;
    Eigen::Vector3d w_one, w_two;
    one.itpQuaternion(0, &q_one, &w_one);
    two.itpQuaternion(0, &q_two, &w_two);
    double error = (one.itpPosition(0) - two.itpPosition(0)).norm()
        + (one.itpPosition<1>(0) - two.itpPosition<1>(0)).norm()
        + (one.itpPosition<2>(0) - two.itpPosition<2>(0)).norm()
        + (one.itpBias(0) - two.itpBias(0)).norm()
        + (q_one.coeffs() - q_two.coeffs()).norm() + (w_one - w_two).norm();
    std::cout << "sole_endpoint_vs_valid_duplicate_total_error=" << error << '\n';
    return error == 0 ? 0 : 1;
}
