#include "purify/spherical_resample.h"
#include <iostream>
#include "purify/utilities.h"
namespace purify {
namespace spherical_resample {

t_real calculate_l(const t_real phi, const t_real theta) { return std::cos(phi) * std::sin(theta); }

t_real calculate_m(const t_real phi, const t_real theta) { return std::sin(phi) * std::sin(theta); }

t_real calculate_n(const t_real theta) { return std::cos(theta); }

t_real calculate_l(const t_real phi, const t_real theta, const t_real alpha, const t_real beta,
                   const t_real gamma) {
  return calculate_rotated_l<t_real>(calculate_l(phi, theta), calculate_m(phi, theta),
                                     calculate_n(theta), alpha, beta, gamma);
}

t_real calculate_m(const t_real phi, const t_real theta, const t_real alpha, const t_real beta,
                   const t_real gamma) {
  return calculate_rotated_m<t_real>(calculate_l(phi, theta), calculate_m(phi, theta),
                                     calculate_n(theta), alpha, beta, gamma);
}

t_real calculate_n(const t_real phi, const t_real theta, const t_real alpha, const t_real beta,
                   const t_real gamma) {
  return calculate_rotated_n<t_real>(calculate_l(phi, theta), calculate_m(phi, theta),
                                     calculate_n(theta), alpha, beta, gamma);
}

std::tuple<t_real, t_real, t_real> matrix_to_euler(const Matrix<t_real> &input) {
  if(input.cols() !=3 or input.rows() !=3)
    throw std::runtime_error("Rotation matrix is not 3x3");
  const t_real det = input.determinant();
  if (std::abs(det) - 1 > 1e-12)
    throw std::runtime_error("Rotation matrix is not unitary. Something is wrong. det = " +
                             std::to_string(det));
  t_real alpha = 0;  // z0
  t_real beta = 0;   // y
  t_real gamma = 0;  // z1

  if (input(2, 2) < 1) {
    if (input(2, 2) > -1) {
      beta = std::acos(input(2, 2));
      alpha = std::atan2(input(1, 2), input(0, 2));
      gamma = std::atan2(input(2, 1), -input(2, 0));
    } else {
      beta = constant::pi;
      alpha = -std::atan2(input(1, 0), input(1, 1));
      gamma = 0.;
    }

  } else {
    beta = 0.;
    alpha = std::atan2(input(1, 0), input(1, 1));
    gamma = 0.;
  }
  PURIFY_LOW_LOG("Euler angles calculated to be (alpha = {}, beta = {}, gamma = {}) degrees in zyz convention",
              alpha * 180. / constant::pi, beta * 180. / constant::pi, gamma * 180. / constant::pi);
  return std::make_tuple(alpha, beta, gamma);
}

std::vector<t_int> generate_indicies(const Vector<t_real> &l, const Vector<t_real> &m,
                                     const Vector<t_real> &n, const t_real L, const t_real M) {
  if (l.size() != m.size()) throw std::runtime_error("number of l and m samples do not match.");
  if (l.size() != n.size()) throw std::runtime_error("number of l and n samples do not match.");
  if (l.size() < 1) throw std::runtime_error("number of l is less than 1.");
  std::vector<t_int> indicies(0);
  for (t_int i = 0; i < l.size(); i++) {
    if (((l(i) * l(i) / (L * L * 0.25) + (m(i) * m(i) / (M * M * 0.25))) < 1.) and (n(i) > 0.)) {
      indicies.push_back(i);
    }
  }
  PURIFY_LOW_LOG("Mask has {} elements.", indicies.size());
  return indicies;
}

Vector<t_real> generate_mask(const Vector<t_real> &l, const Vector<t_real> &m,
                             const Vector<t_real> &n, const t_real L, const t_real M) {
  auto indicies = generate_indicies(l, m, n, L, M);
  if (indicies.size() == 0)
    throw std::runtime_error("Field of view does not overlap with sphere, so mask is empty.");
  Vector<t_real> mask = Vector<t_real>::Zero(l.size());
  for (auto const &k : indicies) mask(k) = 1.;
  return mask;
}

Sparse<t_complex> init_resample_matrix_2d(const Vector<t_real> &l, const Vector<t_real> &m,
                                          const t_int imsizey_upsampled,
                                          const t_int imsizex_upsampled,
                                          const std::function<t_real(t_real)> kernell,
                                          const std::function<t_real(t_real)> kernelm,
                                          const t_int Jl, const t_int Jm,
                                          const std::function<t_complex(t_real, t_real)> &dde,
                                          const t_real dl_upsampled, const t_real dm_upsampled) {
  const t_int rows = l.size();
  const t_int cols = imsizex_upsampled * imsizey_upsampled;
  if (l.size() != m.size())
    throw std::runtime_error(
        "Size of l and m vectors are not the same for creating resampling matrix.");

  Sparse<t_complex> interpolation_matrix(rows, cols);
  interpolation_matrix.reserve(Vector<t_int>::Constant(rows, Jl * Jm));

  const t_int jl_max = std::min(Jl, imsizex_upsampled);
  const t_int jm_max = std::min(Jm, imsizey_upsampled);
  // If I collapse this for loop there is a crash when using MPI... Sparse<>::insert() doesn't work
  // right
#pragma omp parallel for
  for (t_int k = 0; k < rows; ++k) {
    for (t_int jl = 1; jl < jl_max + 1; ++jl) {
      for (t_int jm = 1; jm < jm_max + 1; ++jm) {
        const t_real k_l = std::floor(l(k) - jl_max * 0.5);
        const t_real k_m = std::floor(m(k) - jm_max * 0.5);
        const t_int q = k_l + jl;
        const t_int p = k_m + jm;
        const t_int index = utilities::sub2ind(std::floor((p + imsizey_upsampled * 0.5)),
                                               std::floor((q + imsizex_upsampled * 0.5)),
                                               imsizey_upsampled, imsizex_upsampled);
        assert(k >= 0);
        assert(k < rows);
        if ((cols > index) and (index >= 0))
          interpolation_matrix.insert(k, index) =
              kernell(l(k) - (k_l + jl)) * kernelm(m(k) - (k_m + jm)) *
              std::conj(dde(l(k) * dl_upsampled, m(k) * dm_upsampled));
      }
    }
  }
  return interpolation_matrix;
}

}  // namespace spherical_resample
}  // namespace purify
