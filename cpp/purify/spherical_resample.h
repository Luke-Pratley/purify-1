#ifndef PURIFY_SPHERICAL_RESAMPLE_H
#define PURIFY_SPHERICAL_RESAMPLE_H

#include "purify/config.h"
#include "purify/types.h"
#include "purify/logging.h"
#include "purify/operators.h"
#include "purify/wproj_operators.h"

namespace purify {
namespace spherical_resample {
//! calculate l directional cosine for a given pointing given angles
t_real calculate_l(const t_real phi, const t_real theta);
//! calculate m directional cosine for a given pointing given angles
t_real calculate_m(const t_real phi, const t_real theta);
//! calculate n directional cosine for a given pointing given angles
t_real calculate_n(const t_real theta);
//! calculate the rotated l from euler angles in zyz and starting coordinates (l, m, n)
template <class T>
T calculate_rotated_l(const T &l, const T &m, const T &n, const t_real alpha, const t_real beta,
                      const t_real gamma) {
  return l * (std::cos(alpha) * std::cos(beta) * std::cos(gamma) -
              std::sin(alpha) * std::sin(gamma)) +
         m * (-std::cos(alpha) * std::cos(beta) * std::sin(gamma) -
              std::sin(alpha) * std::cos(gamma)) +
         n * std::cos(alpha) * std::sin(beta);
}
//! calculate the rotated m from euler angles in zyz and starting coordinates (l, m, n)
template <class T>
T calculate_rotated_m(const T &l, const T &m, const T &n, const t_real alpha, const t_real beta,
                      const t_real gamma) {
  return l * (std::sin(alpha) * std::cos(beta) * std::cos(gamma) +
              std::cos(alpha) * std::sin(gamma)) +
         m * (-std::sin(alpha) * std::cos(beta) * std::sin(gamma) +
              std::cos(alpha) * std::cos(gamma)) +
         n * std::sin(alpha) * std::sin(beta);
}
//! calculate the rotated n from euler angles in zyz and starting coordinates (l, m, n)
template <class T>
T calculate_rotated_n(const T &l, const T &m, const T &n, const t_real alpha, const t_real beta,
                      const t_real gamma) {
  return l * (-std::sin(beta) * std::cos(gamma)) + m * (std::sin(beta) * std::sin(gamma)) +
         n * std::cos(beta);
}
//! calculate the l in a rotated frame from euler angles in zyz
t_real calculate_l(const t_real phi, const t_real theta, const t_real alpha, const t_real beta,
                   const t_real gamma);
//! calculate the m in a rotated frame from euler angles in zyz
t_real calculate_m(const t_real phi, const t_real theta, const t_real alpha, const t_real beta,
                   const t_real gamma);
//! calculate the n in a rotated frame from euler angles in zyz
t_real calculate_n(const t_real phi, const t_real theta, const t_real alpha, const t_real beta,
                   const t_real gamma);
//! convert rotation matrix to Euler angles in zyz order
std::tuple<t_real, t_real, t_real> matrix_to_euler(const Matrix<t_real> &input);
//! find prefered rotation to minimize w values
std::tuple<t_real, t_real, t_real> find_prefered_direction(const Vector<t_real> &u,
                                                           const Vector<t_real> &v,
                                                           const Vector<t_real> &w);
#ifdef PURIFY_MPI
//! find prefered rotation to minimize w values with MPI
std::tuple<t_real, t_real, t_real> find_prefered_direction(const Vector<t_real> &u,
                                                           const Vector<t_real> &v,
                                                           const Vector<t_real> &w,
                                                           const sopt::mpi::Communicator &comm);
#endif

//! generate indicies that overlap with imaging field of view for resampling
std::vector<t_int> generate_indicies(const Vector<t_real> &l, const Vector<t_real> &m,
                                     const Vector<t_real> &n, const t_real L, const t_real M);
//! generate a mask for imaged field of view that is going to be resampled
Vector<t_real> generate_mask(const Vector<t_real> &l, const Vector<t_real> &m,
                             const Vector<t_real> &n, const t_real L, const t_real M);
//! sparse matrix that degrids/grids between spherical sampling pattern and regular grid
Sparse<t_complex> init_resample_matrix_2d(
    const Vector<t_real> &l, const Vector<t_real> &m, const t_int imsizey_upsampled,
    const t_int imsizex_upsampled, const std::function<t_real(t_real)> kernell,
    const std::function<t_real(t_real)> kernelm, const t_int Jl, const t_int Jm,
    const std::function<t_complex(t_real, t_real)> &dde = [](const t_real l,
                                                             const t_real m) { return 1.; },
    const t_real dl_upsampled = 0., const t_real dm_upsampled = 0.);

//! generate the l and m coordinates in pixels and their indicies for resampling
template <class T>
std::tuple<Vector<t_real>, Vector<t_real>, std::vector<t_int>> calculate_compressed_lm(
    const t_int number_of_samples, const t_real phi_0, const t_real theta_0, const T &phi,
    const T &theta, const t_int imsizey_upsampled, const t_int imsizex_upsampled,
    const t_real dl_upsampled, const t_real dm_upsampled) {
  Vector<t_real> l = Vector<t_real>::Zero(number_of_samples);
  Vector<t_real> m = Vector<t_real>::Zero(number_of_samples);
  Vector<t_real> n = Vector<t_real>::Zero(number_of_samples);
  for (t_int k = 0; k < number_of_samples; k++) {
    l(k) = spherical_resample::calculate_l(phi(k), theta(k), constant::pi / 2, theta_0, phi_0);
    m(k) = spherical_resample::calculate_m(phi(k), theta(k), constant::pi / 2, theta_0, phi_0);
    n(k) = spherical_resample::calculate_n(phi(k), theta(k), constant::pi / 2, theta_0, phi_0);
  }
  const std::vector<t_int> indicies = generate_indicies(l, m, n, imsizex_upsampled * dl_upsampled,
                                                        imsizey_upsampled * dm_upsampled);
  if (indicies.size() < 1)
    throw std::runtime_error(
        "indicies is zero when constructing spherical resampling; check number_of_samples.");
  Vector<t_real> l_compressed = Vector<t_real>::Zero(indicies.size());
  Vector<t_real> m_compressed = Vector<t_real>::Zero(indicies.size());
  for (t_int k = 0; k < indicies.size(); k++) {
    l_compressed(k) = l(indicies.at(k)) / dl_upsampled;
    m_compressed(k) = m(indicies.at(k)) / dm_upsampled;
  }
  if (indicies.size() != l_compressed.size())
    throw std::runtime_error("Indicies is not the same size as l_compressed.");
  if (indicies.size() != m_compressed.size())
    throw std::runtime_error("Indicies is not the same size as m_compressed.");

  return std::make_tuple(l_compressed, m_compressed, indicies);
}
//! cutout region of sphere used for resampling
template <class K>
std::tuple<sopt::OperatorFunction<K>, sopt::OperatorFunction<K>> init_pop_indicies_operator(
    const std::vector<t_int> &indicies, const t_int number_of_samples) {
  return std::make_tuple(
      [=](K &output, const K &input) {
        assert(input.size() == number_of_samples);
        output = K::Zero(indicies.size());
        for (t_int k = 0; k < indicies.size(); k++) output(k) = input(indicies.at(k));
      },
      [=](K &output, const K &input) {
        assert(input.size() == indicies.size());
        output = K::Zero(number_of_samples);
        for (t_int k = 0; k < indicies.size(); k++) output(indicies.at(k)) = input(k);
      });
}

//! returns an operator that will resample from a subset of coordinates
template <class K>
std::tuple<sopt::OperatorFunction<K>, sopt::OperatorFunction<K>> init_resample_operator_2d(
    const t_int number_of_samples, const Vector<t_real> &l_compressed,
    const Vector<t_real> &m_compressed, const t_int imsizey_upsampled,
    const t_int imsizex_upsampled, const std::function<t_real(t_real)> &kernell,
    const std::function<t_real(t_real)> &kernelm, const t_int Jl, const t_int Jm,
    const std::function<t_complex(t_real, t_real)> &dde, const t_real dl_upsampled,
    const t_real dm_upsampled) {
  if ((Jl <= 0) or (Jm <= 0))
    throw std::runtime_error("Support size of resampling kernel is not positive");
  if ((imsizex_upsampled <= 0) or (imsizey_upsampled <= 0))
    throw std::runtime_error("Tangent image size is not greater than zero.");
  if (number_of_samples <= 0)
    throw std::runtime_error("Number of samples on the sphere is not greater than zero.");

  const Sparse<t_complex> interrpolation_matrix =
      init_resample_matrix_2d(l_compressed, m_compressed, imsizey_upsampled, imsizex_upsampled,
                              kernell, kernelm, Jl, Jm, dde, dl_upsampled, dm_upsampled);
  const Sparse<t_complex> interrpolation_matrix_adjoint = interrpolation_matrix.adjoint();
  return std::make_tuple(
      [=](K &output, const K &input) { output = interrpolation_matrix_adjoint * input; },
      [=](K &output, const K &input) {
        assert(input.size() == imsizex_upsampled * imsizey_upsampled);
        output = interrpolation_matrix * input;
      });
}
//! returns an on the fly operator that will resample from a subset of coordinates
template <class T>
std::tuple<sopt::OperatorFunction<T>, sopt::OperatorFunction<T>>
init_on_the_fly_resample_operator_2d(const Vector<t_real> &l_compressed,
                                     const Vector<t_real> &m_compressed,
                                     const t_int imsizey_upsampled, const t_int imsizex_upsampled,
                                     const std::function<t_real(t_real)> &kernell, const t_int Jl,
                                     const t_int total_samples,
                                     const std::function<t_complex(t_real, t_real)> &dde,
                                     const t_real dl_upsampled, const t_real dm_upsampled) {
  const t_int rows = l_compressed.size();
  const t_int cols = imsizex_upsampled * imsizey_upsampled;
  if (l_compressed.size() != m_compressed.size())
    throw std::runtime_error(
        "Size of l and m vectors are not the same for creating resampling matrix.");

  const t_int jl_max = std::min(Jl, imsizex_upsampled);
  const t_int jm_max = std::min(Jl, imsizey_upsampled);
  const std::shared_ptr<Vector<t_real>> l_ptr = std::make_shared<Vector<t_real>>(l_compressed);
  const std::shared_ptr<Vector<t_real>> m_ptr = std::make_shared<Vector<t_real>>(m_compressed);
  const auto samples = kernels::kernel_samples(
      total_samples, [&](const t_real x) { return kernell(x * jl_max * 0.5); });
  const std::shared_ptr<Vector<t_complex>> dde_ptr =
      std::make_shared<Vector<t_complex>>(Vector<t_complex>::Zero(l_compressed.size()));
  for (t_int k = 0; k < rows; ++k) {
    assert(k < rows);
    assert(k >= 0);
    const t_real l_val = (*l_ptr)(k);
    const t_real m_val = (*m_ptr)(k);
    (*dde_ptr)(k) = dde(l_val * dl_upsampled, m_val * dm_upsampled);
  }
  const auto degrid = [rows, cols, jl_max, jm_max, dde_ptr, l_ptr, m_ptr, imsizex_upsampled,
                       imsizey_upsampled, samples, dl_upsampled, dm_upsampled,
                       total_samples](T &output, const T &input) {
    output = T::Zero(l_ptr->size());
    assert(input.size() == cols);
#pragma omp parallel for
    for (t_int k = 0; k < rows; ++k) {
      assert(k < rows);
      assert(k >= 0);
      const t_real l_val = (*l_ptr)(k);
      const t_real m_val = (*m_ptr)(k);
      const t_complex dde_val = std::conj((*dde_ptr)(k));
      for (t_int jl = 1; jl < jl_max + 1; ++jl) {
        const t_real k_l = std::floor(l_val - jl_max * 0.5);
        const t_int q = k_l + jl;
        const t_real c_0 =
            static_cast<t_int>(std::floor(2 * std::abs(l_val - q) * (total_samples - 1) / jl_max));
        assert(c_0 >= 0);
        assert(c_0 < total_samples);
        const t_complex kernell_val = samples[c_0] * dde_val;
        for (t_int jm = 1; jm < jm_max + 1; ++jm) {
          const t_real k_m = std::floor(m_val - jm_max * 0.5);
          const t_int p = k_m + jm;
          const t_int index = utilities::sub2ind(std::floor((p + imsizey_upsampled * 0.5)),
                                                 std::floor((q + imsizex_upsampled * 0.5)),
                                                 imsizey_upsampled, imsizex_upsampled);
          const t_real i_0 = static_cast<t_int>(
              std::floor(2 * std::abs(m_val - p) * (total_samples - 1) / jm_max));
          if ((cols > index) and (index >= 0))
            output(k) += samples[i_0] * kernell_val * input(index);
        }
      }
    }
  };

  const auto grid = [rows, cols, jl_max, jm_max, dde_ptr, l_ptr, m_ptr, imsizex_upsampled,
                     imsizey_upsampled, samples, dl_upsampled, dm_upsampled,
                     total_samples](T &output, const T &input) {
    assert(input.size() == l_ptr->size());
    output = T::Zero(cols);
#pragma omp parallel for
    for (t_int k = 0; k < rows; ++k) {
      assert(k < rows);
      assert(k >= 0);
      const t_real l_val = (*l_ptr)(k);
      const t_real m_val = (*m_ptr)(k);
      const t_complex dde_val = (*dde_ptr)(k);
      for (t_int jl = 1; jl < jl_max + 1; ++jl) {
        const t_real k_l = std::floor(l_val - jl_max * 0.5);
        const t_int q = k_l + jl;
        const t_real c_0 =
            static_cast<t_int>(std::floor(2 * std::abs(l_val - q) * (total_samples - 1) / jl_max));
        assert(c_0 >= 0);
        assert(c_0 < total_samples);
        const t_complex kernell_val = samples[c_0] * dde_val;
        for (t_int jm = 1; jm < jm_max + 1; ++jm) {
          const t_real k_m = std::floor(m_val - jm_max * 0.5);
          const t_int p = k_m + jm;
          const t_int index = utilities::sub2ind(std::floor((p + imsizey_upsampled * 0.5)),
                                                 std::floor((q + imsizex_upsampled * 0.5)),
                                                 imsizey_upsampled, imsizex_upsampled);
          const t_real i_0 = static_cast<t_int>(
              std::floor(2 * std::abs(m_val - p) * (total_samples - 1) / jm_max));
          if ((cols > index) and (index >= 0)) {
#pragma omp critical(resample_fly)
            {
              output(index) += samples[i_0] * kernell_val * input(k);
            }
          }
        }
      }
    }
  };
  return std::make_tuple(grid, degrid);
}  // namespace spherical_resample
//! return operator that will resample between the sphere and the plane with masking
template <class K, class T>
std::tuple<sopt::OperatorFunction<K>, sopt::OperatorFunction<K>> init_mask_and_resample_operator_2d(
    const t_int number_of_samples, const t_real phi_0, const t_real theta_0, const T &phi,
    const T &theta, const t_int imsizey, const t_int imsizex,
    const t_real oversample_ratio_image_domain, const t_real dl, const t_real dm,
    const std::function<t_real(t_real)> &kernell, const std::function<t_real(t_real)> &kernelm,
    const t_int Jl, const t_int Jm, const std::function<t_complex(t_real, t_real)> &dde,
    const t_real coordinate_scaling) {
  if ((Jl <= 0) or (Jm <= 0))
    throw std::runtime_error("Support size of resampling kernel is not positive");
  if ((imsizex <= 0) or (imsizey <= 0))
    throw std::runtime_error("Tangent image size is not greater than zero.");
  if (number_of_samples <= 0)
    throw std::runtime_error("Number of samples on the sphere is not greater than zero.");
  bool on_the_fly = true;
  // Find indicies for cutting to resample
  const t_int imsizex_upsampled = std::floor(imsizex * oversample_ratio_image_domain);
  const t_int imsizey_upsampled = std::floor(imsizey * oversample_ratio_image_domain);
  const t_real dl_upsampled = dl / oversample_ratio_image_domain;
  const t_real dm_upsampled = dm / oversample_ratio_image_domain;

  const auto lm_with_mask = calculate_compressed_lm<T>(
      number_of_samples, phi_0, theta_0, phi, theta, imsizey_upsampled - Jm, imsizex_upsampled - Jl,
      dl_upsampled, dm_upsampled);
  const Vector<t_real> &l_compressed = std::get<0>(lm_with_mask);
  const Vector<t_real> &m_compressed = std::get<1>(lm_with_mask);
  const std::vector<t_int> &indicies = std::get<2>(lm_with_mask);

  // Create operator to cutout region of sphere for resampling
  const auto cutting_operator = init_pop_indicies_operator<K>(indicies, number_of_samples);
  // Operator to perform resampling
  const auto gridding_operator =
      (on_the_fly)
          ? init_on_the_fly_resample_operator_2d<K>(l_compressed, m_compressed, imsizey_upsampled,
                                                    imsizex_upsampled, kernell, Jl, Jl * 1e5, dde,
                                                    dl_upsampled, dm_upsampled)
          : init_resample_operator_2d<K>(number_of_samples, l_compressed * coordinate_scaling,
                                         m_compressed * coordinate_scaling, imsizey_upsampled,
                                         imsizex_upsampled, kernell, kernelm, Jl, Jm, dde,
                                         dl_upsampled, dm_upsampled);
  // Combining them together
  auto direct =
      sopt::chained_operators<K>(std::get<0>(gridding_operator), std::get<0>(cutting_operator));
  auto indirect =
      sopt::chained_operators<K>(std::get<1>(cutting_operator), std::get<1>(gridding_operator));
  return std::make_tuple(direct, indirect);
}

//! gridding correction and zero padding in the FT domain (with FFT shift operations accounted for)
template <class T>
std::tuple<sopt::OperatorFunction<T>, sopt::OperatorFunction<T>> init_FT_zero_padding_2d(
    const Image<typename T::Scalar> &S, const t_real oversample_ratio) {
  const t_uint imsizex_ = S.cols();
  const t_uint imsizey_ = S.rows();
  const t_uint ftsizeu_ = std::floor(imsizex_ * oversample_ratio);
  const t_uint ftsizev_ = std::floor(imsizey_ * oversample_ratio);
  const t_uint x_start = std::floor(ftsizeu_ * 0.5 - imsizex_ * 0.5);
  const t_uint y_start = std::floor(ftsizev_ * 0.5 - imsizey_ * 0.5);
  auto direct = [=](T &output, const T &x) {
    assert(x.size() == ftsizeu_ * ftsizev_);
    output = T::Zero(imsizey_ * imsizex_);
    for (t_uint j = 0; j < imsizey_; j++) {
      for (t_uint i = 0; i < imsizex_; i++) {
        const t_uint output_index =
            utilities::sub2ind(utilities::mod(j - imsizey_ * 0.5, imsizey_),
                               utilities::mod(i - imsizex_ * 0.5, imsizex_), imsizey_, imsizex_);
        const t_uint input_index = utilities::sub2ind(
            utilities::mod(y_start + j - ftsizev_ * 0.5, ftsizev_),
            utilities::mod(x_start + i - ftsizeu_ * 0.5, ftsizeu_), ftsizev_, ftsizeu_);
        output(output_index) = S(j, i) * x(input_index);
      }
    }
  };
  auto indirect = [=](T &output, const T &x) {
    assert(x.size() == imsizex_ * imsizey_);
    output = T::Zero(ftsizeu_ * ftsizev_);
    for (t_uint j = 0; j < imsizey_; j++) {
      for (t_uint i = 0; i < imsizex_; i++) {
        const t_uint input_index =
            utilities::sub2ind(utilities::mod(j - imsizey_ * 0.5, imsizey_),
                               utilities::mod(i - imsizex_ * 0.5, imsizex_), imsizey_, imsizex_);
        const t_uint output_index = utilities::sub2ind(
            utilities::mod(y_start + j - ftsizev_ * 0.5, ftsizev_),
            utilities::mod(x_start + i - ftsizeu_ * 0.5, ftsizeu_), ftsizev_, ftsizeu_);
        output(output_index) = std::conj(S(j, i)) * x(input_index);
      }
    }
  };
  return std::make_tuple(direct, indirect);
}

//! generates operator to scale, zero padd, FFT, crop, scale
template <class T>
std::tuple<sopt::OperatorFunction<T>, sopt::OperatorFunction<T>> base_padding_and_FFT_2d(
    const std::function<t_real(t_real)> &ftkernelu, const std::function<t_real(t_real)> &ftkernelv,
    const std::function<t_real(t_real)> &ftkernell, const std::function<t_real(t_real)> &ftkernelm,
    const t_uint imsizey, const t_uint imsizex, const t_real oversample_ratio = 2,
    const t_real oversample_ratio_image_domain = 2,
    const operators::fftw_plan &ft_plan = operators::fftw_plan::measure) {
  sopt::OperatorFunction<T> directZ_image_domain, indirectZ_image_domain;
  sopt::OperatorFunction<T> directZ_ft_domain, indirectZ_ft_domain;
  sopt::OperatorFunction<T> directFFT, indirectFFT;
  const t_int imsizex_upsampled = std::floor(imsizex * oversample_ratio_image_domain);
  const t_int imsizey_upsampled = std::floor(imsizey * oversample_ratio_image_domain);
  const t_int ftsizeu_cropped = std::floor(imsizex * oversample_ratio);
  const t_int ftsizev_cropped = std::floor(imsizey * oversample_ratio);
  const t_int ftsizeu = std::floor(imsizex * oversample_ratio * oversample_ratio_image_domain);
  const t_int ftsizev = std::floor(imsizey * oversample_ratio * oversample_ratio_image_domain);
  PURIFY_LOW_LOG("Building Measurement Operator with Sampling on the Sphere");
  PURIFY_LOW_LOG(
      "Constructing Zero Padding "
      "and Correction Operator: "
      "Z");
  PURIFY_MEDIUM_LOG("Image size (width, height): {} x {}", imsizex, imsizey);
  PURIFY_MEDIUM_LOG("Oversampling Factor of FT grid: {}", oversample_ratio);
  PURIFY_MEDIUM_LOG("Oversampling Factor of Image grid: {}", oversample_ratio_image_domain);

  const Image<t_complex> S_l = purify::details::init_correction2d(
      oversample_ratio, imsizey_upsampled, imsizex_upsampled,
      [oversample_ratio_image_domain, &ftkernelu](t_real x) {
        return ftkernelu(x / oversample_ratio_image_domain);
      },
      [oversample_ratio_image_domain, &ftkernelv](t_real x) {
        return ftkernelv(x / oversample_ratio_image_domain);
      },
      0., 0., 0.);
  //                               std::sqrt(static_cast<t_real>(ftsizeu * ftsizev));

  std::tie(directZ_image_domain, indirectZ_image_domain) =
      purify::operators::init_zero_padding_2d<T>(S_l, oversample_ratio);

  const Image<t_complex> S_u = purify::details::init_correction2d(
      oversample_ratio_image_domain, ftsizev_cropped, ftsizeu_cropped,
      [oversample_ratio, &ftkernell](t_real x) { return ftkernell(x / oversample_ratio); },
      [oversample_ratio, &ftkernelm](t_real x) { return ftkernelm(x / oversample_ratio); }, 0., 0.,
      0.);

  std::tie(directZ_ft_domain, indirectZ_ft_domain) =
      spherical_resample::init_FT_zero_padding_2d<T>(S_u, oversample_ratio_image_domain);

  PURIFY_LOW_LOG("Constructing FFT operator: F");
  switch (ft_plan) {
  case operators::fftw_plan::measure:
    PURIFY_MEDIUM_LOG("Measuring Plans...");
    break;
  case operators::fftw_plan::estimate:
    PURIFY_MEDIUM_LOG("Estimating Plans...");
    break;
  }
  std::tie(directFFT, indirectFFT) = purify::operators::init_FFT_2d<T>(
      imsizey_upsampled, imsizex_upsampled, oversample_ratio, ft_plan);

  auto direct = sopt::chained_operators<T>(directZ_ft_domain, directFFT, directZ_image_domain);
  auto indirect =
      sopt::chained_operators<T>(indirectZ_image_domain, indirectFFT, indirectZ_ft_domain);
  return std::make_tuple(direct, indirect);
}

//! generates operator to scale, zero padd, FFT, crop, scale (wtih radial correction)
template <class T>
std::tuple<sopt::OperatorFunction<T>, sopt::OperatorFunction<T>> base_padding_and_FFT_2d(
    const std::function<t_real(t_real)> &ftkerneluv, const std::function<t_real(t_real)> &ftkernell,
    const std::function<t_real(t_real)> &ftkernelm, const t_uint imsizey, const t_uint imsizex,
    const t_real oversample_ratio = 2, const t_real oversample_ratio_image_domain = 2,
    const operators::fftw_plan &ft_plan = operators::fftw_plan::measure) {
  sopt::OperatorFunction<T> directZ_image_domain, indirectZ_image_domain;
  sopt::OperatorFunction<T> directZ_ft_domain, indirectZ_ft_domain;
  sopt::OperatorFunction<T> directFFT, indirectFFT;
  const t_int imsizex_upsampled = std::floor(imsizex * oversample_ratio_image_domain);
  const t_int imsizey_upsampled = std::floor(imsizey * oversample_ratio_image_domain);
  const t_int ftsizeu_cropped = std::floor(imsizex * oversample_ratio);
  const t_int ftsizev_cropped = std::floor(imsizey * oversample_ratio);
  const t_int ftsizeu = std::floor(imsizex * oversample_ratio * oversample_ratio_image_domain);
  const t_int ftsizev = std::floor(imsizey * oversample_ratio * oversample_ratio_image_domain);
  PURIFY_LOW_LOG("Building Measurement Operator with Sampling on the Sphere");
  PURIFY_LOW_LOG(
      "Constructing Zero Padding "
      "and Correction Operator: "
      "Z");
  PURIFY_MEDIUM_LOG("Image size (width, height): {} x {}", imsizex, imsizey);
  PURIFY_MEDIUM_LOG("Oversampling Factor of FT grid: {}", oversample_ratio);
  PURIFY_MEDIUM_LOG("Oversampling Factor of Image grid: {}", oversample_ratio_image_domain);

  const Image<t_complex> S_l = purify::details::init_correction_radial_2d(
      oversample_ratio, imsizey_upsampled, imsizex_upsampled, ftkerneluv, 0., 0., 0.);
  // std::sqrt(static_cast<t_real>(ftsizeu * ftsizev));

  std::tie(directZ_image_domain, indirectZ_image_domain) =
      purify::operators::init_zero_padding_2d<T>(S_l, oversample_ratio);

  const Image<t_complex> S_u = purify::details::init_correction2d(
      oversample_ratio_image_domain, ftsizev_cropped, ftsizeu_cropped,
      [oversample_ratio, &ftkernell](t_real x) { return ftkernell(x / oversample_ratio); },
      [oversample_ratio, &ftkernelm](t_real x) { return ftkernelm(x / oversample_ratio); }, 0., 0.,
      0.);

  std::tie(directZ_ft_domain, indirectZ_ft_domain) =
      spherical_resample::init_FT_zero_padding_2d<T>(S_u, oversample_ratio_image_domain);

  PURIFY_LOW_LOG("Constructing FFT operator: F");
  switch (ft_plan) {
  case operators::fftw_plan::measure:
    PURIFY_MEDIUM_LOG("Measuring Plans...");
    break;
  case operators::fftw_plan::estimate:
    PURIFY_MEDIUM_LOG("Estimating Plans...");
    break;
  }
  std::tie(directFFT, indirectFFT) = purify::operators::init_FFT_2d<T>(
      imsizey_upsampled, imsizex_upsampled, oversample_ratio, ft_plan);

  auto direct = sopt::chained_operators<T>(directZ_ft_domain, directFFT, directZ_image_domain);
  auto indirect =
      sopt::chained_operators<T>(indirectZ_image_domain, indirectFFT, indirectZ_ft_domain);
  return std::make_tuple(direct, indirect);
}

//! operator that degrids from the upsampled plane to the uv domain (no w-projection, only
//! w-stacking)
template <class T, class K>
std::tuple<sopt::OperatorFunction<T>, sopt::OperatorFunction<T>> base_plane_degrid_operator(
    const t_int number_of_samples, const t_real phi_0, const t_real theta_0, const K &phi,
    const K &theta, const Vector<t_real> &u, const Vector<t_real> &v, const Vector<t_real> &w,
    const Vector<t_complex> &weights, const t_real oversample_ratio = 2,
    const t_real oversample_ratio_image_domain = 2,
    const kernels::kernel kernel = kernels::kernel::kb, const t_uint Ju = 4, const t_uint Jv = 4,
    const t_uint Jl = 4, const t_uint Jm = 4,
    const operators::fftw_plan &ft_plan = operators::fftw_plan::measure,
    const bool uvw_stacking = false, const t_real L = 1, const t_real M = 1,
    const t_real coordinate_scaling = 1., const t_real beam_l = 0, const t_real beam_m = 0) {
  bool on_the_fly = true;
  t_real const w_mean = uvw_stacking ? w.array().mean() : 0.;
  t_real const v_mean = uvw_stacking ? v.array().mean() : 0.;
  t_real const u_mean = uvw_stacking ? u.array().mean() : 0.;
  if (uvw_stacking) {
    PURIFY_MEDIUM_LOG("Mean, u: {}, +/- {}", u_mean,
                      (u_mean - u.array()).cwiseAbs().maxCoeff() * 0.5);
    PURIFY_MEDIUM_LOG("Mean, v: {}, +/- {}", v_mean,
                      (v_mean - v.array()).cwiseAbs().maxCoeff() * 0.5);
    PURIFY_MEDIUM_LOG("Mean, w: {}, +/- {}", w_mean,
                      (w_mean - w.array()).cwiseAbs().maxCoeff() * 0.5);
  }

  const t_real dl = std::min({0.25 / ((u.array() - u_mean).cwiseAbs().maxCoeff()), L / 128});
  const t_real dm = std::min({0.25 / ((v.array() - v_mean).cwiseAbs().maxCoeff()), M / 128});
  const t_int imsizex = std::floor(L / dl);
  const t_int imsizey = std::floor(M / dm);
  const t_real du = widefield::dl2du(dl, imsizex, oversample_ratio);
  const t_real dv = widefield::dl2du(dm, imsizey, oversample_ratio);
  PURIFY_MEDIUM_LOG("dl x dm : {} x {} ", dl, dm);
  PURIFY_MEDIUM_LOG("du x dv : {} x {} ", du, dv);
  PURIFY_MEDIUM_LOG("FoV (width, height): {} x {} (L x M)", imsizex * dl, imsizey * dm);
  PURIFY_MEDIUM_LOG("FoV (width, height): {} x {} (deg x deg)",
                    std::asin(imsizex * dl / 2.) * 2. * 180. / constant::pi,
                    std::asin(imsizey * dm / 2.) * 2. * 180. / constant::pi);
  PURIFY_MEDIUM_LOG("Primary Beam (width, height): {} x {} (deg x deg)",
                    std::asin(beam_l / 2 / 2.) * 2. * 180. / constant::pi,
                    std::asin(beam_m / 2. / 2) * 2. * 180. / constant::pi);
  PURIFY_MEDIUM_LOG("Number of visibilities: {}", u.size());
  PURIFY_MEDIUM_LOG("Maximum u value is {} lambda, sphere needs to be sampled at dl = {}.",
                    u.cwiseAbs().maxCoeff(), std::min(0.25 / u.cwiseAbs().maxCoeff(), 1.));
  PURIFY_MEDIUM_LOG("Maximum v value is {} lambda, sphere needs to be sampled at dm = {}.",
                    v.cwiseAbs().maxCoeff(), std::min(0.25 / v.cwiseAbs().maxCoeff(), 1.));
  PURIFY_MEDIUM_LOG("Maximum w value is {} lambda, sphere needs to be sampled at dn = {}.",
                    w.cwiseAbs().maxCoeff(), std::min(0.25 / w.cwiseAbs().maxCoeff(), 1.));

  auto const uvkernels = purify::create_kernels(kernel, Ju, Jv, imsizey, imsizex, oversample_ratio);
  const std::function<t_real(t_real)> &kernelu = std::get<0>(uvkernels);
  const std::function<t_real(t_real)> &kernelv = std::get<1>(uvkernels);
  const std::function<t_real(t_real)> &ftkernelu = std::get<2>(uvkernels);
  const std::function<t_real(t_real)> &ftkernelv = std::get<3>(uvkernels);

  auto const lmkernels =
      purify::create_kernels(kernel, Jl, Jm, imsizey, imsizex, oversample_ratio_image_domain);
  const std::function<t_real(t_real)> &kernell = std::get<0>(lmkernels);
  const std::function<t_real(t_real)> &kernelm = std::get<1>(lmkernels);
  const std::function<t_real(t_real)> &ftkernell = std::get<2>(lmkernels);
  const std::function<t_real(t_real)> &ftkernelm = std::get<3>(lmkernels);

  const t_complex I(0., 1.);
  std::function<t_complex(t_real, t_real)> dde = [I, u_mean, v_mean, w_mean, imsizex, imsizey,
                                                  oversample_ratio, oversample_ratio_image_domain,
                                                  beam_m, beam_l](const t_real l, const t_real m) {
    return std::exp(-2 * constant::pi * I *
                    (u_mean * l + v_mean * m + w_mean * (std::sqrt(1. - l * l - m * m) - 1.))) /
           std::sqrt(1. - l * l - m * m) * (((l * l + m * m) < 1.) ? 1. : 0.) *
           std::pow(boost::math::sinc_pi(beam_l * l * constant::pi), 2) *
           std::pow(boost::math::sinc_pi(beam_m * m * constant::pi), 2) *
           std::sqrt(imsizex * imsizey) * oversample_ratio * oversample_ratio_image_domain;
  };

  PURIFY_LOW_LOG("Constructing Spherical Resampling Operator: P");
  sopt::OperatorFunction<T> directP, indirectP;
  std::tie(directP, indirectP) = init_mask_and_resample_operator_2d<T, K>(
      number_of_samples, phi_0, theta_0, phi, theta, imsizey, imsizex,
      oversample_ratio_image_domain, dl, dm, kernell, kernelm, Jl, Jm, dde, coordinate_scaling);
  sopt::OperatorFunction<T> directZFZ, indirectZFZ;
  std::tie(directZFZ, indirectZFZ) =
      base_padding_and_FFT_2d<T>(ftkernelu, ftkernelv, ftkernell, ftkernelm, imsizey, imsizex,
                                 oversample_ratio, oversample_ratio_image_domain, ft_plan);

  PURIFY_LOW_LOG("Constructing Weighting and Gridding Operators: WG");
  sopt::OperatorFunction<T> directG, indirectG;
  if (on_the_fly)
    std::tie(directG, indirectG) = purify::operators::init_on_the_fly_gridding_matrix_2d<T>(
        (u.array() - u_mean) / du, (v.array() - v_mean) / dv, weights, imsizey, imsizex,
        oversample_ratio, kernelu, kernelv, Ju, Jv, Ju * 1e5 / 2);
  else
    std::tie(directG, indirectG) = purify::operators::init_gridding_matrix_2d<T>(
        (u.array() - u_mean) / du, (v.array() - v_mean) / dv, weights, imsizey, imsizex,
        oversample_ratio, kernelv, kernelu, Ju, Jv);

  const auto direct = sopt::chained_operators<T>(directG, directZFZ, directP);
  const auto indirect = sopt::chained_operators<T>(indirectP, indirectZFZ, indirectG);
  PURIFY_LOW_LOG("Finished consturction of Φ.");
  return std::make_tuple(direct, indirect);
}
//! operator that degrids from the upsampled plane to the uvw domain (w-projection with
//! w-stacking)
template <class T, class K>
std::tuple<sopt::OperatorFunction<T>, sopt::OperatorFunction<T>> base_plane_degrid_wproj_operator(
    const t_int number_of_samples, const t_real phi_0, const t_real theta_0, const K &phi,
    const K &theta, const Vector<t_real> &u, const Vector<t_real> &v, const Vector<t_real> &w,
    const Vector<t_complex> &weights, const t_real oversample_ratio = 2,
    const t_real oversample_ratio_image_domain = 2,
    const kernels::kernel kernel = kernels::kernel::kb, const t_uint Ju = 4, const t_uint Jw = 100,
    const t_uint Jl = 4, const t_uint Jm = 4,
    const operators::fftw_plan &ft_plan = operators::fftw_plan::measure,
    const bool uvw_stacking = false, const t_real L = 1, const t_real absolute_error = 1e-6,
    const t_real relative_error = 1e-6, const t_real coordinate_scaling = 1.,
    const t_real beam_l = 0, const t_real beam_m = 0) {
  t_real const w_mean = uvw_stacking ? w.array().mean() : 0.;
  t_real const v_mean = uvw_stacking ? v.array().mean() : 0.;
  t_real const u_mean = uvw_stacking ? u.array().mean() : 0.;

  const t_real M = L;
  const t_real dl_temp = std::min({0.25 / ((u.array() - u_mean).cwiseAbs().maxCoeff()), L / 128});
  const t_real dm_temp = std::min({0.25 / ((v.array() - v_mean).cwiseAbs().maxCoeff()), M / 128});
  const t_real dl = std::min(dl_temp, dm_temp);
  const t_real dm = dl;
  const t_int imsizex = std::floor(L / dl);
  const t_int imsizey = std::floor(M / dm);
  const t_real du = widefield::dl2du(dl, imsizex, oversample_ratio);
  const t_real dv = widefield::dl2du(dm, imsizey, oversample_ratio);
  PURIFY_MEDIUM_LOG("dl x dm : {} x {} ", dl, dm);
  PURIFY_MEDIUM_LOG("du x dv : {} x {} ", du, dv);
  PURIFY_MEDIUM_LOG("FoV (width, height): {} x {} (L x M)", imsizex * dl, imsizey * dm);
  PURIFY_MEDIUM_LOG("FoV (width, height): {} x {} (deg x deg)",
                    std::asin(imsizex * dl / 2.) * 2. * 180. / constant::pi,
                    std::asin(imsizey * dm / 2.) * 2. * 180. / constant::pi);
  PURIFY_MEDIUM_LOG("Primary Beam (width, height): {} x {} (deg x deg)",
                    std::asin(beam_l / 2.) * 2. * 180. / constant::pi,
                    std::asin(beam_m / 2.) * 2. * 180. / constant::pi);
  PURIFY_MEDIUM_LOG("Number of visibilities: {}", u.size());
  PURIFY_MEDIUM_LOG("Maximum u value is {} lambda, sphere needs to be sampled at dl = {}.",
                    u.cwiseAbs().maxCoeff(), std::min(0.5 / u.cwiseAbs().maxCoeff(), 1.));
  PURIFY_MEDIUM_LOG("Maximum v value is {} lambda, sphere needs to be sampled at dm = {}.",
                    v.cwiseAbs().maxCoeff(), std::min(0.5 / v.cwiseAbs().maxCoeff(), 1.));
  PURIFY_MEDIUM_LOG("Maximum w value is {} lambda, sphere needs to be sampled at dn = {}.",
                    w.cwiseAbs().maxCoeff(), std::min(0.5 / w.cwiseAbs().maxCoeff(), 1.));
  if (uvw_stacking) {
    PURIFY_MEDIUM_LOG("Mean, u: {}, +/- {}", u_mean, (u.maxCoeff() - u.minCoeff()) * 0.5);
    PURIFY_MEDIUM_LOG("Mean, v: {}, +/- {}", v_mean, (v.maxCoeff() - v.minCoeff()) * 0.5);
    PURIFY_MEDIUM_LOG("Mean, w: {}, +/- {}", w_mean, (w.maxCoeff() - w.minCoeff()) * 0.5);
  }

  auto const uvkernels = purify::create_radial_ftkernel(kernel, Ju, oversample_ratio);
  const std::function<t_real(t_real)> &kerneluv = std::get<1>(uvkernels);
  const std::function<t_real(t_real)> &ftkerneluv = std::get<0>(uvkernels);

  auto const lmkernels =
      purify::create_kernels(kernel, Jl, Jm, imsizey, imsizex, oversample_ratio_image_domain);
  const std::function<t_real(t_real)> &kernell = std::get<0>(lmkernels);
  const std::function<t_real(t_real)> &kernelm = std::get<1>(lmkernels);
  const std::function<t_real(t_real)> &ftkernell = std::get<2>(lmkernels);
  const std::function<t_real(t_real)> &ftkernelm = std::get<3>(lmkernels);

  const t_complex I(0., 1.);
  std::function<t_complex(t_real, t_real)> dde = [I, u_mean, v_mean, w_mean, imsizex, imsizey,
                                                  oversample_ratio, oversample_ratio_image_domain,
                                                  beam_l, beam_m](const t_real l, const t_real m) {
    return std::exp(-2 * constant::pi * I *
                    (u_mean * l + v_mean * m + w_mean * (std::sqrt(1. - l * l - m * m) - 1.))) /
           std::sqrt(1. - l * l - m * m) * (((l * l + m * m) < 1.) ? 1. : 0.) *
           std::sqrt(imsizex * imsizey) * oversample_ratio * oversample_ratio_image_domain *
           std::pow(boost::math::sinc_pi(beam_l * l * constant::pi), 2) *
           std::pow(boost::math::sinc_pi(beam_m * m * constant::pi), 2);
  };

  PURIFY_LOW_LOG("Constructing Spherical Resampling Operator: P");
  sopt::OperatorFunction<T> directP, indirectP;
  std::tie(directP, indirectP) = init_mask_and_resample_operator_2d<T, K>(
      number_of_samples, phi_0, theta_0, phi, theta, imsizey, imsizex,
      oversample_ratio_image_domain, dl, dm, kernell, kernelm, Jl, Jm, dde, coordinate_scaling);

  sopt::OperatorFunction<T> directZFZ, indirectZFZ;
  std::tie(directZFZ, indirectZFZ) =
      base_padding_and_FFT_2d<T>(ftkerneluv, ftkernell, ftkernelm, imsizey, imsizex,
                                 oversample_ratio, oversample_ratio_image_domain, ft_plan);
  PURIFY_LOW_LOG("Constructing Weighting and Gridding Operators: WG");
  sopt::OperatorFunction<T> directG, indirectG;
  std::tie(directG, indirectG) = purify::operators::init_gridding_matrix_2d<T>(
      (u.array() - u_mean) / du, (v.array() - v_mean) / dv, w.array() - w_mean, weights, imsizey,
      imsizex, oversample_ratio, ftkerneluv, kerneluv, Ju, Jw, du, dv, absolute_error,
      relative_error, dde_type::wkernel_radial);

  const auto direct = sopt::chained_operators<T>(directG, directZFZ, directP);
  const auto indirect = sopt::chained_operators<T>(indirectP, indirectZFZ, indirectG);
  PURIFY_LOW_LOG("Finished consturction of Φ.");
  return std::make_tuple(direct, indirect);
}
#ifdef PURIFY_MPI
//! all to all operator that degrids from the upsampled plane to the uv domain (no w-projection,
//! only w-stacking)
template <class T, class K>
std::tuple<sopt::OperatorFunction<T>, sopt::OperatorFunction<T>>
base_plane_degrid_all_to_all_operator(
    const sopt::mpi::Communicator &comm, const std::vector<t_int> &image_index,
    const std::vector<std::tuple<t_real, t_real, t_real>> &uvw_stacks,
    const t_int number_of_samples, const t_real phi_0, const t_real theta_0, const K &phi,
    const K &theta, const Vector<t_real> &u, const Vector<t_real> &v, const Vector<t_real> &w,
    const Vector<t_complex> &weights, const t_real oversample_ratio = 2,
    const t_real oversample_ratio_image_domain = 2,
    const kernels::kernel kernel = kernels::kernel::kb, const t_uint Ju = 4, const t_uint Jv = 4,
    const t_uint Jl = 4, const t_uint Jm = 4,
    const operators::fftw_plan &ft_plan = operators::fftw_plan::measure,
    const bool uvw_stacking = false, const t_real L = 1, const t_real M = 1,
    const t_real coordinate_scaling = 1., const t_real beam_l = 0, const t_real beam_m = 0) {
  if (image_index.size() != u.size())
    throw std::runtime_error("Number of stack index is not the same as measurements.");
  bool on_the_fly = true;
  t_real const w_mean = uvw_stacking ? std::get<2>(uvw_stacks.at(comm.rank())) : 0.;
  t_real const v_mean = uvw_stacking ? std::get<1>(uvw_stacks.at(comm.rank())) : 0.;
  t_real const u_mean = uvw_stacking ? std::get<0>(uvw_stacks.at(comm.rank())) : 0.;

  Vector<t_real> u_shifted = Vector<t_real>::Zero(u.size());
  Vector<t_real> v_shifted = Vector<t_real>::Zero(v.size());
  if (uvw_stacking)
    for (t_int index = 0; index < u.size(); index++) {
      u_shifted(index) = u(index) - std::get<0>(uvw_stacks.at(image_index.at(index)));
      v_shifted(index) = v(index) - std::get<1>(uvw_stacks.at(image_index.at(index)));
    }
  else {
    u_shifted = u;
    v_shifted = v;
  }
  const t_real dl =
      std::min({0.25 / comm.all_reduce<t_real>(u_shifted.cwiseAbs().maxCoeff(), MPI_MAX), L / 128});
  const t_real dm =
      std::min({0.25 / comm.all_reduce<t_real>(v_shifted.cwiseAbs().maxCoeff(), MPI_MAX), M / 128});
  const t_int imsizex = std::floor(L / dl);
  const t_int imsizey = std::floor(M / dm);
  const t_real du = widefield::dl2du(dl, imsizex, oversample_ratio);
  const t_real dv = widefield::dl2du(dm, imsizey, oversample_ratio);
  if (comm.is_root()) {
    PURIFY_MEDIUM_LOG("dl x dm : {} x {} ", dl, dm);
    PURIFY_MEDIUM_LOG("du x dv : {} x {} ", du, dv);
    PURIFY_MEDIUM_LOG("FoV (width, height): {} x {} (L x M)", imsizex * dl, imsizey * dm);
    PURIFY_MEDIUM_LOG("FoV (width, height): {} x {} (deg x deg)",
                      std::asin(imsizex * dl / 2.) * 2. * 180. / constant::pi,
                      std::asin(imsizey * dm / 2.) * 2. * 180. / constant::pi);
    PURIFY_MEDIUM_LOG("Primary Beam (width, height): {} x {} (deg x deg)",
                      std::asin(beam_l / 2 / 2.) * 2. * 180. / constant::pi,
                      std::asin(beam_m / 2. / 2) * 2. * 180. / constant::pi);
  }
  PURIFY_MEDIUM_LOG("Number of visibilities: {}", u.size());
  PURIFY_MEDIUM_LOG("Node {} has stack u_mean = {}, v_mean = {}, w_mean = {}", comm.rank(), u_mean,
                    v_mean, w_mean);
  auto const uvkernels = purify::create_kernels(kernel, Ju, Jv, imsizey, imsizex, oversample_ratio);
  const std::function<t_real(t_real)> &kernelu = std::get<0>(uvkernels);
  const std::function<t_real(t_real)> &kernelv = std::get<1>(uvkernels);
  const std::function<t_real(t_real)> &ftkernelu = std::get<2>(uvkernels);
  const std::function<t_real(t_real)> &ftkernelv = std::get<3>(uvkernels);

  auto const lmkernels =
      purify::create_kernels(kernel, Jl, Jm, imsizey, imsizex, oversample_ratio_image_domain);
  const std::function<t_real(t_real)> &kernell = std::get<0>(lmkernels);
  const std::function<t_real(t_real)> &kernelm = std::get<1>(lmkernels);
  const std::function<t_real(t_real)> &ftkernell = std::get<2>(lmkernels);
  const std::function<t_real(t_real)> &ftkernelm = std::get<3>(lmkernels);

  const t_complex I(0., 1.);
  std::function<t_complex(t_real, t_real)> dde = [I, u_mean, v_mean, w_mean, imsizex, imsizey,
                                                  oversample_ratio, oversample_ratio_image_domain,
                                                  beam_m, beam_l](const t_real l, const t_real m) {
    return std::exp(-2 * constant::pi * I *
                    (u_mean * l + v_mean * m + w_mean * (std::sqrt(1. - l * l - m * m) - 1.))) /
           std::sqrt(1. - l * l - m * m) * (((l * l + m * m) < 1.) ? 1. : 0.) *
           std::pow(boost::math::sinc_pi(beam_l * l * constant::pi), 2) *
           std::pow(boost::math::sinc_pi(beam_m * m * constant::pi), 2) *
           std::sqrt(imsizex * imsizey) * oversample_ratio * oversample_ratio_image_domain;
  };

  PURIFY_LOW_LOG("Constructing Spherical Resampling Operator: P");
  const auto P = init_mask_and_resample_operator_2d<T, K>(
      number_of_samples, phi_0, theta_0, phi, theta, imsizey, imsizex,
      oversample_ratio_image_domain, dl, dm, kernell, kernelm, Jl, Jm, dde, coordinate_scaling);
  const auto ZFZ =
      base_padding_and_FFT_2d<T>(ftkernelu, ftkernelv, ftkernell, ftkernelm, imsizey, imsizex,
                                 oversample_ratio, oversample_ratio_image_domain, ft_plan);

  PURIFY_LOW_LOG("Constructing Weighting and Gridding Operators: WG");
  const auto G = (on_the_fly)
                     ? purify::operators::init_on_the_fly_gridding_matrix_2d<T>(
                           comm, comm.size(), image_index, u_shifted / du, v_shifted / dv, weights,
                           imsizey, imsizex, oversample_ratio, kernelu, Ju, Ju * 1e5)
                     : purify::operators::init_gridding_matrix_2d<T>(
                           u_shifted / du, v_shifted / dv, weights, imsizey, imsizex,
                           oversample_ratio, kernelv, kernelu, Ju, Jv);
  PURIFY_LOW_LOG("Finished consturction of Φ.");
  return std::make_tuple(
      sopt::chained_operators<T>(std::get<0>(G), std::get<0>(ZFZ), std::get<0>(P)),
      sopt::chained_operators<T>(std::get<1>(P), std::get<1>(ZFZ), std::get<1>(G)));
}
//! all to all operator that degrids from the upsampled plane to the uv domain (with w-projection
//! and w-stacking)
template <class T, class K>
std::tuple<sopt::OperatorFunction<T>, sopt::OperatorFunction<T>>
base_plane_degrid_wproj_all_to_all_operator(
    const sopt::mpi::Communicator &comm, const std::vector<t_int> &image_index,
    const std::vector<t_real> &w_stacks, const t_int number_of_samples, const t_real phi_0,
    const t_real theta_0, const K &phi, const K &theta, const Vector<t_real> &u,
    const Vector<t_real> &v, const Vector<t_real> &w, const Vector<t_complex> &weights,
    const t_real oversample_ratio = 2, const t_real oversample_ratio_image_domain = 2,
    const kernels::kernel kernel = kernels::kernel::kb, const t_uint Ju = 4, const t_uint Jw = 100,
    const t_uint Jl = 4, const t_uint Jm = 4,
    const operators::fftw_plan &ft_plan = operators::fftw_plan::measure,
    const bool uvw_stacking = false, const t_real L = 1, const t_real absolute_error = 1e-6,
    const t_real relative_error = 1e-6, const t_real coordinate_scaling = 1.,
    const t_real beam_l = 0, const t_real beam_m = 0) {
  t_real const w_mean = uvw_stacking ? w_stacks.at(comm.rank()) : 0.;
  t_real const v_mean = uvw_stacking ? v.array().mean() : 0.;
  t_real const u_mean = uvw_stacking ? u.array().mean() : 0.;

  const t_real M = L;
  const t_real dl_temp = std::min({0.25 / ((u.array() - u_mean).cwiseAbs().maxCoeff()), L / 128});
  const t_real dm_temp = std::min({0.25 / ((v.array() - v_mean).cwiseAbs().maxCoeff()), M / 128});
  const t_real dl = std::max(dl_temp, dm_temp);
  const t_real dm = dl;
  const t_int imsizex = std::floor(L / dl);
  const t_int imsizey = std::floor(M / dm);
  const t_real du = widefield::dl2du(dl, imsizex, oversample_ratio);
  const t_real dv = widefield::dl2du(dm, imsizey, oversample_ratio);
  PURIFY_MEDIUM_LOG("dl x dm : {} x {} ", dl, dm);
  PURIFY_MEDIUM_LOG("du x dv : {} x {} ", du, dv);
  PURIFY_MEDIUM_LOG("FoV (width, height): {} x {} (L x M)", imsizex * dl, imsizey * dm);
  PURIFY_MEDIUM_LOG("FoV (width, height): {} x {} (deg x deg)",
                    std::asin(imsizex * dl / 2.) * 2. * 180. / constant::pi,
                    std::asin(imsizey * dm / 2.) * 2. * 180. / constant::pi);
  PURIFY_MEDIUM_LOG("Primary Beam (width, height): {} x {} (deg x deg)",
                    std::asin(beam_l / 2.) * 2. * 180. / constant::pi,
                    std::asin(beam_m / 2.) * 2. * 180. / constant::pi);
  PURIFY_MEDIUM_LOG("Number of visibilities: {}", u.size());
  PURIFY_MEDIUM_LOG("Maximum u value is {} lambda, sphere needs to be sampled at dl = {}.",
                    u.cwiseAbs().maxCoeff(), std::min(0.25 / u.cwiseAbs().maxCoeff(), 1.));
  PURIFY_MEDIUM_LOG("Maximum v value is {} lambda, sphere needs to be sampled at dm = {}.",
                    v.cwiseAbs().maxCoeff(), std::min(0.25 / v.cwiseAbs().maxCoeff(), 1.));
  PURIFY_MEDIUM_LOG("Maximum w value is {} lambda, sphere needs to be sampled at dn = {}.",
                    w.cwiseAbs().maxCoeff(), std::min(0.25 / w.cwiseAbs().maxCoeff(), 1.));
  if (uvw_stacking) {
    PURIFY_MEDIUM_LOG("Mean, u: {}, +/- {}", u_mean, (u.maxCoeff() - u.minCoeff()) * 0.5);
    PURIFY_MEDIUM_LOG("Mean, v: {}, +/- {}", v_mean, (v.maxCoeff() - v.minCoeff()) * 0.5);
    PURIFY_MEDIUM_LOG("Mean, w: {}, +/- {}", w_mean, (w.maxCoeff() - w.minCoeff()) * 0.5);
  }

  auto const uvkernels = purify::create_radial_ftkernel(kernel, Ju, oversample_ratio);
  const std::function<t_real(t_real)> &kerneluv = std::get<1>(uvkernels);
  const std::function<t_real(t_real)> &ftkerneluv = std::get<0>(uvkernels);

  auto const lmkernels =
      purify::create_kernels(kernel, Jl, Jm, imsizey, imsizex, oversample_ratio_image_domain);
  const std::function<t_real(t_real)> &kernell = std::get<0>(lmkernels);
  const std::function<t_real(t_real)> &kernelm = std::get<1>(lmkernels);
  const std::function<t_real(t_real)> &ftkernell = std::get<2>(lmkernels);
  const std::function<t_real(t_real)> &ftkernelm = std::get<3>(lmkernels);

  const t_complex I(0., 1.);
  std::function<t_complex(t_real, t_real)> dde = [I, u_mean, v_mean, w_mean, imsizex, imsizey,
                                                  oversample_ratio, oversample_ratio_image_domain,
                                                  beam_l, beam_m](const t_real l, const t_real m) {
    return std::exp(-2 * constant::pi * I *
                    (u_mean * l + v_mean * m + w_mean * (std::sqrt(1. - l * l - m * m) - 1.))) /
           std::sqrt(1. - l * l - m * m) * (((l * l + m * m) < 1.) ? 1. : 0.) *
           std::sqrt(imsizex * imsizey) * oversample_ratio * oversample_ratio_image_domain *
           boost::math::sinc_pi(beam_l * l * constant::pi) *
           boost::math::sinc_pi(beam_m * m * constant::pi);
  };

  PURIFY_LOW_LOG("Constructing Spherical Resampling Operator: P");
  sopt::OperatorFunction<T> directP, indirectP;
  std::tie(directP, indirectP) = init_mask_and_resample_operator_2d<T, K>(
      number_of_samples, phi_0, theta_0, phi, theta, imsizey, imsizex,
      oversample_ratio_image_domain, dl, dm, kernell, kernelm, Jl, Jm, dde, coordinate_scaling);

  sopt::OperatorFunction<T> directZFZ, indirectZFZ;
  std::tie(directZFZ, indirectZFZ) =
      base_padding_and_FFT_2d<T>(ftkerneluv, ftkernell, ftkernelm, imsizey, imsizex,
                                 oversample_ratio, oversample_ratio_image_domain, ft_plan);
  PURIFY_LOW_LOG("Constructing Weighting and Gridding Operators: WG");
  const t_uint number_of_images = comm.size();
  if (std::any_of(image_index.begin(), image_index.end(), [&number_of_images](int index) {
        return index < 0 or index > (number_of_images - 1);
      }))
    throw std::runtime_error("Image index is out of bounds");
  const t_int local_grid_size =
      std::floor(imsizey * oversample_ratio) * std::floor(imsizex * oversample_ratio);
  sopt::OperatorFunction<T> directG, indirectG;
  std::tie(directG, indirectG) =
      purify::operators::init_gridding_matrix_2d_all_to_all<T, std::int64_t>(
          comm, static_cast<std::int64_t>(local_grid_size),
          static_cast<std::int64_t>(comm.rank()) * static_cast<std::int64_t>(local_grid_size),
          number_of_images, image_index,
          (uvw_stacking) ? w_stacks : std::vector<t_real>(comm.size(), 0.),
          (u.array() - u_mean) / du, (v.array() - v_mean) / dv, w, weights, imsizey, imsizex,
          oversample_ratio, ftkerneluv, kerneluv, Ju, Jw, du, dv, absolute_error, relative_error,
          dde_type::wkernel_radial);

  const auto direct = sopt::chained_operators<T>(directG, directZFZ, directP);
  const auto indirect = sopt::chained_operators<T>(indirectP, indirectZFZ, indirectG);
  PURIFY_LOW_LOG("Finished consturction of Φ.");
  return std::make_tuple(direct, indirect);
}

#endif
namespace measurement_operator {

//! measurement operator that degrids from the upsampled plane to the uv domain (no w-projection,
//! only w-stacking)
template <class T, class K>
std::shared_ptr<sopt::LinearTransform<T>> planar_degrid_operator(
    const t_int number_of_samples, const t_real phi_0, const t_real theta_0, const K &phi,
    const K &theta, const utilities::vis_params &uv_data, const t_real oversample_ratio = 2,
    const t_real oversample_ratio_image_domain = 2,
    const kernels::kernel kernel = kernels::kernel::kb, const t_uint Ju = 4, const t_uint Jv = 4,
    const t_uint Jl = 4, const t_uint Jm = 4,
    const operators::fftw_plan &ft_plan = operators::fftw_plan::measure,
    const bool uvw_stacking = false, const t_real L = 1, const t_real M = 1,
    const t_real beam_l = 0, const t_real beam_m = 0) {
  if (uv_data.units != utilities::vis_units::lambda)
    throw std::runtime_error("Units for spherical imaging must be in lambda");
  const auto m_op = base_plane_degrid_operator<T, K>(
      number_of_samples, phi_0, theta_0, phi, theta, uv_data.u, uv_data.v, uv_data.w,
      uv_data.weights, oversample_ratio, oversample_ratio_image_domain, kernel, Ju, Jv, Jl, Jm,
      ft_plan, uvw_stacking, L, M, 1, beam_l, beam_m);

  const std::array<t_int, 3> outsize = {0, 1, static_cast<t_int>(uv_data.u.size())};
  const std::array<t_int, 3> insize = {0, 1, number_of_samples};

  return std::make_shared<sopt::LinearTransform<Vector<t_complex>>>(std::get<0>(m_op), outsize,
                                                                    std::get<1>(m_op), insize);
}
//! operator that degrids from the upsampled plane to the uvw domain (w-projection with
//! w-stacking)
template <class T, class K>
std::shared_ptr<sopt::LinearTransform<T>> nonplanar_degrid_wproj_operator(
    const t_int number_of_samples, const t_real phi_0, const t_real theta_0, const K &phi,
    const K &theta, const utilities::vis_params &uv_data, const t_real oversample_ratio = 2,
    const t_real oversample_ratio_image_domain = 2,
    const kernels::kernel kernel = kernels::kernel::kb, const t_uint Ju = 4, const t_uint Jw = 100,
    const t_uint Jl = 4, const t_uint Jm = 4,
    const operators::fftw_plan &ft_plan = operators::fftw_plan::measure,
    const bool uvw_stacking = false, const t_real L = 1, const t_real absolute_error = 1e-6,
    const t_real relative_error = 1e-6, const t_real beam_l = 0, const t_real beam_m = 0) {
  if (uv_data.units != utilities::vis_units::lambda)
    throw std::runtime_error("Units for spherical imaging must be in lambda");
  const auto m_op = base_plane_degrid_wproj_operator<T, K>(
      number_of_samples, phi_0, theta_0, phi, theta, uv_data.u, uv_data.v, uv_data.w,
      uv_data.weights, oversample_ratio, oversample_ratio_image_domain, kernel, Ju, Jw, Jl, Jm,
      ft_plan, uvw_stacking, L, absolute_error, relative_error, 1, beam_l, beam_m);

  const std::array<t_int, 3> outsize = {0, 1, static_cast<t_int>(uv_data.u.size())};
  const std::array<t_int, 3> insize = {0, 1, number_of_samples};

  return std::make_shared<sopt::LinearTransform<Vector<t_complex>>>(std::get<0>(m_op), outsize,
                                                                    std::get<1>(m_op), insize);
}
#ifdef PURIFY_MPI
//! Returns linear transform that is the weighted degridding operator with mpi all sum all
template <class T, class K>
std::shared_ptr<sopt::LinearTransform<T>> nonplanar_degrid_wproj_operator(
    const sopt::mpi::Communicator &comm, const t_int number_of_samples, const t_real phi_0,
    const t_real theta_0, const K &phi, const K &theta, const utilities::vis_params &uv_data,
    const t_real oversample_ratio = 2, const t_real oversample_ratio_image_domain = 2,
    const kernels::kernel kernel = kernels::kernel::kb, const t_uint Ju = 4, const t_uint Jw = 100,
    const t_uint Jl = 4, const t_uint Jm = 4,
    const operators::fftw_plan &ft_plan = operators::fftw_plan::measure,
    const bool uvw_stacking = false, const t_real L = 1, const t_real absolute_error = 1e-6,
    const t_real relative_error = 1e-6) {
  if (uv_data.units != utilities::vis_units::lambda)
    throw std::runtime_error("Units for spherical imaging must be in lambda");
  const auto m_op = base_plane_degrid_wproj_operator<T, K>(
      number_of_samples, phi_0, theta_0, phi, theta, uv_data.u, uv_data.v, uv_data.w,
      uv_data.weights, oversample_ratio, oversample_ratio_image_domain, kernel, Ju, Jw, Jl, Jm,
      ft_plan, uvw_stacking, L, absolute_error, relative_error);
  const auto allsumall = purify::operators::init_all_sum_all<T>(comm);
  const auto &direct = std::get<0>(m_op);
  const auto &indirect = sopt::chained_operators<T>(allsumall, std::get<1>(m_op));

  const std::array<t_int, 3> outsize = {0, 1, static_cast<t_int>(uv_data.u.size())};
  const std::array<t_int, 3> insize = {0, 1, number_of_samples};

  return std::make_shared<sopt::LinearTransform<Vector<t_complex>>>(direct, outsize, indirect,
                                                                    insize);
}
//! Returns linear transform that is the weighted degridding operator with mpi all sum all (for
//! planar array)
template <class T, class K>
std::shared_ptr<sopt::LinearTransform<T>> planar_degrid_operator(
    const sopt::mpi::Communicator &comm, const t_int number_of_samples, const t_real phi_0,
    const t_real theta_0, const K &phi, const K &theta, const utilities::vis_params &uv_data,
    const t_real oversample_ratio = 2, const t_real oversample_ratio_image_domain = 2,
    const kernels::kernel kernel = kernels::kernel::kb, const t_uint Ju = 4, const t_uint Jv = 4,
    const t_uint Jl = 4, const t_uint Jm = 4,
    const operators::fftw_plan &ft_plan = operators::fftw_plan::measure,
    const bool uvw_stacking = false, const t_real L = 1, const t_real M = 1,
    const t_real beam_l = 0., const t_real beam_m = 0.) {
  if (uv_data.units != utilities::vis_units::lambda)
    throw std::runtime_error("Units for spherical imaging must be in lambda");
  const auto &m_op = base_plane_degrid_operator<T, K>(
      number_of_samples, phi_0, theta_0, phi, theta, uv_data.u, uv_data.v, uv_data.w,
      uv_data.weights, oversample_ratio, oversample_ratio_image_domain, kernel, Ju, Jv, Jl, Jm,
      ft_plan, uvw_stacking, L, M, 1, beam_l, beam_m);

  const auto allsumall = purify::operators::init_all_sum_all<T>(comm);
  const auto &direct = std::get<0>(m_op);
  const auto &indirect = sopt::chained_operators<T>(allsumall, std::get<1>(m_op));

  const std::array<t_int, 3> outsize = {0, 1, static_cast<t_int>(uv_data.u.size())};
  const std::array<t_int, 3> insize = {0, 1, number_of_samples};

  return std::make_shared<sopt::LinearTransform<Vector<t_complex>>>(direct, outsize, indirect,
                                                                    insize);
}
//! The all to all w-stacking w-projection measurement operator
template <class T, class K>
std::shared_ptr<sopt::LinearTransform<T>> non_planar_degrid_wproj_all_to_all_operator(
    const sopt::mpi::Communicator &comm, const std::vector<t_int> &image_index,
    const std::vector<t_real> &w_stacks, const t_int number_of_samples, const t_real phi_0,
    const t_real theta_0, const K &phi, const K &theta, const utilities::vis_params &uv_data,
    const t_real oversample_ratio = 2, const t_real oversample_ratio_image_domain = 2,
    const kernels::kernel kernel = kernels::kernel::kb, const t_uint Ju = 4, const t_uint Jw = 100,
    const t_uint Jl = 4, const t_uint Jm = 4,
    const operators::fftw_plan &ft_plan = operators::fftw_plan::measure,
    const bool uvw_stacking = false, const t_real L = 1, const t_real absolute_error = 1e-6,
    const t_real relative_error = 1e-6, const t_real beam_l = 0, const t_real beam_m = 0) {
  if (uv_data.units != utilities::vis_units::lambda)
    throw std::runtime_error("Units for spherical imaging must be in lambda");
  const auto &m_op = base_plane_degrid_wproj_all_to_all_operator<T, K>(
      comm, image_index, w_stacks, number_of_samples, phi_0, theta_0, phi, theta, uv_data.u,
      uv_data.v, uv_data.w, uv_data.weights, oversample_ratio, oversample_ratio_image_domain,
      kernel, Ju, Jw, Jl, Jm, ft_plan, uvw_stacking, L, absolute_error, relative_error, 1, beam_l,
      beam_m);
  const std::array<t_int, 3> outsize = {0, 1, static_cast<t_int>(uv_data.u.size())};
  const std::array<t_int, 3> insize = {0, 1, number_of_samples};
  const auto allsumall = purify::operators::init_all_sum_all<T>(comm);
  const auto &direct = std::get<0>(m_op);
  const auto &indirect = sopt::chained_operators<T>(allsumall, std::get<1>(m_op));

  return std::make_shared<sopt::LinearTransform<Vector<t_complex>>>(direct, outsize, indirect,
                                                                    insize);
}
//! The all to all w-stacking w-projection measurement operator
template <class T, class K>
std::shared_ptr<sopt::LinearTransform<T>> planar_degrid_all_to_all_operator(
    const sopt::mpi::Communicator &comm, const std::vector<t_int> &image_index,
    const std::vector<std::tuple<t_real, t_real, t_real>> &uvw_stacks,
    const t_int number_of_samples, const t_real phi_0, const t_real theta_0, const K &phi,
    const K &theta, const utilities::vis_params &uv_data, const t_real oversample_ratio = 2,
    const t_real oversample_ratio_image_domain = 2,
    const kernels::kernel kernel = kernels::kernel::kb, const t_uint Ju = 4, const t_uint Jv = 4,
    const t_uint Jl = 4, const t_uint Jm = 4,
    const operators::fftw_plan &ft_plan = operators::fftw_plan::measure,
    const bool uvw_stacking = false, const t_real L = 1, const t_real M = 1,
    const t_real beam_l = 0, const t_real beam_m = 0) {
  if (uv_data.units != utilities::vis_units::lambda)
    throw std::runtime_error("Units for spherical imaging must be in lambda");
  const auto &m_op = base_plane_degrid_all_to_all_operator<T, K>(
      comm, image_index, uvw_stacks, number_of_samples, phi_0, theta_0, phi, theta, uv_data.u,
      uv_data.v, uv_data.w, uv_data.weights, oversample_ratio, oversample_ratio_image_domain,
      kernel, Ju, Jv, Jl, Jm, ft_plan, uvw_stacking, L, M, 1, beam_l, beam_m);

  const std::array<t_int, 3> outsize = {0, 1, static_cast<t_int>(uv_data.u.size())};
  const std::array<t_int, 3> insize = {0, 1, number_of_samples};
  const auto allsumall = purify::operators::init_all_sum_all<T>(comm);
  const auto &direct = std::get<0>(m_op);
  const auto &indirect = sopt::chained_operators<T>(allsumall, std::get<1>(m_op));

  return std::make_shared<sopt::LinearTransform<Vector<t_complex>>>(direct, outsize, indirect,
                                                                    insize);
}
#endif
}  // namespace measurement_operator
}  // namespace spherical_resample
}  // namespace purify
#endif
