#include "purify/config.h"
#include "purify/types.h"
#include "purify/logging.h"
#include "purify/operators.h"
#include "purify/utilities.h"

#include "purify/directories.h"
#include "purify/kernels.h"
#include "purify/pfitsio.h"
#include "purify/spherical_resample.h"
using namespace purify;
using namespace purify::notinstalled;

using namespace purify;

int main(int nargs, char const** args) {
  purify::logging::initialize();
  purify::logging::set_level("debug");
#define ARGS_MACRO(NAME, ARGN, VALUE, TYPE)                                                  \
  auto const NAME = static_cast<TYPE>(                                                       \
      (nargs > ARGN) ? std::stod(static_cast<std::string>(args[ARGN])) / 180. * constant::pi \
                     : VALUE);

  ARGS_MACRO(phi_0, 1, 0., t_real)
  ARGS_MACRO(theta_0, 2, 0., t_real)
#undef ARGS_MACRO
#define ARGS_MACRO(NAME, ARGN, VALUE, TYPE) \
  auto const NAME =                         \
      static_cast<TYPE>((nargs > ARGN) ? std::stod(static_cast<std::string>(args[ARGN])) : VALUE);

  ARGS_MACRO(L, 3, 1., t_real)
  ARGS_MACRO(u_val, 4, 0., t_real)
  ARGS_MACRO(v_val, 5, 0., t_real)
  ARGS_MACRO(w_val, 6, 0., t_real)
  ARGS_MACRO(uvw_stacking, 7, false, bool)
  ARGS_MACRO(coordinate_scaling, 8, 1., t_real)
#undef ARGS_MACRO
  const t_real M = L;
  const t_int num_theta = 4096;
  const t_int num_phi = 2 * num_theta - 1;
  const t_int number_of_samples = num_phi * num_theta;
  const t_real beam_l = 0;
  const t_real beam_m = beam_l;
  const t_int Jl = 6;
  const t_int Jm = 6;
  const t_int Ju = 4;
  const t_int Jv = 4;
  const t_int Jw = 1000;
  const t_real oversample_ratio_image_domain = 1;
  const t_real oversample_ratio = 2;
  const kernels::kernel kernel = kernels::kernel::kb;
  const operators::fftw_plan ft_plan = operators::fftw_plan::estimate;

  const t_int num_vis = 1;

  const Vector<t_real> u = Vector<t_real>::Constant(num_vis, u_val);
  const Vector<t_real> v = Vector<t_real>::Constant(num_vis, v_val);
  const Vector<t_real> w = Vector<t_real>::Constant(num_vis, w_val);
  const Vector<t_complex> weights = Vector<t_complex>::Constant(num_vis, 1.);

  PURIFY_LOW_LOG("Resolution : {} x {} (deg x deg)", 360. / num_phi, 180. / num_theta);

  const auto theta = [num_theta](const t_int k) -> t_real {
    return utilities::ind2row(k, num_theta, num_phi) * constant::pi / num_theta;
  };
  const auto phi = [num_phi, num_theta](const t_int k) -> t_real {
    return utilities::ind2col(k, num_theta, num_phi) * 2 * constant::pi / num_phi;
  };
  //
  t_real const offset_dec = constant::pi * 0. / 180.;
  t_real const offset_ra = constant::pi * 0. / 180.;
  const auto measure_op =
      (w(0) > 0)
          ? spherical_resample::base_plane_degrid_wproj_operator<Vector<t_complex>,
                                                                 std::function<t_real(t_int)>>(
                number_of_samples, phi_0, theta_0, phi, theta, u, v, w, weights, oversample_ratio,
                oversample_ratio_image_domain, kernel, Ju, Jw, Jl, Jm, ft_plan, uvw_stacking, L,
                1e-8, 1e-8, coordinate_scaling, beam_l, beam_m)
          : spherical_resample::base_plane_degrid_operator<Vector<t_complex>,
                                                           std::function<t_real(t_int)>>(
                number_of_samples, phi_0, theta_0, phi, theta, u, v, w, weights, oversample_ratio,
                oversample_ratio_image_domain, kernel, Ju, Jv, Jl, Jm, ft_plan, uvw_stacking, L, L,
                coordinate_scaling, beam_l, beam_m);

  sopt::LinearTransform<Vector<t_complex>> const m_op = sopt::LinearTransform<Vector<t_complex>>(
      std::get<0>(measure_op), {0, 1, number_of_samples}, std::get<1>(measure_op), {0, 1, num_vis});
  const Vector<t_complex> output = m_op.adjoint() * (Vector<t_complex>::Ones(1));

  Vector<t_real> l = Vector<t_real>::Zero(number_of_samples);
  Vector<t_real> m = Vector<t_real>::Zero(number_of_samples);
  Vector<t_real> n = Vector<t_real>::Zero(number_of_samples);
  Vector<t_real> th = Vector<t_real>::Zero(number_of_samples);
  Vector<t_real> ph = Vector<t_real>::Zero(number_of_samples);
  Vector<t_complex> fourier_mode = Vector<t_complex>::Zero(number_of_samples);
  Vector<t_real> diff = Vector<t_real>::Zero(number_of_samples);
  for (t_int k = 0; k < number_of_samples; k++) {
    l(k) = spherical_resample::calculate_l(phi(k), theta(k), constant::pi / 2, theta_0, phi_0);
    m(k) = spherical_resample::calculate_m(phi(k), theta(k), constant::pi / 2, theta_0, phi_0);
    n(k) = spherical_resample::calculate_n(phi(k), theta(k), constant::pi / 2, theta_0, phi_0);
    ph(k) = phi(k);
    th(k) = theta(k);
  }

  for (t_int k = 0; k < number_of_samples; k++) {
    fourier_mode(k) = ((n(k) > 0) ? std::conj(std::exp(-2 * constant::pi * t_complex(0., 1.) *
                                                       (l(k) * u(0) * coordinate_scaling +
                                                        m(k) * v(0) * coordinate_scaling +
                                                        (n(k) - 1) * w(0) * coordinate_scaling))) /
                                        n(k)
                                  : 0.) *
                      std::pow(boost::math::sinc_pi(beam_l * l(k) * constant::pi), 2) *
                      std::pow(boost::math::sinc_pi(beam_m * m(k) * constant::pi), 2);
    if (std::abs(output(k)) > 0)
      diff(k) = 2. * std::abs(fourier_mode(k) - output(k)) /
                (std::abs(fourier_mode(k)) + std::abs(output(k)));
    else
      diff(k) = NAN;
  }

  // pfitsio::write2d(Image<t_real>::Map(l.data(), num_phi, num_theta), "l_coordinates.fits");
  //  pfitsio::write2d(Image<t_real>::Map(m.data(), num_phi, num_theta), "m_coordinates.fits");
  //  pfitsio::write2d(Image<t_real>::Map(n.data(), num_phi, num_theta), "n_coordinates.fits");
  //  pfitsio::write2d(Image<t_real>::Map(mask.data(), num_phi, num_theta),
  //  "mask_coordinates.fits");

  // output name
  const std::string& name = "fourier_mode_u_" + std::to_string(static_cast<int>(std::floor(u(0)))) +
                            "_v_" + std::to_string(static_cast<int>(std::floor(v(0)))) + "_w_" +
                            std::to_string(static_cast<int>(std::floor(w(0))));
  pfitsio::write2d(Image<t_real>::Map(diff.data(), num_phi, num_theta), name + "_diff.fits");

  pfitsio::write2d(Image<t_complex>::Map(output.data(), num_phi, num_theta).real(),
                   name + "_real_calculated.fits");
  pfitsio::write2d(Image<t_complex>::Map(output.data(), num_phi, num_theta).imag(),
                   name + "_imag_calculated.fits");
  // pfitsio::write2d(Image<t_complex>::Map(fourier_mode.data(), num_phi, num_theta).real(),
  //                 "fourier_mode_real_expected.fits");
  //  pfitsio::write2d(Image<t_complex>::Map(fourier_mode.data(), num_phi,
  //  num_theta).imag(),
  //                "fourier_mode_imag_expected.fits");
}
