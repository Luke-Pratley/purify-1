#include "purify/types.h"
#include <array>
#include <memory>
#include <random>
#include <boost/math/special_functions/erf.hpp>
#include "purify/cimg.h"
#include "purify/directories.h"
#include "purify/distribute.h"
#include "purify/logging.h"
#include "purify/mpi_utilities.h"
#include "purify/pfitsio.h"
#include "purify/read_measurements.h"
#include "purify/spherical_resample.h"
#include "purify/utilities.h"
#include <sopt/credible_region.h>
#include <sopt/imaging_padmm.h>
#include <sopt/power_method.h>
#include <sopt/relative_variation.h>
#include <sopt/utilities.h>
#include <sopt/wavelets.h>
#include <sopt/wavelets/sara.h>

#include "purify/algorithm_factory.h"
#include "purify/wavelet_operator_factory.h"
using namespace purify;
using namespace purify::notinstalled;

int main(int nargs, char const **args) {
  sopt::logging::initialize();
  purify::logging::initialize();
  sopt::logging::set_level("debug");
  purify::logging::set_level("debug");
  auto const session = sopt::mpi::init(nargs, args);
  auto const comm = sopt::mpi::Communicator::World();

  const std::vector<std::string> &file_names = std::vector<std::string>{"vela_small.uvfits"};
  const std::string &outfile_fits = "sphere_sol.fits";
  const std::string &residual_fits = "sphere_res.fits";
  const std::string &dirtyfile = "sphere_dirty.fits";
  const std::string &psffile = "sphere_psf.fits";

  const t_real L = 1.9;
  const t_real max_w = 0.;  // lambda
  const t_real snr = 30;

  const t_real phi_0 = 0. * constant::pi / 180.;
  const t_real theta_0 = 90. * constant::pi / 180.;
  const t_int max_ell = 2048;

  t_uint const imsizex = max_ell;
  t_uint const imsizey = max_ell;
  const t_int num_phi = imsizex;
  const t_int num_theta = imsizey;

  const t_int number_of_samples = num_phi * num_theta;
  const t_int Jl = 6;
  const t_int Jm = 6;
  const t_int Ju = 4;
  const t_int Jv = 4;
  const t_int Jw = 256;
  const t_real oversample_ratio_image_domain = 2;
  const t_real oversample_ratio = 2;
  const bool uvw_stacking = true;
  const kernels::kernel kernel = kernels::kernel::kb;
  const operators::fftw_plan ft_plan = operators::fftw_plan::measure;
  utilities::vis_params uv_data;
  std::vector<t_int> image_index;
  std::vector<t_real> w_stacks;
  t_real alpha = 0.;
  t_real beta = 0.;
  t_real gamma = 0.;
  {
    uv_data =
        read_measurements::read_measurements(file_names, comm, distribute::plan::w_term, true);
    t_int flag_size = 0;
    for (t_int i = 0; i < uv_data.size(); i++)
      if (std::sqrt(std::pow(uv_data.u(i), 2) + std::pow(uv_data.v(i), 2) +
                    std::pow(uv_data.w(i), 2)) > 0)
        if (std::sqrt(std::pow(uv_data.u(i), 2) + std::pow(uv_data.v(i), 2) +
                      std::pow(uv_data.w(i), 2)) < 300)
          flag_size++;
    Vector<t_real> u = Vector<t_real>::Zero(flag_size);
    Vector<t_real> v = Vector<t_real>::Zero(flag_size);
    Vector<t_real> w = Vector<t_real>::Zero(flag_size);
    Vector<t_complex> vis = Vector<t_complex>::Zero(flag_size);
    Vector<t_complex> weights = Vector<t_complex>::Zero(flag_size);

    t_int count = 0;
    for (t_int i = 0; i < uv_data.size(); i++)
      if (std::sqrt(std::pow(uv_data.u(i), 2) + std::pow(uv_data.v(i), 2) +
                    std::pow(uv_data.w(i), 2)) > 0)
        if (std::sqrt(std::pow(uv_data.u(i), 2) + std::pow(uv_data.v(i), 2) +
                      std::pow(uv_data.w(i), 2)) < 300) {
          u(count) = uv_data.u(i);
          v(count) = uv_data.v(i);
          w(count) = uv_data.w(i);
          vis(count) = uv_data.vis(i);
          weights(count) = uv_data.weights(i);
          count++;
        }

    const auto euler_angles = spherical_resample::find_prefered_direction(u, v, w, comm);
    alpha = std::get<0>(euler_angles);
    beta = std::get<1>(euler_angles);
    gamma = std::get<2>(euler_angles);

    uv_data.u = spherical_resample::calculate_rotated_l(u, v, w, alpha, beta, gamma);
    uv_data.v = spherical_resample::calculate_rotated_m(u, v, w, alpha, beta, gamma);
    uv_data.w = -spherical_resample::calculate_rotated_n(u, v, w, alpha, beta, gamma);
    uv_data.vis =
        vis.array() * Eigen::exp(-2 * constant::pi * t_complex(0, 1.) * (w - uv_data.w).array());
    uv_data.weights = weights;
    uv_data = utilities::conjugate_w(uv_data);
    uv_data = utilities::distribute_all_to_all(uv_data, comm);
    const t_real norm = std::sqrt(
        comm.all_sum_all((uv_data.weights.real().array() * uv_data.weights.real().array()).sum()) /
        comm.all_sum_all(uv_data.size()));
    // normalise weights
    uv_data.weights = uv_data.weights / norm;
    uv_data.vis = uv_data.vis.array() * uv_data.weights.array();
    PURIFY_DEBUG("Node {} has {} visibilities after flaggging long baselines.", comm.rank(),
                 uv_data.size());
  }
  const auto theta = [num_theta, num_phi](const t_int k) -> t_real {
    return (utilities::ind2row(k, num_theta, num_phi)) * constant::pi / num_theta;
  };
  const auto phi = [num_phi, num_theta](const t_int k) -> t_real {
    return (utilities::ind2col(k, num_theta, num_phi)) * constant::pi / num_phi +
           constant::pi * 0.5;
  };

  const t_real sigma = 320;

  const auto uvw_stacks = distribute::uv_all_to_all(comm, uv_data.u, uv_data.v, uv_data.w);
  std::shared_ptr<sopt::LinearTransform<Vector<t_complex>>> const measurements_transform =
      spherical_resample::measurement_operator::planar_degrid_all_to_all_operator<
          Vector<t_complex>, std::function<t_real(t_int)>>(
          comm, std::get<0>(uvw_stacks), std::get<1>(uvw_stacks), number_of_samples, phi_0, theta_0,
          phi, theta, uv_data, oversample_ratio, oversample_ratio_image_domain, kernel, Ju, Jv, Jl,
          Jm, ft_plan, uvw_stacking, L, L, 0., 0.);
  {
    const Vector<t_complex> dmap = measurements_transform->adjoint() * uv_data.vis;
    const Image<t_complex> dmap_image =
        Image<t_complex>::Map(dmap.data(), imsizex, imsizey) / comm.all_sum_all(uv_data.size());
    if (comm.is_root()) pfitsio::write2d(dmap_image.real(), dirtyfile);
    const Vector<t_complex> psf = measurements_transform->adjoint() * uv_data.weights;
    const Image<t_complex> psf_image =
        Image<t_complex>::Map(psf.data(), imsizex, imsizey) / comm.all_sum_all(uv_data.size());
    if (comm.is_root()) pfitsio::write2d(psf_image.real(), psffile);
  }
  // calculate norm
  const t_real op_norm = std::get<0>(sopt::algorithm::power_method<Vector<t_complex>>(
      *measurements_transform, 1000, 1e-3,
      comm.broadcast<Vector<t_complex>>(Vector<t_complex>::Random(imsizex * imsizey).eval())));
  // wavelet transform
  t_uint sara_size = 0.;
  std::vector<std::tuple<std::string, t_uint>> const sara{
      std::make_tuple("dirac", 1u), std::make_tuple("db1", 5u), std::make_tuple("db2", 5u),
      std::make_tuple("db3", 5u),   std::make_tuple("db4", 5u), std::make_tuple("db5", 5u),
      std::make_tuple("db6", 5u),   std::make_tuple("db7", 5u), std::make_tuple("db8", 5u)};
  auto const wavelets = factory::wavelet_operator_factory<Vector<t_complex>>(
      factory::distributed_wavelet_operator::mpi_sara, sara, imsizey, imsizex, sara_size);

  auto const algo = factory::fb_factory<sopt::algorithm::ImagingForwardBackward<t_complex>>(
      factory::algo_distribution::mpi_serial, measurements_transform, wavelets, uv_data, sigma,
      sigma * sigma * 0.5, 0, imsizey, imsizex, sara_size, 1000, true, true, false, 1e-5, 1e-5, 50,
      op_norm);
  // primaldual->l1_proximal_weights(l1_weights);
  auto const diagnostic = (*algo)();

  Image<t_complex> image = Image<t_complex>::Map(diagnostic.x.data(), imsizex, imsizey);
  if (comm.is_root()) pfitsio::write2d(image.real(), outfile_fits);
  Vector<t_complex> residuals = measurements_transform->adjoint() *
                                (uv_data.vis - ((*measurements_transform) * diagnostic.x));
  Image<t_complex> residual_image = Image<t_complex>::Map(residuals.data(), imsizex, imsizey);
  if (comm.is_root()) pfitsio::write2d(residual_image.real(), residual_fits);

  return 0;
}
