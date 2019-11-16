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

  const std::vector<std::string> &file_names =
      std::vector<std::string>{"vela_small.uvfits"};
  const std::string &outfile_fits = "sphere_sol.fits";
  const std::string &residual_fits = "sphere_res.fits";
  const std::string &dirtyfile = "sphere_dirty.fits";
  const t_real L = 0.5;
  const t_real max_w = 0.;  // lambda
  const t_real snr = 30;

  const t_real phi_0 = 0. * constant::pi / 180.;
  const t_real theta_0 = 90. * constant::pi / 180.;

  t_uint const imsizex = 1024;
  t_uint const imsizey = 512;
  const t_int num_phi = imsizex;
  const t_int num_theta = imsizey;

  const t_int number_of_samples = num_phi * num_theta;
  const t_int Jl = 4;
  const t_int Jm = 4;
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
  {
    uv_data =
        read_measurements::read_measurements(file_names, comm, distribute::plan::radial, true);
    uv_data = utilities::conjugate_w(uv_data);
    t_int const imsize = comm.all_reduce<t_real>(
        std::pow(
            2,
            std::ceil(std::log2(std::floor(
                L / std::min({0.25 / ((uv_data.u.array() - uv_data.u.mean()).cwiseAbs().maxCoeff()),
                              L / Jw * 2}))))),
        MPI_MAX);

    const t_real dl = L / imsize;

    const t_real du = widefield::dl2du(dl, imsize, oversample_ratio);
    uv_data = utilities::w_stacking(uv_data,
                                 comm, 100,
                                 [](t_real x ){return x * x;},
                                 1e-3);
      const t_real norm =
          std::sqrt(comm.all_sum_all(
                        (uv_data.weights.real().array() * uv_data.weights.real().array()).sum()) /
                    comm.all_sum_all(uv_data.size()));
      // normalise weights
      uv_data.weights = uv_data.weights / norm;
  }
  const auto theta = [num_theta, num_phi](const t_int k) -> t_real {
    return (utilities::ind2row(k, num_theta, num_phi)) * constant::pi / num_theta;
  };
  const auto phi = [num_phi, num_theta](const t_int k) -> t_real {
    return (utilities::ind2col(k, num_theta, num_phi)) * 2 * constant::pi / num_phi;
  };

  t_real sigma = 5;

  std::shared_ptr<sopt::LinearTransform<Vector<t_complex>>> const measurements_transform =
      spherical_resample::measurement_operator::planar_degrid_operator<
          Vector<t_complex>, std::function<t_real(t_int)>>(
          comm, number_of_samples, phi_0, theta_0, phi, theta, uv_data,
          oversample_ratio, oversample_ratio_image_domain, kernel, Ju, Jv, Jl, Jm, ft_plan,
          uvw_stacking, L, L, 0., 0.);
  const t_real op_norm = std::get<0>(sopt::algorithm::power_method<Vector<t_complex>>(
      *measurements_transform, 1000, 1e-3,
      comm.broadcast<Vector<t_complex>>(Vector<t_complex>::Random(imsizex * imsizey).eval())));
  Vector<t_complex> dmap = measurements_transform->adjoint() * uv_data.vis;
  Image<t_complex> dmap_image = Image<t_complex>::Map(dmap.data(), imsizex, imsizey)/comm.all_sum_all(uv_data.size());
  if (comm.is_root()) pfitsio::write2d(dmap_image.real(), dirtyfile);
  // wavelet transform
  t_uint sara_size = 0.;
  std::vector<std::tuple<std::string, t_uint>> const sara{std::make_tuple("dirac", 1u)};
  auto const wavelets = factory::wavelet_operator_factory<Vector<t_complex>>(
      factory::distributed_wavelet_operator::mpi_sara, sara, imsizey, imsizex, sara_size);

  auto const primaldual =
      factory::primaldual_factory<sopt::algorithm::ImagingPrimalDual<t_complex>>(
          factory::algo_distribution::mpi_serial, measurements_transform, wavelets, uv_data, sigma,
          imsizey, imsizex, sara_size, 500,
    true, true,
    1e-3, 1, op_norm);
  auto const diagnostic = (*primaldual)();

  assert(diagnostic.x.size() == all_sky_image.size());
  Image<t_complex> image = Image<t_complex>::Map(diagnostic.x.data(), imsizex, imsizey);
  if (comm.is_root()) pfitsio::write2d(image.real(), outfile_fits);
  Vector<t_complex> residuals = measurements_transform->adjoint() *
                                (uv_data.vis - ((*measurements_transform) * diagnostic.x));
  Image<t_complex> residual_image = Image<t_complex>::Map(residuals.data(), imsizex, imsizey);
  if (comm.is_root()) pfitsio::write2d(residual_image.real(), residual_fits);

  return 0;
}
