#ifndef PURIFY_DISTRIBUTE_H
#define PURIFY_DISTRIBUTE_H
#include "purify/config.h"
#include <iostream>
#include <stdio.h>
#include <string>
#ifdef PURIFY_MPI
#include <sopt/mpi/communicator.h>
#endif
#include "purify/types.h"
#include "purify/logging.h"
#include "purify/uvw_utilities.h"

namespace purify {
namespace distribute {
enum class plan { none, equal, radial, w_term, uv_stack, u_term };
//! Distribute visiblities into groups
std::vector<t_int> distribute_measurements(Vector<t_real> const &u, Vector<t_real> const &v,
                                           Vector<t_real> const &w, t_int const number_of_nodes,
                                           distribute::plan const distribution_plan = plan::equal,
                                           t_int const grid_size = 128);

#ifdef PURIFY_MPI
inline std::vector<t_int> distribute_measurements(
    utilities::vis_params const &params, sopt::mpi::Communicator const &comm,
    distribute::plan const distribution_plan = plan::equal, t_int const grid_size = 128) {
  return distribute_measurements(params.u, params.v, params.w, comm.size(), distribution_plan,
                                 grid_size);
}
#endif
//! patition w terms using k-means
std::tuple<std::vector<t_int>, std::vector<t_real>> kmeans_algo(
    const Vector<t_real> &w, const t_int number_of_nodes, const t_int iters,
    const std::function<t_real(t_real)> &cost = [](t_real x) { return x * x; },
    const t_real rel_diff = 1e-3);
#ifdef PURIFY_MPI
//! use mpi to calculate uv stacks
std::tuple<std::vector<t_int>, std::vector<std::tuple<t_real, t_real, t_real>>> uv_all_to_all(
    sopt::mpi::Communicator const &comm, Vector<t_real> const &u, Vector<t_real> const &v,
    Vector<t_real> const &w);
//! use mpi to distribute uv_stacks
std::vector<t_int> uv_distribution(sopt::mpi::Communicator const &comm, Vector<t_real> const &u,
                                   Vector<t_real> const &v, const t_int nodes);
//! patition w terms using k-means over MPI
//! Details returns a tuple (indices for group, centre mean for each group)
std::tuple<std::vector<t_int>, std::vector<t_real>> kmeans_algo(
    const Vector<t_real> &w, const t_int number_of_nodes, const t_int iters,
    sopt::mpi::Communicator const &comm,
    const std::function<t_real(t_real)> &cost = [](t_real x) { return x * x; },
    const t_real rel_diff = 1e-3);
//! Indicies to evenly distribute kernel coefficients values across nodes
std::vector<t_int> w_support(Vector<t_real> const &w, const std::vector<t_int> &image_index,
                             const std::vector<t_real> &w_stacks, const t_real du,
                             const t_int min_support, const t_int max_support,
                             const t_real fill_relaxation, sopt::mpi::Communicator const &comm);
#endif
//! Distribute visibilities into nodes in order of w terms (useful for w-stacking)
Vector<t_int> w_distribution(Vector<t_real> const &u, const Vector<t_real> &v,
                             const Vector<t_real> &w);
Vector<t_int> w_distribution(Vector<t_real> const &w);
//! Distribute visiblities into nodes in order of distance from the centre
Vector<t_int> distance_distribution(Vector<t_real> const &u, Vector<t_real> const &v);
//! Distribute the visiblities into nodes in order of density
Vector<t_int> equal_distribution(Vector<t_real> const &u, Vector<t_real> const &v,
                                 t_int const grid_size);
//! Distribute the visiblities into nodes by making a grid
std::vector<t_int> uv_distribution(Vector<t_real> const &u, Vector<t_real> const &v,
                                   t_int const nodes);
std::vector<t_int> uv_distribution(Vector<t_real> const &u, Vector<t_real> const &v,
                                   const t_int nodes, const t_real u_min, const t_real u_max,
                                   const t_real v_min, const t_real v_max);
}  // namespace distribute
}  // namespace purify
#endif
