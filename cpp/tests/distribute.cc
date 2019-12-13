#include "purify/distribute.h"
#include "purify/types.h"
#include "catch.hpp"
#include "purify/directories.h"
#include "purify/utilities.h"

using namespace purify;
using namespace purify::notinstalled;
TEST_CASE("Distribute") {
  // Test splitting up of data for separate nodes
  auto const uv_data = utilities::read_visibility(vla_filename("at166B.3C129.c0.vis"));
  std::printf("Distributing groups\n");
  t_int number_of_groups = 4;
  t_int number_of_vis = uv_data.u.size();
  std::vector<t_int> groups_equal = distribute::distribute_measurements(
      uv_data.u.segment(0, number_of_vis), uv_data.v.segment(0, number_of_vis),
      uv_data.w.segment(0, number_of_vis), number_of_groups, distribute::plan::equal, 1024);
  // Testing number of visiblities in groups adds to total
  CHECK(number_of_vis == groups_equal.size());
  for (t_int i = 0; i < groups_equal.size(); i++) {
    // checking nodes are in allowable values
    CHECK(groups_equal[i] >= 0);
    CHECK(groups_equal[i] < number_of_groups);
  }
  std::vector<t_int> groups_distance = distribute::distribute_measurements(
      uv_data.u.segment(0, number_of_vis), uv_data.v.segment(0, number_of_vis),
      uv_data.w.segment(0, number_of_vis), number_of_groups, distribute::plan::radial);
  // Testing number of visiblities in groups adds to total
  CHECK(number_of_vis == groups_distance.size());
  for (t_int i = 0; i < groups_distance.size(); i++) {
    // checking nodes are in allowable values
    CHECK(groups_distance[i] >= 0);
    CHECK(groups_distance[i] < number_of_groups);
  }
  std::vector<t_int> groups_noorder = distribute::distribute_measurements(
      uv_data.u.segment(0, number_of_vis), uv_data.v.segment(0, number_of_vis),
      uv_data.w.segment(0, number_of_vis), number_of_groups, distribute::plan::none);
  // Testing number of visiblities in groups adds to total
  CHECK(number_of_vis == groups_noorder.size());
  for (t_int i = 0; i < groups_noorder.size(); i++) {
    // checking nodes are in allowable values
    CHECK(groups_noorder[i] >= 0);
    CHECK(groups_noorder[i] < number_of_groups);
  }
  std::vector<t_int> groups_w_term = distribute::distribute_measurements(
      uv_data.u.segment(0, number_of_vis), uv_data.v.segment(0, number_of_vis),
      uv_data.w.segment(0, number_of_vis), number_of_groups, distribute::plan::w_term);
  // Testing number of visiblities in groups adds to total
  CHECK(number_of_vis == groups_distance.size());
  for (t_int i = 0; i < groups_distance.size(); i++) {
    // checking nodes are in allowable values
    CHECK(groups_distance[i] >= 0);
    CHECK(groups_distance[i] < number_of_groups);
  }
}
TEST_CASE("uv_stacking") {
  SECTION("4 groups") {
    const t_int groups = 4;
    const t_int M = 10;
    Vector<t_real> u = Vector<t_real>::Ones(M * groups);
    Vector<t_real> v = Vector<t_real>::Ones(M * groups);
    CHECK(u.size() == 40);
    u.segment(0, M) *= -1;
    v.segment(0, M) *= -1;

    u.segment(M, M) *= -1;
    v.segment(M, M) *= 1;

    u.segment(2 * M, M) *= 1;
    v.segment(2 * M, M) *= -1;

    u.segment(3 * M, M) *= 1;
    v.segment(3 * M, M) *= 1;
    const auto order = distribute::uv_distribution(u, v, groups);
    const Vector<t_int> result = Vector<t_int>::Map(order.data(), order.size());
    CAPTURE(result.segment(0, M))
    CAPTURE(result.segment(1 * M, M))
    CAPTURE(result.segment(2 * M, M))
    CAPTURE(result.segment(3 * M, M))
    CHECK(result.segment(0, M).isApprox(Vector<t_int>::Constant(M, 0)));
    CHECK(result.segment(M, M).isApprox(Vector<t_int>::Constant(M, 1)));
    CHECK(result.segment(2 * M, M).isApprox(Vector<t_int>::Constant(M, 2)));
    CHECK(result.segment(3 * M, M).isApprox(Vector<t_int>::Constant(M, 3)));
  }
  SECTION("16 groups") {
    const t_int groups = 16;
    const t_int M = 10;
    Vector<t_real> u = Vector<t_real>::Ones(M * groups);
    Vector<t_real> v = Vector<t_real>::Ones(M * groups);
    CHECK(u.size() == 160);
    const t_int u_g = std::sqrt(groups);
    const t_int v_g = std::sqrt(groups);
    for (t_int i = 0; i < u_g; i++)
      for (t_int j = 0; j < v_g; j++) {
        const t_int index = i * u_g + j;
        u.segment(index * M, M) *= i;
        v.segment(index * M, M) *= j;
      }

    const auto order = distribute::uv_distribution(u, v, groups);
    const Vector<t_int> result = Vector<t_int>::Map(order.data(), order.size());
    CHECK(result.size() == groups);
    for (t_int i = 0; i < u_g; i++)
      for (t_int j = 0; j < v_g; j++) {
        const t_int index = i * u_g + j;
        CAPTURE(result.segment(index * M, M))
        CHECK(result.segment(index * M, M).isApprox(Vector<t_int>::Constant(M, index)));
      }
  }
}
