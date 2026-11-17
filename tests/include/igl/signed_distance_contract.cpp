// This file is part of libigl, a simple c++ geometry processing library.
//
// Contract tests for igl::signed_distance.
//
// The signed distance query bundles two logically separate things:
//   1. a purely geometric closest-point query (unsigned Euclidean distance d,
//      a closest face index I and a closest point C), and
//   2. a *signing strategy* (pseudo-normal, exact winding number, fast winding
//      number, ...) that multiplies d by a sign-like factor s.
//
// A closest point can be perfectly correct while the returned number is still
// not the Euclidean distance (|S| != d) or the inside/outside classification is
// wrong. These tests pin that contract down. All meshes are small procedural
// grids built in this file, so no external mesh data is required.
#include <test_common.h>
#include <igl/signed_distance.h>
#include <igl/winding_number.h>
#include <igl/fast_winding_number.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/barycentric_coordinates.h>
#include <igl/is_edge_manifold.h>

#include <cmath>
#include <limits>
#include <vector>

namespace
{
  // Use Eigen's row-major 1x3 type on purpose: in IGL_STATIC_LIBRARY builds
  // the single-point igl::winding_number / igl::fast_winding_number explicit
  // instantiations only exist for this storage order.
  using RowVector3d = Eigen::RowVector3d;

  // Tolerances. The exact (double precision) signing strategies are accurate to
  // machine precision on these simple meshes. The fast winding number runs in
  // single precision with a Barnes-Hut expansion: near well-resolved face
  // centers it reaches ~1e-7, while near edges/vertices or thin nested shells
  // the measured sign-factor error reaches a few 1e-3, i.e. ~1e-2 on a
  // unit-scale distance. The FWN tolerances are therefore kept separate, and
  // sign checks always stay strict: a classification flip must never be
  // hidden behind a magnitude tolerance.
  constexpr double EPS_EXACT      = 1e-12;
  constexpr double EPS_FWN        = 1e-7;
  constexpr double EPS_FWN_LOOSE  = 1e-2;

  struct SDResult
  {
    Eigen::VectorXd S;
    Eigen::VectorXi I;
    Eigen::MatrixXd C;
  };

  SDResult run_signed_distance(
    const Eigen::MatrixXd & P,
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const igl::SignedDistanceType type)
  {
    SDResult R;
    Eigen::MatrixXd N;
    igl::signed_distance(P,V,F,type,R.S,R.I,R.C,N);
    return R;
  }

  // Build an axis-aligned cube centered at `center` with half-side h. The 12
  // triangles are consistently oriented outward, so (V,F) is a closed,
  // edge-manifold genus-0 solid with an unambiguous inside and outside.
  //
  // Face layout (per block of two triangles):
  //   0,1  z = -h   2,3  z = +h   4,5  y = -h   6,7  y = +h
  //   8,9  x = -h  10,11 x = +h
  void make_cube(
    const double h,
    const RowVector3d & center,
    Eigen::MatrixXd & V,
    Eigen::MatrixXi & F)
  {
    V.resize(8,3);
    V <<
    -h,-h,-h,
     h,-h,-h,
     h, h,-h,
    -h, h,-h,
    -h,-h, h,
     h,-h, h,
     h, h, h,
    -h, h, h;
    V.rowwise() += center;
    F.resize(12,3);
    F <<
    0,3,2,  0,2,1,  // bottom  (normal -z)
    4,5,6,  4,6,7,  // top     (normal +z)
    0,1,5,  0,5,4,  // y = -h  (normal -y)
    3,7,6,  3,6,2,  // y = +h  (normal +y)
    0,4,7,  0,7,3,  // x = -h  (normal -x)
    1,2,6,  1,6,5;  // x = +h  (normal +x)
  }

  void make_cube(const double h, Eigen::MatrixXd & V, Eigen::MatrixXi & F)
  {
    make_cube(h, RowVector3d(0,0,0), V, F);
  }

  // Reverse every triangle while keeping the vertex positions untouched. This
  // is a pure orientation change: every unsigned distance must stay identical.
  Eigen::MatrixXi reverse_faces(const Eigen::MatrixXi & F)
  {
    Eigen::MatrixXi R(F.rows(),3);
    for(int f = 0;f<F.rows();f++)
    {
      R(f,0) = F(f,0);
      R(f,1) = F(f,2);
      R(f,2) = F(f,1);
    }
    return R;
  }

  // Re-order triangles (cyclic row permutation) and rotate each triangle's
  // vertices by one position. Geometry and orientation are preserved, but face
  // indices and the local vertex ordering change.
  Eigen::MatrixXi reorder_faces(const Eigen::MatrixXi & F)
  {
    Eigen::MatrixXi R(F.rows(),3);
    for(int f = 0;f<F.rows();f++)
    {
      const int g = (f + 3) % F.rows();
      R(g,0) = F(f,1);
      R(g,1) = F(f,2);
      R(g,2) = F(f,0);
    }
    return R;
  }

  // Closest point on a triangle to p. Independent implementation of Ericson's
  // barycentric-region projection (Real-Time Collision Detection, sec. 5.1.5),
  // deliberately not using igl's AABB so that the test's reference distance is
  // computed by a different code path than the one under test. Note that in the
  // interior region the ab coefficient is vb/denom (not vc/denom): each v*
  // numerator names the opposite vertex's barycentric coordinate.
  RowVector3d closest_point_on_triangle(
    const RowVector3d & p,
    const RowVector3d & a,
    const RowVector3d & b,
    const RowVector3d & c)
  {
    const RowVector3d ab = b-a;
    const RowVector3d ac = c-a;
    const RowVector3d ap = p-a;
    const double d1 = ab.dot(ap);
    const double d2 = ac.dot(ap);
    if(d1 <= 0.0 && d2 <= 0.0) return a;

    const RowVector3d bp = p-b;
    const double d3 = ab.dot(bp);
    const double d4 = ac.dot(bp);
    if(d3 >= 0.0 && d4 <= d3) return b;

    const RowVector3d cp = p-c;
    const double d5 = ab.dot(cp);
    const double d6 = ac.dot(cp);
    if(d6 >= 0.0 && d5 <= d6) return c;

    const double vc = d1*d4 - d3*d2;
    if(vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
      const double v = d1/(d1-d3);
      return a + v*ab;
    }

    const double vb = d5*d2 - d1*d6;
    if(vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
      const double w = d2/(d2-d6);
      return a + w*ac;
    }

    const double va = d3*d6 - d5*d4;
    if(va <= 0.0 && (d4-d3) >= 0.0 && (d5-d6) >= 0.0)
    {
      const double w = (d4-d3)/((d4-d3)+(d5-d6));
      return b + w*(c-b);
    }

    const double denom = 1.0/(va+vb+vc);
    const double v = vb*denom;
    const double w = vc*denom;
    return a + ab*v + ac*w;
  }

  // Brute-force closest point over the whole triangle soup.
  void brute_force_closest(
    const RowVector3d & p,
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    double & sqr_d,
    int & face,
    RowVector3d & closest)
  {
    sqr_d = std::numeric_limits<double>::infinity();
    face = -1;
    for(int f = 0;f<F.rows();f++)
    {
      const RowVector3d c = closest_point_on_triangle(
        p, V.row(F(f,0)), V.row(F(f,1)), V.row(F(f,2)));
      const double d2 = (p-c).squaredNorm();
      if(d2 < sqr_d)
      {
        sqr_d = d2;
        face = f;
        closest = c;
      }
    }
  }

  // True iff c lies on triangle F(f) (barycentric coordinates in [0,1]).
  bool point_is_on_face(
    const RowVector3d & c,
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const int f,
    const double tol = 1e-10)
  {
    if(f < 0 || f >= F.rows()) return false;
    RowVector3d A = V.row(F(f,0));
    RowVector3d B = V.row(F(f,1));
    RowVector3d D = V.row(F(f,2));
    RowVector3d P = c;
    RowVector3d L;
    igl::barycentric_coordinates(P,A,B,D,L);
    if(std::abs(L.sum()-1.0) > tol) return false;
    for(int k = 0;k<3;k++)
    {
      if(L(k) < -tol || L(k) > 1.0+tol) return false;
    }
    const RowVector3d reconstructed =
      L(0)*A + L(1)*B + L(2)*D;
    return (reconstructed-c).norm() <= tol;
  }

  // Contract checks shared by every query on an ordinary mesh:
  //   * |S| is the Euclidean distance to the soup,
  //   * C is the closest point computed independently by brute force,
  //   * I is a valid face index and C actually lies on that face.
  void require_geometry_contract(
    const Eigen::MatrixXd & P,
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const SDResult & R,
    const double tol)
  {
    REQUIRE(R.S.size() == P.rows());
    REQUIRE(R.I.size() == P.rows());
    REQUIRE(R.C.rows() == P.rows());
    for(int p = 0;p<P.rows();p++)
    {
      CAPTURE(p);
      double d2_ref;
      int f_ref;
      RowVector3d c_ref;
      brute_force_closest(P.row(p),V,F,d2_ref,f_ref,c_ref);

      REQUIRE(R.I(p) >= 0);
      REQUIRE(R.I(p) <  F.rows());
      test_common::assert_near(
        Eigen::RowVector3d(R.C.row(p)), c_ref, tol);
      REQUIRE(std::abs(std::abs(R.S(p)) - std::sqrt(d2_ref)) < tol);
      // The reported magnitude must equal the distance to the reported point.
      REQUIRE((P.row(p)-R.C.row(p)).norm() - std::abs(R.S(p)) < tol);
      REQUIRE(std::abs(R.S(p)) - (P.row(p)-R.C.row(p)).norm() < tol);
      REQUIRE(point_is_on_face(R.C.row(p),V,F,R.I(p),tol));
    }
  }

  // Fast winding number at a single query point.
  double fwn_at(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const RowVector3d & q)
  {
    igl::FastWindingNumberBVH bvh;
    igl::fast_winding_number(V.cast<float>().eval(), F, 2, bvh);
    Eigen::VectorXf W;
    Eigen::MatrixXf Q(1,3);
    Q = q.cast<float>();
    igl::fast_winding_number(bvh, 2.0f, Q, W);
    return double(W(0));
  }

  const std::vector<igl::SignedDistanceType> & all_3d_types()
  {
    static const std::vector<igl::SignedDistanceType> types =
    {
      igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
      igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
      igl::SIGNED_DISTANCE_TYPE_DEFAULT,
      igl::SIGNED_DISTANCE_TYPE_UNSIGNED,
      igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
    };
    return types;
  }

  double tol_for(const igl::SignedDistanceType type)
  {
    return type == igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER ?
      EPS_FWN_LOOSE : EPS_EXACT;
  }
}

// ---------------------------------------------------------------------------
// 1. Oriented closed mesh: distance magnitude and closest point must agree for
//    every signing strategy, and the signed variants must classify the known
//    interior/exterior points correctly.
// ---------------------------------------------------------------------------
TEST_CASE("signed_distance_contract: closed_oriented_cube", "[igl]")
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  make_cube(1.0, V, F);

  // Query, exact unsigned distance, expected inside (negative sign).
  struct Probe { RowVector3d p; double d; bool inside; };
  const std::vector<Probe> probes =
  {
    {RowVector3d( 2.0, 0.0, 0.0), 1.0,              false}, // face center
    {RowVector3d( 0.5, 0.0, 0.0), 0.5,              true }, // face center
    {RowVector3d( 2.0, 2.0, 0.0), std::sqrt(2.0),   false}, // over edge
    {RowVector3d( 2.0, 2.0, 2.0), std::sqrt(3.0),   false}, // over vertex
    {RowVector3d( 0.5, 2.0, 0.0), 1.0,              false}, // edge projection
    {RowVector3d( 0.9, 0.8, 0.7), 0.1,              true }, // near corner, unique face
    {RowVector3d(-1.3, 0.2,-0.4), 0.3,              false}, // offset face
  };

  Eigen::MatrixXd P(probes.size(),3);
  Eigen::VectorXd Dref(probes.size());
  Eigen::VectorXi Inside(probes.size());
  for(size_t k = 0;k<probes.size();k++)
  {
    P.row(k) = probes[k].p;
    Dref(k)  = probes[k].d;
    Inside(k) = probes[k].inside ? 1 : 0;
  }

  for(const auto type : all_3d_types())
  {
    CAPTURE(type);
    const SDResult R = run_signed_distance(P,V,F,type);
    require_geometry_contract(P,V,F,R,tol_for(type));

    for(int k = 0;k<P.rows();k++)
    {
      CAPTURE(k);
      // Magnitude is the geometric distance regardless of the signer.
      REQUIRE(std::abs(std::abs(R.S(k)) - Dref(k)) < tol_for(type));
      if(type == igl::SIGNED_DISTANCE_TYPE_UNSIGNED)
      {
        // The unsigned strategy must never emit a sign.
        REQUIRE(R.S(k) >= 0.0);
      }
      else
      {
        // Strict sign checks, independent of the magnitude tolerance: a
        // classification flip cannot be explained away by approximation.
        if(Inside(k)) REQUIRE(R.S(k) < 0.0);
        else          REQUIRE(R.S(k) > 0.0);
      }
    }
  }

  // A point exactly on the surface is a valid zero distance, not NaN.
  Eigen::MatrixXd Psurf(1,3);
  Psurf << 1.0, 0.3, 0.4;
  for(const auto type : all_3d_types())
  {
    const SDResult R = run_signed_distance(Psurf,V,F,type);
    REQUIRE_FALSE(std::isnan(R.S(0)));
    REQUIRE(std::abs(R.S(0)) < tol_for(type));
    test_common::assert_near(R.C, Psurf, tol_for(type));
    REQUIRE(point_is_on_face(R.C.row(0),V,F,R.I(0),tol_for(type)));
  }

  // The two FWN tolerances answer different questions. EPS_FWN_LOOSE measures
  // the approximation error against the true Euclidean distance: on this mesh
  // it reaches ~1e-3 because of the float32 Barnes-Hut expansion. EPS_FWN
  // instead measures internal consistency: signed_distance's per-point FWN
  // path and the standalone batch FWN query run the same order-2 BVH over the
  // same data, so their answers must agree to float-scale precision even when
  // that shared answer is only approximate.
  {
    Eigen::MatrixXd Pfc(2,3);
    Pfc << 2.0, 0.0, 0.0, 0.5, 0.0, 0.0;
    Eigen::VectorXd Dfc(2);
    Dfc << 1.0, 0.5;
    const SDResult R = run_signed_distance(Pfc,V,F,
      igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER);
    for(int k = 0;k<2;k++)
    {
      // Geometric truth: only met to the loose approximation tolerance.
      REQUIRE(std::abs(std::abs(R.S(k)) - Dfc(k)) < EPS_FWN_LOOSE);
      // Cross-path consistency: recompute the sign factor through the
      // standalone batch API and reconstruct the expected signed value.
      const double w = std::abs(fwn_at(V,F,Pfc.row(k)));
      const double d = (Pfc.row(k)-R.C.row(k)).norm();
      REQUIRE(std::abs(R.S(k) - (1.0-2.0*w)*d) < EPS_FWN);
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Open mesh (a single triangle): the closest point is trivially correct,
//    but winding-number signing does not produce the Euclidean distance. Pins
//    down what each strategy means on a mesh without an interior.
// ---------------------------------------------------------------------------
TEST_CASE("signed_distance_contract: open_triangle_signing_strategies", "[igl]")
{
  Eigen::MatrixXd V(3,3);
  V <<
    0,0,0,
    1,0,0,
    0,1,0;
  Eigen::MatrixXi F(1,3);
  F << 0,1,2;  // normal +z

  Eigen::MatrixXd P(2,3);
  P <<
    0.2, 0.2,  1.0,  // above the triangle
    0.2, 0.2, -1.0;  // below the triangle
  const double d = 1.0;

  // The generalized winding number of an open triangle is a fractional solid
  // angle, not an integer 0/1 membership. The two sides get opposite signs and
  // neither reaches the unit magnitude of a closed solid.
  const double w_above = igl::winding_number(V,F,RowVector3d(P.row(0)));
  const double w_below = igl::winding_number(V,F,RowVector3d(P.row(1)));
  REQUIRE(std::abs(w_above) >  1e-12);
  REQUIRE(std::abs(w_above) <  0.5);
  REQUIRE(std::abs(w_below) >  1e-12);
  REQUIRE(std::abs(w_below) <  0.5);
  REQUIRE(w_above*w_below < 0.0);  // opposite sides, opposite solid-angle signs

  // Exact winding number / default: S = (1 - 2 w) d, so |S| is strictly
  // smaller than the Euclidean distance. Taking abs(S) as an unsigned distance
  // is therefore wrong, and this assertion fails if anybody "repairs" it.
  for(const auto type :
      {igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
       igl::SIGNED_DISTANCE_TYPE_DEFAULT})
  {
    const SDResult R = run_signed_distance(P,V,F,type);
    for(int k = 0;k<2;k++)
    {
      const double w = igl::winding_number(V,F,RowVector3d(P.row(k)));
      const double s_expected = 1.0 - 2.0*w;
      CAPTURE(type); CAPTURE(k);
      REQUIRE(std::abs(R.S(k) - s_expected*d) < EPS_EXACT);
      REQUIRE(std::abs(std::abs(R.S(k)) - d) > 1e-10);
      // The closest point is still geometrically exact even though the
      // returned number is not the Euclidean distance.
      test_common::assert_near(
        R.C.row(k), P.row(k) + (k == 0 ?
          RowVector3d(0,0,-d) : RowVector3d(0,0,d)), EPS_EXACT);
      REQUIRE(R.I(k) == 0);
    }
  }

  // Fast winding number approximates the same fractional solid angle; the
  // implementation signs with 1 - 2|w|, so above/below get the same sign here
  // (orientation of an open soup is intentionally discarded by |.|).
  {
    const double wf = fwn_at(V,F,P.row(0));
    REQUIRE(std::abs(wf - w_above) < 1e-5);
    const SDResult R =
      run_signed_distance(P,V,F,igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER);
    for(int k = 0;k<2;k++)
    {
      const double wf_k = std::abs(fwn_at(V,F,P.row(k)));
      REQUIRE(std::abs(R.S(k) - (1.0-2.0*wf_k)*d) < EPS_FWN_LOOSE);
      REQUIRE(R.S(k) > 0.0);  // |w| < 0.5 on both sides
    }
  }

  // Pseudo-normal signing is locally well defined even on an open soup:
  // above the plane is outside (+), below is inside (-), magnitude exact.
  {
    const SDResult R =
      run_signed_distance(P,V,F,igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL);
    REQUIRE(std::abs(R.S(0) -  d) < EPS_EXACT);
    REQUIRE(std::abs(R.S(1) - -d) < EPS_EXACT);
  }

  // Unsigned: orientation-agnostic, exact Euclidean distance on both sides.
  {
    const SDResult R =
      run_signed_distance(P,V,F,igl::SIGNED_DISTANCE_TYPE_UNSIGNED);
    REQUIRE(std::abs(R.S(0) - d) < EPS_EXACT);
    REQUIRE(std::abs(R.S(1) - d) < EPS_EXACT);
  }
}

// ---------------------------------------------------------------------------
// 3. Nested outward-oriented shells: winding number reaches integer levels
//    0, 1, 2. In the inner cavity the signed value is -3*d, whose absolute
//    value is not the Euclidean distance, while the pseudo-normal strategy
//    only sees the nearest shell and cannot represent the second layer.
// ---------------------------------------------------------------------------
TEST_CASE("signed_distance_contract: nested_shells_winding_levels", "[igl]")
{
  Eigen::MatrixXd Vo, Vi;
  Eigen::MatrixXi Fo, Fi;
  make_cube(1.0, Vo, Fo);
  make_cube(0.5, Vi, Fi);

  Eigen::MatrixXd V(Vo.rows()+Vi.rows(),3);
  V << Vo, Vi;
  Eigen::MatrixXi F(Fo.rows()+Fi.rows(),3);
  F << Fo, (Fi.array() + Vo.rows()).matrix();

  Eigen::MatrixXd P(3,3);
  P <<
    2.0,  0.0, 0.0,  // outside both shells        w = 0
    0.75, 0.0, 0.0,  // between the shells          w = 1
    0.25, 0.0, 0.0;  // inside the inner shell      w = 2
  const Eigen::Vector3d Dref(1.0, 0.25, 0.25);
  const Eigen::Vector3d Wref(0.0, 1.0, 2.0);

  for(int k = 0;k<3;k++)
  {
    REQUIRE(std::abs(
      igl::winding_number(V,F,RowVector3d(P.row(k))) - Wref(k))
            < EPS_EXACT);
  }

  for(const auto type :
      {igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
       igl::SIGNED_DISTANCE_TYPE_DEFAULT,
       igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER})
  {
    const SDResult R = run_signed_distance(P,V,F,type);
    const double tol = tol_for(type);
    for(int k = 0;k<3;k++)
    {
      CAPTURE(type); CAPTURE(k);
      const double w =
        type == igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER ?
        std::abs(fwn_at(V,F,P.row(k))) : Wref(k);
      // The signed value follows the winding-level formula exactly.
      REQUIRE(std::abs(R.S(k) - (1.0-2.0*w)*Dref(k)) < tol);
      // The geometric part of the contract is independent of the winding level.
      REQUIRE((P.row(k)-R.C.row(k)).norm() - Dref(k) < tol);
      REQUIRE(Dref(k) - (P.row(k)-R.C.row(k)).norm() < tol);
    }
    // w = 0 and w = 1 levels: |S| coincides with the Euclidean distance.
    REQUIRE(std::abs(std::abs(R.S(0)) - Dref(0)) < tol);
    REQUIRE(std::abs(std::abs(R.S(1)) - Dref(1)) < tol);
    // The core cavity (w = 2): signed value -0.75 vs Euclidean distance 0.25.
    // An implementation returning abs(S) as the distance is off by 0.5 here.
    REQUIRE(std::abs(std::abs(R.S(2)) - Dref(2)) > 0.4);
    // Middle and core queries are closest to the inner shell (faces 12..23).
    REQUIRE(R.I(1) >= Fo.rows());
    REQUIRE(R.I(2) >= Fo.rows());
    REQUIRE(point_is_on_face(R.C.row(1),V,F,R.I(1),tol));
    REQUIRE(point_is_on_face(R.C.row(2),V,F,R.I(2),tol));
  }

  // Pseudo-normal: only the nearest, locally consistent shell contributes. The
  // middle point is outside the inner shell, so it is classified positive; the
  // core point is inside it and classified negative. Both magnitudes equal the
  // Euclidean distance exactly; the outer shell is invisible to this strategy.
  {
    const SDResult R =
      run_signed_distance(P,V,F,igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL);
    REQUIRE(R.S(0) > 0.0);                 // outside the outer shell
    REQUIRE(std::abs(R.S(0) - Dref(0)) < EPS_EXACT);
    REQUIRE(R.S(1) > 0.0);                 // outside the inner shell
    REQUIRE(std::abs(R.S(1) - Dref(1)) < EPS_EXACT);
    REQUIRE(R.S(2) < 0.0);                 // inside the inner shell
    REQUIRE(std::abs(std::abs(R.S(2)) - Dref(2)) < EPS_EXACT);
  }
}

// ---------------------------------------------------------------------------
// 4. Reversing orientation is not a geometry change: the unsigned distance and
//    closest point never move, but every winding-based result changes. With an
//    inward orientation the interior winding is -1 instead of +1, so the sign
//    factor becomes 1 - 2(-1) = 3 and the returned magnitude is no longer the
//    Euclidean distance. The fast winding number is orientation-blind (it signs
//    via |w|), which is separate, source-supported behavior.
// ---------------------------------------------------------------------------
TEST_CASE("signed_distance_contract: orientation_reversal_vs_geometry_change",
  "[igl]")
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  make_cube(1.0, V, F);
  const Eigen::MatrixXi F_rev = reverse_faces(F);

  Eigen::MatrixXd P(2,3);
  P <<
    0.3, 0.0, 0.0,  // interior point: d = 0.7
    2.0, 0.0, 0.0;  // exterior point: d = 1.0, w = 0 either way

  // Exact winding number flips sign at the interior point.
  REQUIRE(std::abs(igl::winding_number(V,F,RowVector3d(P.row(0))) -  1.0)
    < EPS_EXACT);
  REQUIRE(std::abs(igl::winding_number(V,F_rev,RowVector3d(P.row(0))) - -1.0)
    < EPS_EXACT);

  for(const auto type :
      {igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
       igl::SIGNED_DISTANCE_TYPE_DEFAULT})
  {
    const SDResult R     = run_signed_distance(P,V,F,    type);
    const SDResult R_rev = run_signed_distance(P,V,F_rev,type);
    CAPTURE(type);
    // Interior point: s goes from -1 (w = +1) to +3 (w = -1). The sign flips
    // and the magnitudes differ by a factor of 3; neither signed value is
    // obtained from the other by just negating it.
    REQUIRE(R.S(0) < 0.0);
    REQUIRE(R_rev.S(0) > 0.0);
    REQUIRE(std::abs(R.S(0) + 0.7) < EPS_EXACT);
    REQUIRE(std::abs(R_rev.S(0) - 2.1) < EPS_EXACT);
    // The closest point is unchanged by a pure orientation change.
    test_common::assert_near(R.C, R_rev.C, EPS_EXACT);
    // Exterior point: w = 0 on either orientation, so the value is unchanged.
    REQUIRE(R.S(1) > 0.0);
    REQUIRE(R_rev.S(1) > 0.0);
    REQUIRE(std::abs(R.S(1) - R_rev.S(1)) < EPS_EXACT);
  }

  // Unsigned distance is completely orientation-blind.
  {
    const SDResult R     = run_signed_distance(P,V,F,
      igl::SIGNED_DISTANCE_TYPE_UNSIGNED);
    const SDResult R_rev = run_signed_distance(P,V,F_rev,
      igl::SIGNED_DISTANCE_TYPE_UNSIGNED);
    test_common::assert_near(R.S, R_rev.S, EPS_EXACT);
  }

  // Fast winding number signs with 1 - 2|w|, so reversed orientation produces
  // the same signed distance. Contrast with the exact winding number above.
  for(int k = 0;k<2;k++)
  {
    const SDResult R     = run_signed_distance(P,V,F,
      igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER);
    const SDResult R_rev = run_signed_distance(P,V,F_rev,
      igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER);
    // Same |w| to float/Barnes-Hut accuracy; orientation is discarded by |.|.
    REQUIRE(std::abs(R.S(k) - R_rev.S(k)) < 1e-5);
  }

  // Real geometry change: a larger cube with the *same* orientation changes the
  // unsigned distance, unlike the orientation-only reversal above.
  Eigen::MatrixXd V_big;
  Eigen::MatrixXi F_big;
  make_cube(1.5, V_big, F_big);
  {
    const SDResult R     = run_signed_distance(P,V,
      F, igl::SIGNED_DISTANCE_TYPE_UNSIGNED);
    const SDResult R_big = run_signed_distance(P,V_big,
      F_big, igl::SIGNED_DISTANCE_TYPE_UNSIGNED);
    // Interior probe: 0.7 vs 1.2 to the x = +/-h face; exterior: 1.0 vs 0.5.
    REQUIRE(std::abs(R_big.S(0) - R.S(0)) > 0.4);
    REQUIRE(std::abs(R_big.S(1) - R.S(1)) > 0.4);
  }
}

// ---------------------------------------------------------------------------
// 5. Closest point moving from face interior to a shared edge and to a
//    vertex: distance magnitude stays compatible with the owning geometry.
//    Face indices may legitimately change under triangle re-ordering or
//    equidistant ties; the closest-point consistency check still has teeth.
// ---------------------------------------------------------------------------
TEST_CASE("signed_distance_contract: face_edge_vertex_closest_point_transitions",
  "[igl]")
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  make_cube(1.0, V, F);

  // Each probe targets a different region of the mesh.
  Eigen::MatrixXd P(4,3), Cexpect(4,3);
  P <<
    0.25, 2.0, 0.0,   // face interior (y = +1 face)
    2.0,  2.0, 0.25,  // edge (x = y = 1)
    2.0,  2.0, 2.0,   // vertex (1,1,1)
    0.5,  2.0, 0.5;   // exactly over the edge shared by two y = +1 triangles
  Cexpect <<
    0.25, 1.0, 0.0,
    1.0,  1.0, 0.25,
    1.0,  1.0, 1.0,
    0.5,  1.0, 0.5;

  for(const auto type : all_3d_types())
  {
    const SDResult R = run_signed_distance(P,V,F,type);
    const double tol = tol_for(type);
    for(int k = 0;k<4;k++)
    {
      CAPTURE(type); CAPTURE(k);
      test_common::assert_near(R.C.row(k), Cexpect.row(k), tol);
      REQUIRE(std::abs((P.row(k)-R.C.row(k)).norm() -
                       std::abs(R.S(k))) < tol);
      REQUIRE(point_is_on_face(R.C.row(k),V,F,R.I(k),tol));
    }
    // Last probe sits on a shared edge: either adjacent face index is legal.
    const bool adjacent =
      R.I(3) == 6 || R.I(3) == 7;  // the two y = +1 triangles
    REQUIRE(adjacent);
  }

  // Re-ordered triangulation: same geometry, rotated vertices, shifted rows.
  const Eigen::MatrixXi F2 = reorder_faces(F);
  for(const auto type : all_3d_types())
  {
    const SDResult R1 = run_signed_distance(P,V,F, type);
    const SDResult R2 = run_signed_distance(P,V,F2,type);
    const double tol = tol_for(type);
    for(int k = 0;k<4;k++)
    {
      CAPTURE(type); CAPTURE(k);
      // Distances and closest points are invariant; face indices are allowed
      // to differ, but every reported face must still own the closest point.
      test_common::assert_near(R1.C.row(k), R2.C.row(k), tol);
      REQUIRE(std::abs(R1.S(k) - R2.S(k)) < tol);
      REQUIRE(point_is_on_face(R2.C.row(k),V,F2,R2.I(k),tol));
    }
  }

  // Demonstrate that the on-face assertion can actually fail: a closest point
  // shifted off the surface must be rejected.
  {
    const SDResult R = run_signed_distance(P,V,F,
      igl::SIGNED_DISTANCE_TYPE_UNSIGNED);
    RowVector3d tampered = R.C.row(0);
    tampered(1) += 0.1;  // move it off the y = +1 support plane
    REQUIRE_FALSE(point_is_on_face(tampered,V,F,R.I(0),1e-10));
  }
}

// ---------------------------------------------------------------------------
// 6. Two triangles sharing an edge with an exactly equidistant query: either
//    face index is acceptable, but the closest point and distance are fixed.
//    A query uniquely inside one triangle keeps a deterministic index.
// ---------------------------------------------------------------------------
TEST_CASE("signed_distance_contract: equidistant_face_index_tie", "[igl]")
{
  // Unit-size quad split along the B-D diagonal.
  Eigen::MatrixXd V(4,3);
  V <<
    0,0,0,   // A
    2,0,0,   // B
    2,2,0,   // C
    0,2,0;   // D
  Eigen::MatrixXi F(2,3);
  F <<
    0,1,3,  // A B D (region x + y <= 2)
    1,2,3;  // B C D

  Eigen::MatrixXd P(3,3);
  P <<
    1.0, 1.0, 1.0,  // over the shared diagonal: both faces equidistant
    1.0, 1.0, 0.0,  // on the diagonal itself: valid zero distance
    0.5, 0.25,1.0;  // uniquely over triangle 0

  for(const auto type : all_3d_types())
  {
    const SDResult R = run_signed_distance(P,V,F,type);
    const double tol = tol_for(type);
    CAPTURE(type);
    const bool winding_based =
      type == igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER ||
      type == igl::SIGNED_DISTANCE_TYPE_DEFAULT ||
      type == igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER;

    // Tie: either face, fixed point. Face index and closest point are purely
    // geometric and must hold for every strategy.
    REQUIRE((R.I(0) == 0 || R.I(0) == 1));
    test_common::assert_near(R.C.row(0),
      RowVector3d(1,1,0), tol);
    if(!winding_based)
    {
      // Pseudo-normal / unsigned carry the exact Euclidean magnitude.
      REQUIRE(std::abs(std::abs(R.S(0)) - 1.0) < tol);
    }
    else
    {
      // Winding-based signing on an open soup: the fractional winding makes
      // |S| differ from the Euclidean distance while the geometry is exact.
      REQUIRE(std::abs((P.row(0)-R.C.row(0)).norm() - 1.0) < tol);
      REQUIRE(std::abs(std::abs(R.S(0)) - 1.0) > 1e-6);
    }
    REQUIRE(point_is_on_face(R.C.row(0),V,F,R.I(0),tol));

    // On-surface query is zero, never NaN (all sign factors vanish at d = 0).
    REQUIRE_FALSE(std::isnan(R.S(1)));
    REQUIRE(std::abs(R.S(1)) < tol);
    test_common::assert_near(R.C.row(1), P.row(1), tol);

    // Unique closest region: face index is deterministic even though ties are
    // allowed elsewhere; the geometric facts are strategy-independent.
    REQUIRE(R.I(2) == 0);
    test_common::assert_near(R.C.row(2),
      RowVector3d(0.5,0.25,0.0), tol);
    REQUIRE((P.row(2)-R.C.row(2)).norm() - 1.0 < tol);
    if(!winding_based)
    {
      REQUIRE(std::abs(std::abs(R.S(2)) - 1.0) < tol);
    }
  }
}

// ---------------------------------------------------------------------------
// 7. Inconsistent orientation and non-manifold adjacency remove the closed-
//    solid premise, so integer inside/outside classification no longer
//    exists. The tests pin the method-specific, deterministic behavior and
//    keep the still-valid geometric facts separate.
// ---------------------------------------------------------------------------
TEST_CASE("signed_distance_contract:"
  " inconsistent_orientation_and_nonmanifold_classification", "[igl]")
{
  // (a) Cube with the whole x = +1 square (faces 10 and 11) flipped.
  Eigen::MatrixXd V;
  Eigen::MatrixXi F_bad;
  make_cube(1.0, V, F_bad);
  F_bad.row(10) << 1,6,2;  // reverse of (1,2,6)
  F_bad.row(11) << 1,5,6;  // reverse of (1,6,5)

  const RowVector3d center(0,0,0);
  // Each square side subtends winding 1/6 at the center. Flipping the x = +1
  // side changes its contribution sign: w = 5/6 - 1/6 = 2/3, so the mesh has
  // no valid integer inside/outside classification.
  const double w_center = igl::winding_number(V,F_bad,center);
  REQUIRE(std::abs(w_center - 2.0/3.0) < EPS_EXACT);

  {
    const SDResult R = run_signed_distance(
      center, V, F_bad, igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER);
    // s = 1 - 2*(2/3) = -1/3 while the geometric distance is 1.
    REQUIRE(std::abs(R.S(0) - (-1.0/3.0)) < EPS_EXACT);
    REQUIRE(std::abs(std::abs(R.S(0)) - 1.0) > 0.3);
  }
  // The unsigned geometry remains valid even though classification does not.
  {
    const SDResult R = run_signed_distance(
      center, V, F_bad, igl::SIGNED_DISTANCE_TYPE_UNSIGNED);
    REQUIRE(std::abs(R.S(0) - 1.0) < EPS_EXACT);
  }
  // Pseudo-normal is local: interior points near a consistent face get the
  // usual sign, while a point near the flipped face gets the opposite one.
  {
    Eigen::MatrixXd Q(2,3);
    Q <<
      0.0, 0.0,  0.9,  // near consistent z = +1 face
      0.9, 0.0,  0.0;  // near the flipped x = +1 face
    const SDResult R = run_signed_distance(Q,V,F_bad,
      igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL);
    REQUIRE(R.S(0) < 0.0);
    REQUIRE(R.S(1) > 0.0);
    for(int k = 0;k<2;k++)
    {
      REQUIRE(std::abs(std::abs(R.S(k)) - 0.1) < EPS_EXACT);
    }
  }

  // (b) Non-manifold cube: the x = +1 face is duplicated, so its three edges
  //     are incident to three faces.
  Eigen::MatrixXi F_nm(F_bad.rows()+1,3);
  Eigen::MatrixXi F_good;
  make_cube(1.0, V, F_good);
  F_nm.topRows(F_good.rows()) = F_good;
  F_nm.row(F_good.rows()) = F_good.row(10);
  REQUIRE(igl::is_edge_manifold(F_good));
  REQUIRE_FALSE(igl::is_edge_manifold(F_nm));

  // w = 1 + 1/12 = 13/12: fractional winding, no solid membership available.
  const double w_nm = igl::winding_number(V,F_nm,center);
  REQUIRE(std::abs(w_nm - 13.0/12.0) < EPS_EXACT);
  {
    const SDResult R = run_signed_distance(
      center, V, F_nm, igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER);
    // s = 1 - 2*(13/12) = -7/6
    REQUIRE(std::abs(R.S(0) - (-7.0/6.0)) < EPS_EXACT);
    REQUIRE(std::abs(std::abs(R.S(0)) - 1.0) > 0.1);
  }
  // Away from the non-manifold edges the pseudo-normal query still runs and
  // returns the exact local magnitude and sign.
  {
    Eigen::MatrixXd Q(1,3);
    Q << 0.0, 0.0, 0.9;
    const SDResult R = run_signed_distance(Q,V,F_nm,
      igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL);
    REQUIRE(R.S(0) < 0.0);
    REQUIRE(std::abs(std::abs(R.S(0)) - 0.1) < EPS_EXACT);
  }
}

// ---------------------------------------------------------------------------
// 8. Near-surface queries: numerical approximation error and a wrong sign are
//    different things. Magnitude tolerances are strategy-specific; sign checks
//    stay strict and cannot be masked by loosening the FWN tolerance.
// ---------------------------------------------------------------------------
TEST_CASE("signed_distance_contract: near_surface_precision_vs_wrong_sign",
  "[igl]")
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  make_cube(1.0, V, F);

  const double delta = 1e-10;
  Eigen::MatrixXd Pclose(2,3);
  Pclose <<
    1.0 + delta, 0.3, 0.4,  // just outside
    1.0 - delta, 0.3, 0.4;  // just inside

  for(const auto type : all_3d_types())
  {
    if(type == igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)
    {
      // Single-precision FWN cannot resolve a 1e-10 offset: the float32
      // geometry rounds both queries onto the surface, so the signed value
      // collapses toward zero. Only finiteness and a float-scale near-zero
      // bound are asserted here -- deliberately no sign requirement.
      const SDResult R = run_signed_distance(Pclose,V,F,type);
      REQUIRE(std::isfinite(R.S(0)));
      REQUIRE(std::isfinite(R.S(1)));
      REQUIRE(std::abs(R.S(0)) < 2e-7);
      REQUIRE(std::abs(R.S(1)) < 2e-7);
      continue;
    }
    const SDResult R = run_signed_distance(Pclose,V,F,type);
    CAPTURE(type);
    if(type == igl::SIGNED_DISTANCE_TYPE_UNSIGNED)
    {
      REQUIRE(R.S(0) > 0.0);
      REQUIRE(R.S(1) > 0.0);
    }
    else
    {
      // Strict classification even 1e-10 from the surface.
      REQUIRE(R.S(0) > 0.0);
      REQUIRE(R.S(1) < 0.0);
    }
    REQUIRE(std::abs(std::abs(R.S(0)) - delta) < EPS_EXACT);
    REQUIRE(std::abs(std::abs(R.S(1)) - delta) < EPS_EXACT);
  }

  // At a resolvable offset the FWN classification must also be correct: its
  // loose magnitude tolerance must not excuse a sign flip.
  Eigen::MatrixXd Pmid(2,3);
  Pmid <<
    1.0 + 1e-4, 0.3, 0.4,
    1.0 - 1e-4, 0.3, 0.4;
  {
    const SDResult R = run_signed_distance(Pmid,V,F,
      igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER);
    REQUIRE(R.S(0) > 0.0);
    REQUIRE(R.S(1) < 0.0);
    REQUIRE(std::abs(std::abs(R.S(0)) - 1e-4) < EPS_FWN_LOOSE);
    REQUIRE(std::abs(std::abs(R.S(1)) - 1e-4) < EPS_FWN_LOOSE);
  }
}

// ---------------------------------------------------------------------------
// 9. Distance-range filtering: excluded queries are NaN with the documented
//    sentinel face, which is distinct from a valid on-surface zero. In a mixed
//    batch the validity of S, I and C stays consistent row by row.
// ---------------------------------------------------------------------------
TEST_CASE("signed_distance_contract: range_filter_nan_vs_valid_zero", "[igl]")
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  make_cube(1.0, V, F);

  const double lower_bound = 0.2;
  const double upper_bound = 0.8;  // accept d in [0.2, 0.8)

  Eigen::MatrixXd P(4,3);
  Eigen::VectorXi filtered(4);
  P <<
    1.7, 0.0, 0.0,  // d = 0.7  -> kept
    2.0, 0.0, 0.0,  // d = 1.0  -> too far, filtered
    0.9, 0.0, 0.0,  // d = 0.1  -> too close, filtered
    1.0, 0.0, 0.0;  // d = 0.0  -> filtered by the range, but a genuine zero
                    //             in the unbounded query below
  filtered << 0, 1, 1, 1;

  for(const auto type : all_3d_types())
  {
    Eigen::VectorXd S;
    Eigen::VectorXi I;
    Eigen::MatrixXd C,N;
    igl::signed_distance(P,V,F,type,lower_bound,upper_bound,S,I,C,N);
    CAPTURE(type);
    for(int k = 0;k<4;k++)
    {
      if(filtered(k))
      {
        REQUIRE(std::isnan(S(k)));
        REQUIRE(I(k) == F.rows()+1);       // documented sentinel
        REQUIRE(C.row(k).norm() == 0.0);   // closest point zeroed
      }
      else
      {
        REQUIRE_FALSE(std::isnan(S(k)));
        REQUIRE(I(k) >= 0);
        REQUIRE(I(k) <  F.rows());
        REQUIRE(std::abs(std::abs(S(k)) - 0.7) < tol_for(type));
        REQUIRE(point_is_on_face(C.row(k),V,F,I(k),tol_for(type)));
      }
    }
  }

  // The unbounded query at the surface point is a real zero: finite S, valid
  // face, closest point coincident with the query.
  Eigen::MatrixXd Pzero = P.bottomRows(1);
  for(const auto type : all_3d_types())
  {
    const SDResult R = run_signed_distance(Pzero,V,F,type);
    REQUIRE_FALSE(std::isnan(R.S(0)));
    REQUIRE(std::abs(R.S(0)) < tol_for(type));
    REQUIRE(R.I(0) >= 0);
    test_common::assert_near(R.C, Pzero, tol_for(type));
  }

  // Mixed batch without range cuts: per-row S/I/C validity correspondence and
  // agreement with the independent point_mesh_squared_distance query.
  Eigen::MatrixXd Pb(6,3);
  Pb <<
    2.0, 0.0, 0.0,
    0.5, 0.0, 0.0,
    2.0, 2.0, 2.0,
   -0.4,-0.4,-0.4,
    0.9,-0.9, 0.3,
    0.2,-1.3, 0.2;
  for(const auto type : all_3d_types())
  {
    const SDResult R = run_signed_distance(Pb,V,F,type);
    Eigen::VectorXd sqrD;
    Eigen::VectorXi I0;
    Eigen::MatrixXd C0;
    igl::point_mesh_squared_distance(Pb,V,F,sqrD,I0,C0);
    const double tol = tol_for(type);
    for(int k = 0;k<Pb.rows();k++)
    {
      CAPTURE(type); CAPTURE(k);
      REQUIRE(std::isfinite(R.S(k)));
      REQUIRE(R.I(k) >= 0);
      REQUIRE(R.I(k) <  F.rows());
      REQUIRE(point_is_on_face(R.C.row(k),V,F,R.I(k),tol));
      // Signed-distance magnitude matches the unsigned squared-distance API.
      // Compare the magnitudes (not S² vs sqrD): squaring amplifies the FWN
      // sign-factor error by a factor of 2*d.
      REQUIRE(std::abs(std::abs(R.S(k)) - std::sqrt(sqrD(k))) < tol);
      test_common::assert_near(R.C.row(k), C0.row(k), tol);
    }
  }
}
