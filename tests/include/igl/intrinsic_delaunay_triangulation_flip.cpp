// This file is part of libigl, a simple c++ geometry processing library.
//
// These tests pin down the invariance properties of an _intrinsic_ Delaunay
// triangulation: flipping an edge changes the face connectivity, but the edge
// length metric and the discrete operators built on it must still describe the
// same intrinsic surface.  All geometric examples are intentionally tiny so
// that the assertions can be reasoned about by hand.
#include <test_common.h>
#include <igl/intrinsic_delaunay_triangulation.h>
#include <igl/is_intrinsic_delaunay.h>
#include <igl/edge_lengths.h>
#include <igl/triangulated_grid.h>
#include <igl/unique_edge_map.h>
#include <igl/boundary_facets.h>
#include <igl/cotmatrix_intrinsic.h>
#include <igl/massmatrix_intrinsic.h>
#include <igl/doublearea.h>
#include <igl/EPS.h>

#include <numeric>
#include <random>
#include <set>
#include <utility>

namespace
{
  // Compare two sparse matrices via their bilinear form on random vectors.
  // There is no sparse overload of test_common::assert_near, so this mirrors
  // the check used in cotmatrix_intrinsic.cpp's manifold_meshes test.
  template <typename Scalar>
  void assert_sparse_near(
    const Eigen::SparseMatrix<Scalar> & A,
    const Eigen::SparseMatrix<Scalar> & B,
    const Scalar eps)
  {
    REQUIRE(A.rows() == B.rows());
    REQUIRE(A.cols() == B.cols());
    const Eigen::Matrix<Scalar,Eigen::Dynamic,1> u =
      Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Random(A.cols());
    const Eigen::Matrix<Scalar,Eigen::Dynamic,1> v =
      Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Random(A.cols());
    const Scalar scale = u.norm()*v.norm();
    REQUIRE(u.dot(A*v)/scale == Approx(u.dot(B*v)/scale).margin(eps));
  }

  // Run intrinsic Delaunay on (l_in, F_in) and return the intrinsic cotangent
  // Laplacian of the resulting triangulation.
  template <typename Derivedl, typename DerivedF>
  Eigen::SparseMatrix<double> intrinsic_delaunay_cotmatrix(
    const Eigen::MatrixBase<Derivedl> & l_in,
    const Eigen::MatrixBase<DerivedF> & F_in)
  {
    Eigen::MatrixXd l;
    Eigen::MatrixXi F;
    igl::intrinsic_delaunay_triangulation(l_in, F_in, l, F);
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix_intrinsic(l, F, L);
    return L;
  }
}

// An already intrinsic-Delaunay mesh must be left completely untouched: no
// edge is flipped, so the connectivity and every edge length are identical to
// the input.  This covers the "un-flipped edge" case.
TEST_CASE("intrinsic_delaunay_triangulation_flip: already_delaunay_unchanged", "[igl]")
{
  // Convex diamond (two triangles sharing the vertical diagonal 0-3); the
  // shared diagonal is the Delaunay choice for these four points.
  const Eigen::MatrixXd V =
    (Eigen::MatrixXd(4,2)<<
     0,10,
     1,0,
     1,20,
     2,10).finished();
  const Eigen::MatrixXi F_in =
    (Eigen::MatrixXi(2,3)<<
     0,1,3,
     0,3,2).finished();
  Eigen::MatrixXd l_in;
  igl::edge_lengths(V,F_in,l_in);

  // Sanity check: every edge really is intrinsic Delaunay before flipping.
  Eigen::Matrix<bool,Eigen::Dynamic,3> D_before;
  igl::is_intrinsic_delaunay(l_in,F_in,D_before);
  REQUIRE(D_before.all());

  Eigen::MatrixXd l;
  Eigen::MatrixXi F;
  igl::intrinsic_delaunay_triangulation(l_in,F_in,l,F);

  test_common::assert_eq(F,F_in);
  test_common::assert_near(l,l_in,igl::EPS<double>());
}

// Boundary edges can never be flipped (they only have one incident face), so
// the boundary loop of the surface must be preserved by the algorithm even on
// an input where interior edges do get flipped.
TEST_CASE("intrinsic_delaunay_triangulation_flip: boundary_edges_preserved", "[igl]")
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F_in;
  // Small open patch: 3x3 vertices, 8 triangles.  Skewing makes the regular
  // diagonals non-Delaunay so that interior flips actually happen.
  igl::triangulated_grid(3,3,V,F_in);
  V.col(0) -= 1.5*V.col(1);
  Eigen::MatrixXd l_in;
  igl::edge_lengths(V,F_in,l_in);

  Eigen::MatrixXi B_in;
  igl::boundary_facets(F_in,B_in);
  REQUIRE(B_in.rows() > 0);

  Eigen::MatrixXd l;
  Eigen::MatrixXi F;
  Eigen::MatrixXi E,uE;
  Eigen::VectorXi EMAP;
  std::vector<std::vector<int> > uE2E;
  igl::intrinsic_delaunay_triangulation(l_in,F_in,l,F,E,uE,EMAP,uE2E);

  // The output must be fully intrinsic Delaunay...
  Eigen::Matrix<bool,Eigen::Dynamic,3> D;
  igl::is_intrinsic_delaunay(l,F,uE2E,D);
  REQUIRE(D.all());

  // ...and the boundary edges of the domain must be exactly the same
  // (unordered) set: boundary half-edges are never flipped.
  Eigen::MatrixXi B_out;
  igl::boundary_facets(F,B_out);
  REQUIRE(B_in.rows() == B_out.rows());
  auto edge_set = [](const Eigen::MatrixXi & edges) {
    std::set<std::pair<int,int> > s;
    for(int i = 0;i<edges.rows();i++)
    {
      s.insert(std::minmax(edges(i,0),edges(i,1)));
    }
    return s;
  };
  REQUIRE(edge_set(B_in) == edge_set(B_out));

  // Every unique boundary edge still has exactly one incident half-edge.
  Eigen::MatrixXi bE,b_uE;
  Eigen::VectorXi b_EMAP;
  std::vector<std::vector<int> > b_uE2E;
  igl::unique_edge_map(F,bE,b_uE,b_EMAP,b_uE2E);
  int num_boundary = 0;
  for(const auto & half_edges : b_uE2E)
  {
    if(half_edges.size() == 1) num_boundary++;
  }
  REQUIRE(num_boundary == B_in.rows());
}

// Degenerate inputs must be handled without crashing and without inventing
// geometry: an empty mesh and a single triangle contain no flippable interior
// edge, so everything passes through unchanged.
TEST_CASE("intrinsic_delaunay_triangulation_flip: degenerate_inputs", "[igl]")
{
  SECTION("empty mesh")
  {
    // 0 rows but 3 columns: well-formed empty edge-length / face matrices.
    Eigen::MatrixXd l_in(0,3);
    Eigen::MatrixXi F_in(0,3);
    Eigen::MatrixXd l;
    Eigen::MatrixXi F;
    REQUIRE_NOTHROW(igl::intrinsic_delaunay_triangulation(l_in,F_in,l,F));
    REQUIRE(l.rows() == 0);
    REQUIRE(l.cols() == 3);
    REQUIRE(F.rows() == 0);
    REQUIRE(F.cols() == 3);
  }

  SECTION("single triangle")
  {
    // All three edges are boundary edges: nothing can ever be flipped.
    const Eigen::MatrixXd V =
      (Eigen::MatrixXd(3,2)<<0,0, 1,0, 0,2).finished();
    const Eigen::MatrixXi F_in = (Eigen::MatrixXi(1,3)<<0,1,2).finished();
    Eigen::MatrixXd l_in;
    igl::edge_lengths(V,F_in,l_in);

    Eigen::MatrixXd l;
    Eigen::MatrixXi F;
    igl::intrinsic_delaunay_triangulation(l_in,F_in,l,F);
    test_common::assert_eq(F,F_in);
    test_common::assert_near(l,l_in,igl::EPS<double>());

    // The sole triangle should still be reported intrinsic Delaunay.
    Eigen::Matrix<bool,Eigen::Dynamic,3> D;
    igl::is_intrinsic_delaunay(l,F,D);
    REQUIRE(D.all());
  }

  SECTION("collinear zero-area triangles")
  {
    // Four collinear points split into two zero-area triangles: a degenerate
    // metric (triangle inequalities hold only with equality).  The half-angle
    // formulas are exercised on their limiting inputs; the routine must
    // terminate with finite, non-negative edge lengths and the face count
    // intact, while representing the (zero) area faithfully.
    const Eigen::MatrixXd V =
      (Eigen::MatrixXd(4,2)<<
       0.0,0.0,
       1.0,0.0,
       2.0,0.0,
       3.0,0.0).finished();
    const Eigen::MatrixXi F_in =
      (Eigen::MatrixXi(2,3)<<
       0,1,2,
       2,1,3).finished();
    Eigen::MatrixXd l_in;
    igl::edge_lengths(V,F_in,l_in);

    Eigen::MatrixXd l;
    Eigen::MatrixXi F;
    REQUIRE_NOTHROW(igl::intrinsic_delaunay_triangulation(l_in,F_in,l,F));
    REQUIRE(F.rows() == F_in.rows());
    for(int i = 0;i<l.size();i++)
    {
      REQUIRE(std::isfinite(l(i)));
      REQUIRE(l(i) >= 0.0);
    }
    // Total (zero) area is still represented faithfully.
    Eigen::VectorXd dblA;
    igl::doublearea(l,0.0,dblA);
    REQUIRE(dblA.sum() == Approx(0.0).margin(1.0e-12));
  }
}

// The intrinsic Delaunay condition is a property of the surface, not of the
// order in which candidate edges are processed.  Reordering the input faces
// changes the order in which edges enter the work queue (and thus the actual
// flip sequence), but the resulting discrete operators must be identical.
TEST_CASE("intrinsic_delaunay_triangulation_flip: flip_order_independence", "[igl]")
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F_in;
  // 4x4 vertices -> 18 triangles, several of which flip after skewing.
  igl::triangulated_grid(4,4,V,F_in);
  V.col(0) -= 1.5*V.col(1);
  Eigen::MatrixXd l_in;
  igl::edge_lengths(V,F_in,l_in);

  // Reference result on the untouched ordering.
  const Eigen::SparseMatrix<double> L_ref =
    intrinsic_delaunay_cotmatrix(l_in,F_in);

  std::vector<int> perm(F_in.rows());
  std::iota(perm.begin(),perm.end(),0);
  std::mt19937 rng(0xC0FFEE);
  for(int trial = 0; trial < 4; trial++)
  {
    std::shuffle(perm.begin(),perm.end(),rng);
    Eigen::MatrixXi F_perm(F_in.rows(),3);
    Eigen::MatrixXd l_perm(l_in.rows(),3);
    for(int f = 0;f<F_in.rows();f++)
    {
      F_perm.row(f) = F_in.row(perm[f]);
      l_perm.row(f) = l_in.row(perm[f]);
    }

    Eigen::MatrixXd l;
    Eigen::MatrixXi F;
    igl::intrinsic_delaunay_triangulation(l_perm,F_perm,l,F);

    // Regardless of flip order, the output is fully intrinsic Delaunay...
    Eigen::Matrix<bool,Eigen::Dynamic,3> D;
    igl::is_intrinsic_delaunay(l,F,D);
    REQUIRE(D.all());

    // ...and expresses the same Laplace-Beltrami operator on the same
    // vertices (vertex labels are untouched, so the matrices line up
    // directly).
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix_intrinsic(l,F,L);
    assert_sparse_near(L_ref,L,1.0e-12);
  }
}

// Direct check on the smallest mesh that actually flips.  The flipped edge
// gets a new intrinsic length; the assertions pin down what "same surface"
// means concretely:
//   * the new lengths are exactly the Euclidean lengths of the re-connected
//     planar triangulation (the metric stays compatible with the embedding),
//   * the discrete operator built from the intrinsic metric coincides with
//     the ordinary operator of that re-connected embedding,
//   * total area and barycentric mass are metric invariants of the flip.
// Note that the operator is NOT expected to equal the pre-flip one: the input
// diagonal is non-Delaunay, and the whole point of the intrinsic Delaunay
// construction is to replace its operator (which carries a negative cotangent
// weight on the bad interior edge) by the Delaunay one, whose interior edges
// all have non-negative cotangent weights.  Independence from the flip
// *order* is checked separately in flip_order_independence.
TEST_CASE("intrinsic_delaunay_triangulation_flip: operators_invariant_under_flip", "[igl]")
{
  // Same four points as is_intrinsic_delaunay: two_triangles, but with the
  // non-Delaunay diagonal so a single flip is forced.
  const Eigen::MatrixXd V =
    (Eigen::MatrixXd(4,2)<<
     0,12,
     1,0,
     1,20,
     2,9).finished();
  const Eigen::MatrixXi F_in =
    (Eigen::MatrixXi(2,3)<<
     0,1,2,
     2,1,3).finished();
  Eigen::MatrixXd l_in;
  igl::edge_lengths(V,F_in,l_in);

  // Precondition: the shared interior edge 1-2 is not intrinsic Delaunay.
  Eigen::Matrix<bool,Eigen::Dynamic,3> D_before;
  igl::is_intrinsic_delaunay(l_in,F_in,D_before);
  REQUIRE_FALSE(D_before.all());

  Eigen::MatrixXd l;
  Eigen::MatrixXi F;
  igl::intrinsic_delaunay_triangulation(l_in,F_in,l,F);

  // A flip really did change the connectivity.
  REQUIRE((F.array() != F_in.array()).any());

  // The intrinsic lengths must match the Euclidean lengths of the flipped
  // triangulation of the same planar point set.
  Eigen::MatrixXd l_extrinsic;
  igl::edge_lengths(V,F,l_extrinsic);
  test_common::assert_near(l,l_extrinsic,1.0e-10);

  // Discrete operator self-consistency: the intrinsic cotangent Laplacian
  // built after flipping equals the ordinary cotangent Laplacian evaluated on
  // the same planar embedding with the flipped connectivity.
  Eigen::SparseMatrix<double> L_intrinsic;
  igl::cotmatrix_intrinsic(l,F,L_intrinsic);
  Eigen::SparseMatrix<double> L_extrinsic_flipped;
  igl::cotmatrix_intrinsic(l_extrinsic,F,L_extrinsic_flipped);
  assert_sparse_near(L_intrinsic,L_extrinsic_flipped,1.0e-12);

  // Delaunay means the cotangent weight of every INTERIOR edge is
  // non-negative; boundary edges may still have negative weights when a
  // boundary angle is obtuse.  The two faces share interior edge 1-2, which
  // the flip replaces by edge 0-3.  In igl's cotmatrix the off-diagonal
  // entries are the cotangent weights themselves.
  auto entry = [](const Eigen::SparseMatrix<double> & L, int i, int j) {
    return Eigen::MatrixXd(L)(i,j);
  };
  Eigen::SparseMatrix<double> L_before;
  igl::cotmatrix_intrinsic(l_in,F_in,L_before);
  // Pre-flip interior edge 1-2 carries a strictly negative cotangent weight
  // (the two opposite angles are both obtuse)...
  REQUIRE(entry(L_before,1,2) < -1.0e-12);
  // ...the flipped interior edge 0-3 has a strictly positive one.
  REQUIRE(entry(L_intrinsic,0,3) > 1.0e-12);

  // Total area is preserved exactly by an intrinsic flip.
  Eigen::VectorXd dblA_in,dblA_out;
  igl::doublearea(l_in,0.0,dblA_in);
  igl::doublearea(l   ,0.0,dblA_out);
  REQUIRE(dblA_in.sum() == Approx(dblA_out.sum()).margin(1.0e-12));

  // Barycentric (area) mass only depends on per-face areas, which an
  // intrinsic flip preserves, so the total surface mass is unchanged.
  Eigen::SparseMatrix<double> M_in,M_out;
  igl::massmatrix_intrinsic(
    l_in,F_in,igl::MASSMATRIX_TYPE_BARYCENTRIC,M_in);
  igl::massmatrix_intrinsic(
    l   ,F   ,igl::MASSMATRIX_TYPE_BARYCENTRIC,M_out);
  REQUIRE(M_in.sum() == Approx(M_out.sum()).margin(1.0e-12));
}
