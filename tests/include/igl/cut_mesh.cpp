#include <test_common.h>
#include <igl/cut_mesh.h>
#include <igl/vertex_components.h>
#include <igl/edges.h>
#include <igl/boundary_loop.h>
#include <igl/facet_components.h>
#include <igl/is_edge_manifold.h>
#include <igl/seam_edges.h>
#include <igl/unzip_corners.h>
#include <igl/readOBJ.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

TEST_CASE("cut_mesh: seperate mesh", "[igl]") {
  
  Eigen::MatrixXd V(9,3);
  V << 0,0,0,
       0,1,0,
       0,2,0,
       1,2,0,
       2,2,0,
       2,1,0,
       2,0,0,
       1,0,0,
       1,1,0;
  Eigen::MatrixXi F(8,3);
  F << 0,1,8,
       1,2,8,
       2,3,8,
       3,4,8,
       4,5,8,
       5,6,8,
       6,7,8,
       7,0,8;
  
  Eigen::MatrixXi C(8,3);
  C << 0,1,1,
       0,1,1,
       0,0,1,
       0,0,0,
       0,0,0,
       0,0,0,
       0,0,0,
       0,1,0;
  Eigen::VectorXi I;
  igl::cut_mesh(V,F,C,I);
  Eigen::VectorXi count;
  igl::vertex_components(F, count);
  REQUIRE(count.maxCoeff() == 2);

}

TEST_CASE("cut_mesh: single edge", "[igl]") {
  
  Eigen::MatrixXd V(9,3);
  V << 0,0,0,
       0,1,0,
       0,2,0,
       1,2,0,
       2,2,0,
       2,1,0,
       2,0,0,
       1,0,0,
       1,1,0;
  Eigen::MatrixXi F(8,3);
  F << 0,1,8,
       1,2,8,
       2,3,8,
       3,4,8,
       4,5,8,
       5,6,8,
       6,7,8,
       7,0,8;
  
  Eigen::MatrixXi C(8,3);
  C << 0,1,0,
       0,0,1,
       0,0,0,
       0,0,0,
       0,0,0,
       0,0,0,
       0,0,0,
       0,0,0;
  Eigen::VectorXi I;
  igl::cut_mesh(V,F,C,I);
  Eigen::VectorXi count;
  igl::vertex_components(F, count);
  REQUIRE(0 == count.maxCoeff());
  Eigen::MatrixXi E;
  igl::edges(F, E);
  const auto euler = V.rows() - E.rows() + F.rows();
  REQUIRE ( 1 == euler );

}

TEST_CASE("cut_mesh: two triangles", "[igl]") {

  Eigen::MatrixXd V(4,3);
  V << 0,0,0,
       0,1,0,
       1,0,0,
       1,1,0;
  Eigen::MatrixXi F(2,3);
  F << 0,1,2,
       2,1,3;
  // Only mark one side of the internal edge to cut. This should still work.
  Eigen::MatrixXi C(2,3);
  C << 0,0,0,
       1,0,0;

  Eigen::MatrixXd VCut;
  Eigen::MatrixXi FCut;
  igl::cut_mesh(V,F,C,VCut,FCut);
  REQUIRE( VCut.rows() == 6 );
}

namespace
{
  // ---------------------------------------------------------------------------
  // Helpers for the texture-seam contract tests below.
  //
  // The contract under test: cutting a mesh along texture seams must
  //   1. keep, for every face corner of the cut mesh, a traceable link to the
  //      original geometric vertex *and* to the original texture attribute,
  //   2. produce a mesh whose boundary loops are valid, without turning UV
  //      seams into geometric cross-sections elsewhere, with one consistent
  //      attribute per face corner, and
  //   3. allow texture seams, genuine mesh boundaries and UV foldovers to be
  //      told apart.
  // ---------------------------------------------------------------------------

  typedef std::pair<int,int> Edge;
  typedef std::pair<int,int> Corner;

  Edge edge_key(const int a, const int b)
  {
    return Edge(std::min(a,b),std::max(a,b));
  }

  /// Undirected edge -> list of (face,corner) incidences
  std::map<Edge,std::vector<Corner> > edge_corners(
    const Eigen::MatrixXi & F)
  {
    std::map<Edge,std::vector<Corner> > result;
    for(int f = 0;f<F.rows();f++)
    {
      for(int k = 0;k<3;k++)
      {
        result[edge_key(F(f,k),F(f,(k+1)%3))].push_back(Corner(f,k));
      }
    }
    return result;
  }

  /// Undirected edge -> number of incident face corners
  std::map<Edge,int> edge_multiplicity(const Eigen::MatrixXi & F)
  {
    std::map<Edge,int> result;
    for(const auto & kv : edge_corners(F))
    {
      result[kv.first] = static_cast<int>(kv.second.size());
    }
    return result;
  }

  /// Edges with exactly one incident face
  std::set<Edge> boundary_edge_set(const Eigen::MatrixXi & F)
  {
    std::set<Edge> result;
    for(const auto & kv : edge_corners(F))
    {
      if(kv.second.size() == 1) { result.insert(kv.first); }
    }
    return result;
  }

  /// Undirected edges described by the rows of a [face;corner] matrix such as
  /// the "seams", "boundaries" and "foldovers" output of igl::seam_edges.
  std::set<Edge> edge_set_from_rows(
    const Eigen::MatrixXi & F,
    const Eigen::MatrixXi & rows)
  {
    std::set<Edge> result;
    for(int r = 0;r<rows.rows();r++)
    {
      result.insert(edge_key(F(rows(r,0),rows(r,1)),
                             F(rows(r,0),(rows(r,1)+1)%3)));
    }
    return result;
  }

  /// Split the corners of each vertex into sectors: two corners of the same
  /// vertex share a sector iff their faces can be walked from one to the other
  /// around that vertex without crossing a cut edge. These sectors are exactly
  /// the texture chart vertices a cut should produce.
  std::vector<int> corner_sectors(
    const Eigen::MatrixXi & F,
    const std::set<Edge> & cuts,
    int & num_sectors)
  {
    const int num_corners = 3*F.rows();
    std::vector<int> parent(num_corners);
    std::iota(parent.begin(),parent.end(),0);
    std::function<int(int)> find_root = [&](int x)
    {
      while(parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
      return x;
    };
    const auto unite = [&](int a,int b)
    {
      a = find_root(a);
      b = find_root(b);
      if(a != b) { parent[a] = b; }
    };
    const auto corner_at = [&](const int f,const int v)
    {
      for(int k = 0;k<3;k++) { if(F(f,k) == v) { return f*3+k; } }
      return -1;
    };
    for(const auto & kv : edge_corners(F))
    {
      if(cuts.count(kv.first)) { continue; }
      const int v0 = kv.first.first;
      const int v1 = kv.first.second;
      for(size_t i = 0;i<kv.second.size();i++)
      {
        for(size_t j = i+1;j<kv.second.size();j++)
        {
          const int f = kv.second[i].first;
          const int g = kv.second[j].first;
          const int cf0 = corner_at(f,v0);
          const int cg0 = corner_at(g,v0);
          const int cf1 = corner_at(f,v1);
          const int cg1 = corner_at(g,v1);
          if(cf0 >= 0 && cg0 >= 0) { unite(cf0,cg0); }
          if(cf1 >= 0 && cg1 >= 0) { unite(cf1,cg1); }
        }
      }
    }
    std::map<int,int> label;
    std::vector<int> sectors(num_corners);
    for(int c = 0;c<num_corners;c++)
    {
      const int r = find_root(c);
      if(label.count(r) == 0) { label[r] = static_cast<int>(label.size()); }
      sectors[c] = label[r];
    }
    num_sectors = static_cast<int>(label.size());
    return sectors;
  }

  /// Build texture coordinates with one chart per corner sector: corners share
  /// a texture index iff they share a sector, and the UV positions coincide
  /// with the 3D positions (so no accidental foldovers are introduced).
  void texture_from_sectors(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const std::vector<int> & sectors,
    const int num_sectors,
    Eigen::MatrixXd & TC,
    Eigen::MatrixXi & FTC)
  {
    TC.resize(num_sectors,2);
    FTC.resize(F.rows(),3);
    for(int f = 0;f<F.rows();f++)
    {
      for(int k = 0;k<3;k++)
      {
        const int s = sectors[f*3+k];
        FTC(f,k) = s;
        TC(s,0) = V(F(f,k),0);
        TC(s,1) = V(F(f,k),1);
      }
    }
  }

  /// Cut flags for every edge in `edges` (both incident faces where present).
  Eigen::MatrixXi flags_for_edges(
    const Eigen::MatrixXi & F,
    const std::set<Edge> & edges)
  {
    Eigen::MatrixXi C = Eigen::MatrixXi::Zero(F.rows(),3);
    for(int f = 0;f<F.rows();f++)
    {
      for(int k = 0;k<3;k++)
      {
        if(edges.count(edge_key(F(f,k),F(f,(k+1)%3))))
        {
          C(f,k) = 1;
        }
      }
    }
    return C;
  }

  /// Cut flags on only one of the two faces incident on each edge. The
  /// documentation promises that flagging either incident face is enough.
  Eigen::MatrixXi flags_for_first_side(
    const Eigen::MatrixXi & F,
    const std::set<Edge> & edges)
  {
    Eigen::MatrixXi C = Eigen::MatrixXi::Zero(F.rows(),3);
    const auto corners = edge_corners(F);
    for(const Edge & e : edges)
    {
      const auto it = corners.find(e);
      REQUIRE(it != corners.end());
      Corner first = it->second.front();
      for(const Corner & c : it->second)
      {
        if(c < first) { first = c; }
      }
      C(first.first,first.second) = 1;
    }
    return C;
  }

  void make_grid_mesh(
    const int nx,
    const int ny,
    Eigen::MatrixXd & V,
    Eigen::MatrixXi & F)
  {
    V.resize(nx*ny,3);
    for(int j = 0;j<ny;j++)
    {
      for(int i = 0;i<nx;i++)
      {
        V.row(j*nx+i) << i,j,0;
      }
    }
    std::vector<Eigen::Vector3i> faces;
    for(int j = 0;j<ny-1;j++)
    {
      for(int i = 0;i<nx-1;i++)
      {
        const int a = j*nx+i;
        const int b = a+1;
        const int c = (j+1)*nx+i;
        const int d = c+1;
        faces.push_back(Eigen::Vector3i(a,b,d));
        faces.push_back(Eigen::Vector3i(a,d,c));
      }
    }
    F.resize(faces.size(),3);
    for(size_t k = 0;k<faces.size();k++) { F.row(k) = faces[k].transpose(); }
  }

  struct CutMesh
  {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::VectorXi I;
  };

  CutMesh apply_cut(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const Eigen::MatrixXi & C)
  {
    CutMesh result;
    result.V = V;
    result.F = F;
    igl::cut_mesh(result.V,result.F,C,result.I);
    return result;
  }

  /// Every face corner of the cut mesh must reference a copy of the original
  /// vertex that the corresponding corner of the input face referenced, and the
  /// copies must sit exactly at the original position.
  void require_geometric_provenance(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const CutMesh & cut)
  {
    REQUIRE(cut.F.rows() == F.rows());
    REQUIRE(cut.I.size() == cut.V.rows());
    for(int f = 0;f<cut.F.rows();f++)
    {
      for(int k = 0;k<3;k++)
      {
        CAPTURE(f);
        CAPTURE(k);
        REQUIRE(cut.I(cut.F(f,k)) == F(f,k));
        REQUIRE((cut.V.row(cut.F(f,k)) - V.row(F(f,k))).norm() == 0.0);
      }
    }
  }

  std::map<int,std::vector<int> > copies_by_original(const CutMesh & cut)
  {
    std::map<int,std::vector<int> > result;
    for(int v = 0;v<cut.V.rows();v++) { result[cut.I(v)].push_back(v); }
    return result;
  }

  /// Each new vertex must lie in exactly one corner sector (i.e. it belongs to
  /// one texture chart), the copies of an original vertex must cover exactly
  /// that vertex's sectors, no new vertex may go unused, and every corner that
  /// references a new vertex must carry the same texture index.
  void require_attribute_consistency(
    const Eigen::MatrixXi & F,
    const Eigen::MatrixXi & FTC,
    const std::vector<int> & sectors,
    const CutMesh & cut)
  {
    std::map<int,std::set<int> > sectors_of_new_vertex;
    std::map<int,std::set<int> > textures_of_new_vertex;
    for(int f = 0;f<cut.F.rows();f++)
    {
      for(int k = 0;k<3;k++)
      {
        const int nv = cut.F(f,k);
        sectors_of_new_vertex[nv].insert(sectors[f*3+k]);
        textures_of_new_vertex[nv].insert(FTC(f,k));
      }
    }
    std::vector<int> used(cut.V.rows(),0);
    for(int f = 0;f<cut.F.rows();f++)
    {
      for(int k = 0;k<3;k++) { used[cut.F(f,k)] = 1; }
    }
    for(int v = 0;v<cut.V.rows();v++)
    {
      CAPTURE(v);
      REQUIRE(used[v] == 1);
      REQUIRE(sectors_of_new_vertex.at(v).size() == 1);
      REQUIRE(textures_of_new_vertex.at(v).size() == 1);
    }

    // Copies of an original vertex <-> the sectors around that vertex.
    std::map<int,std::set<int> > sectors_of_original_vertex;
    for(int f = 0;f<F.rows();f++)
    {
      for(int k = 0;k<3;k++)
      {
        sectors_of_original_vertex[F(f,k)].insert(sectors[f*3+k]);
      }
    }
    const auto copies = copies_by_original(cut);
    for(const auto & kv : sectors_of_original_vertex)
    {
      CAPTURE(kv.first);
      REQUIRE(copies.count(kv.first) == 1);
      REQUIRE(static_cast<int>(copies.at(kv.first).size()) ==
              static_cast<int>(kv.second.size()));
      std::set<int> copy_sectors;
      for(const int nv : copies.at(kv.first))
      {
        copy_sectors.insert(*sectors_of_new_vertex.at(nv).begin());
      }
      REQUIRE(copy_sectors == kv.second);
    }
  }

  /// Boundary loops must be simple closed loops whose directed edges are
  /// exactly the boundary half-edges of the mesh, each used once. Returns the
  /// number of undirected boundary edges.
  int require_valid_boundary_loops(const Eigen::MatrixXi & F)
  {
    const std::map<Edge,int> mult = edge_multiplicity(F);
    std::set<std::pair<int,int> > directed_boundary;
    int num_boundary_edges = 0;
    for(int f = 0;f<F.rows();f++)
    {
      for(int k = 0;k<3;k++)
      {
        if(mult.at(edge_key(F(f,k),F(f,(k+1)%3))) == 1)
        {
          directed_boundary.insert(std::make_pair(F(f,k),F(f,(k+1)%3)));
          num_boundary_edges++;
        }
      }
    }
    REQUIRE(static_cast<int>(directed_boundary.size()) == num_boundary_edges);

    std::vector<std::vector<int> > loops;
    igl::boundary_loop(F,loops);
    REQUIRE(loops.size() > 0);
    std::set<std::pair<int,int> > visited;
    for(const std::vector<int> & loop : loops)
    {
      REQUIRE(loop.size() >= 3);
      const std::set<int> distinct(loop.begin(),loop.end());
      REQUIRE(distinct.size() == loop.size());
      for(size_t i = 0;i<loop.size();i++)
      {
        const std::pair<int,int> e =
          std::make_pair(loop[i],loop[(i+1)%loop.size()]);
        CAPTURE(e.first);
        CAPTURE(e.second);
        REQUIRE(directed_boundary.count(e) == 1);
        // Each boundary edge may appear in exactly one loop, once.
        REQUIRE(visited.insert(e).second);
      }
    }
    REQUIRE(static_cast<int>(visited.size()) == num_boundary_edges);
    return num_boundary_edges;
  }

  /// Every interior edge that is not cut must survive as a single interior
  /// edge: both of its faces must keep using the very same edge of the cut
  /// mesh, and that edge must not become a boundary (i.e. the texture seam must
  /// not turn into a geometric cross-section).
  void require_uncut_interior_edges_preserved(
    const Eigen::MatrixXi & F,
    const std::set<Edge> & cuts,
    const CutMesh & cut)
  {
    const std::map<Edge,int> cut_mult = edge_multiplicity(cut.F);
    for(const auto & kv : edge_corners(F))
    {
      if(kv.second.size() != 2) { continue; } // only interior edges
      if(cuts.count(kv.first)) { continue; }
      const Corner c0 = kv.second[0];
      const Corner c1 = kv.second[1];
      const Edge e0 = edge_key(cut.F(c0.first,c0.second),
                               cut.F(c0.first,(c0.second+1)%3));
      const Edge e1 = edge_key(cut.F(c1.first,c1.second),
                               cut.F(c1.first,(c1.second+1)%3));
      CAPTURE(kv.first.first);
      CAPTURE(kv.first.second);
      REQUIRE(e0 == e1);
      REQUIRE(cut_mult.at(e0) == 2);
    }
  }

  /// Every cut (seam) edge must have its two sides separated: the two incident
  /// faces must reference different edges of the cut mesh, each on the
  /// boundary.
  void require_seam_sides_separated(
    const Eigen::MatrixXi & F,
    const Eigen::MatrixXi & seam_rows,
    const CutMesh & cut)
  {
    const std::map<Edge,int> cut_mult = edge_multiplicity(cut.F);
    REQUIRE(seam_rows.rows() > 0);
    for(int r = 0;r<seam_rows.rows();r++)
    {
      const int f0 = seam_rows(r,0);
      const int k0 = seam_rows(r,1);
      const int f1 = seam_rows(r,2);
      const int k1 = seam_rows(r,3);
      const Edge e0 = edge_key(cut.F(f0,k0),cut.F(f0,(k0+1)%3));
      const Edge e1 = edge_key(cut.F(f1,k1),cut.F(f1,(k1+1)%3));
      CAPTURE(r);
      REQUIRE(e0 != e1);
      REQUIRE(cut_mult.at(e0) == 1);
      REQUIRE(cut_mult.at(e1) == 1);
    }
  }

  /// Count boundary edges of the cut mesh grouped by the original edge they
  /// come from. A genuine boundary edge maps back to an edge with one incident
  /// face; a seam-derived boundary edge maps back to a cut edge.
  struct BoundaryOrigin
  {
    int real_boundary;
    int seam;
    int uncut_interior;
  };

  BoundaryOrigin classify_boundary_edges(
    const Eigen::MatrixXi & F,
    const CutMesh & cut,
    const std::set<Edge> & seam_edges)
  {
    const std::map<Edge,int> orig_mult = edge_multiplicity(F);
    const std::map<Edge,int> cut_mult = edge_multiplicity(cut.F);
    BoundaryOrigin counts = {0,0,0};
    for(int f = 0;f<cut.F.rows();f++)
    {
      for(int k = 0;k<3;k++)
      {
        if(cut_mult.at(edge_key(cut.F(f,k),cut.F(f,(k+1)%3))) != 1)
        {
          continue;
        }
        const Edge e = edge_key(cut.I(cut.F(f,k)),cut.I(cut.F(f,(k+1)%3)));
        if(orig_mult.at(e) == 1) { counts.real_boundary++; }
        else if(seam_edges.count(e)) { counts.seam++; }
        else { counts.uncut_interior++; }
      }
    }
    return counts;
  }

  /// Require two edge sets to agree, with readable output on failure.
  void require_same_edges(
    const std::set<Edge> & actual,
    const std::set<Edge> & expected)
  {
    INFO("expected " << expected.size() << " edges, got " << actual.size());
    for(const Edge & e : expected)
    {
      CAPTURE(e.first);
      CAPTURE(e.second);
      REQUIRE(actual.count(e) == 1);
    }
    for(const Edge & e : actual)
    {
      CAPTURE(e.first);
      CAPTURE(e.second);
      REQUIRE(expected.count(e) == 1);
    }
  }

  /// The attribute mapping reconstructed by igl::unzip_corners must agree with
  /// the per-corner attributes of the cut mesh: U(G(f,k),0) is the new vertex
  /// and U(G(f,k),1) the texture index used by corner (f,k).
  void require_unzip_corners_consistent(
    const Eigen::MatrixXi & Fn,
    const Eigen::MatrixXi & FTC,
    const int num_new_vertices)
  {
    Eigen::MatrixXi U,G;
    Eigen::VectorXi J;
    // unzip_corners takes non-const references, so hand it copies.
    std::vector<std::reference_wrapper<Eigen::MatrixXi> > attributes;
    Eigen::MatrixXi Fn_copy = Fn;
    Eigen::MatrixXi FTC_copy = FTC;
    attributes.push_back(Fn_copy);
    attributes.push_back(FTC_copy);
    igl::unzip_corners<Eigen::MatrixXi>(attributes,U,G,J);
    REQUIRE(U.rows() == num_new_vertices);
    REQUIRE(G.rows() == Fn.rows());
    for(int f = 0;f<Fn.rows();f++)
    {
      for(int k = 0;k<3;k++)
      {
        CAPTURE(f);
        CAPTURE(k);
        REQUIRE(U(G(f,k),0) == Fn(f,k));
        REQUIRE(U(G(f,k),1) == FTC(f,k));
      }
    }
  }
}

TEST_CASE("cut_mesh: crossing texture seams", "[igl]")
{
  // 3x3 grid of quads (8 triangles) cut along four seams meeting at the centre
  // vertex 4: a texture cross whose four arms end on the mesh boundary.
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  make_grid_mesh(3,3,V,F);

  const std::set<Edge> seams = {
    edge_key(1,4), edge_key(3,4), edge_key(4,5), edge_key(4,7)};

  int num_sectors = 0;
  const std::vector<int> sectors = corner_sectors(F,seams,num_sectors);
  Eigen::MatrixXd TC;
  Eigen::MatrixXi FTC;
  texture_from_sectors(V,F,sectors,num_sectors,TC,FTC);

  // The texture layout reproduces exactly the requested seams, and the mesh
  // boundary is reported as boundaries (not as seams).
  Eigen::MatrixXi seam_rows,boundary_rows,foldover_rows;
  igl::seam_edges(V,TC,F,FTC,seam_rows,boundary_rows,foldover_rows);
  REQUIRE(seam_rows.rows() == 4);
  REQUIRE(foldover_rows.rows() == 0);
  require_same_edges(edge_set_from_rows(F,seam_rows),seams);
  require_same_edges(edge_set_from_rows(F,boundary_rows),boundary_edge_set(F));

  const CutMesh cut = apply_cut(V,F,flags_for_edges(F,seams));
  REQUIRE(cut.V.rows() == 16); // 9 original vertices + 7 seam copies
  REQUIRE(cut.F.rows() == 8);

  // 1. Every corner keeps its original geometry and its texture index.
  require_geometric_provenance(V,F,cut);
  require_attribute_consistency(F,FTC,sectors,cut);
  const auto copies = copies_by_original(cut);
  REQUIRE(copies.at(0).size() == 1);
  REQUIRE(copies.at(1).size() == 2);
  REQUIRE(copies.at(3).size() == 2);
  REQUIRE(copies.at(4).size() == 4); // crossing point of four seams
  REQUIRE(copies.at(5).size() == 2);
  REQUIRE(copies.at(7).size() == 2);

  // 2. Boundary loops are valid and each seam side became its own boundary.
  const int num_boundary = require_valid_boundary_loops(cut.F);
  REQUIRE(num_boundary == 16); // 8 original + 2 per seam
  std::vector<std::vector<int> > loops;
  igl::boundary_loop(cut.F,loops);
  REQUIRE(loops.size() == 4);
  for(const auto & loop : loops) { REQUIRE(loop.size() == 4); }
  REQUIRE(igl::is_edge_manifold(cut.F));
  Eigen::MatrixXi E;
  igl::edges(cut.F,E);
  REQUIRE(cut.V.rows() - E.rows() + cut.F.rows() == 4);
  Eigen::VectorXi components;
  igl::facet_components(cut.F,components);
  REQUIRE(components.maxCoeff() + 1 == 4);

  // 3. Seam sides are separated; uncut interior edges did not become sections.
  require_seam_sides_separated(F,seam_rows,cut);
  require_uncut_interior_edges_preserved(F,seams,cut);

  const BoundaryOrigin origins = classify_boundary_edges(F,cut,seams);
  REQUIRE(origins.real_boundary == 8);
  REQUIRE(origins.seam == 8);
  REQUIRE(origins.uncut_interior == 0);

  require_unzip_corners_consistent(cut.F,FTC,cut.V.rows());
}

TEST_CASE("cut_mesh: texture seam touching the boundary", "[igl]")
{
  // A single seam edge from interior vertex 4 to boundary vertex 1. Only the
  // boundary vertex needs duplicating (the interior end is a dangling seam).
  Eigen::MatrixXd V(9,3);
  V << 0,0,0,
       1,0,0,
       2,0,0,
       0,1,0,
       1,1,0,
       2,1,0,
       0,2,0,
       1,2,0,
       2,2,0;
  Eigen::MatrixXi F(8,3);
  F << 0,1,4,
       0,4,3,
       1,2,5,
       1,5,4,
       3,4,7,
       3,7,6,
       4,5,8,
       4,8,7;

  const std::set<Edge> seams = {edge_key(1,4)};

  int num_sectors = 0;
  const std::vector<int> sectors = corner_sectors(F,seams,num_sectors);
  Eigen::MatrixXd TC;
  Eigen::MatrixXi FTC;
  texture_from_sectors(V,F,sectors,num_sectors,TC,FTC);

  Eigen::MatrixXi seam_rows,boundary_rows,foldover_rows;
  igl::seam_edges(V,TC,F,FTC,seam_rows,boundary_rows,foldover_rows);
  REQUIRE(seam_rows.rows() == 1);
  REQUIRE(foldover_rows.rows() == 0);
  require_same_edges(edge_set_from_rows(F,seam_rows),seams);
  require_same_edges(edge_set_from_rows(F,boundary_rows),boundary_edge_set(F));

  const CutMesh cut = apply_cut(V,F,flags_for_edges(F,seams));
  REQUIRE(cut.V.rows() == 10);
  REQUIRE(cut.F.rows() == 8);

  require_geometric_provenance(V,F,cut);
  require_attribute_consistency(F,FTC,sectors,cut);
  const auto copies = copies_by_original(cut);
  REQUIRE(copies.at(1).size() == 2); // seam meets the mesh boundary here
  REQUIRE(copies.at(4).size() == 1); // dangling seam end does not split

  const int num_boundary = require_valid_boundary_loops(cut.F);
  REQUIRE(num_boundary == 10); // 8 original + 2 seam sides
  std::vector<std::vector<int> > loops;
  igl::boundary_loop(cut.F,loops);
  REQUIRE(loops.size() == 1);
  REQUIRE(loops[0].size() == 10);

  require_seam_sides_separated(F,seam_rows,cut);
  require_uncut_interior_edges_preserved(F,seams,cut);

  // Seam-derived boundary edges map back to an interior original edge, while
  // the genuine boundary edges map back to boundary edges: the two kinds of
  // boundary are distinguishable.
  const BoundaryOrigin origins = classify_boundary_edges(F,cut,seams);
  REQUIRE(origins.real_boundary == 8);
  REQUIRE(origins.seam == 2);
  REQUIRE(origins.uncut_interior == 0);
  REQUIRE(igl::is_edge_manifold(cut.F));
  Eigen::VectorXi components;
  igl::facet_components(cut.F,components);
  REQUIRE(components.maxCoeff() + 1 == 1);

  // Flagging only one of the two incident faces is documented to be enough.
  const CutMesh one_sided =
    apply_cut(V,F,flags_for_first_side(F,seams));
  REQUIRE(one_sided.V.rows() == cut.V.rows());
  REQUIRE(one_sided.F == cut.F);
  REQUIRE(one_sided.I == cut.I);
}

TEST_CASE("cut_mesh: vertex duplicated by many seams", "[igl]")
{
  // Six triangles around vertex 0, cut along every spoke: vertex 0 must be
  // copied once per incident triangle and each copy must keep only its own
  // corner's texture index.
  Eigen::MatrixXd V(7,3);
  // M_PI is not part of standard C++, so use an explicit constant.
  const double two_pi = 6.283185307179586;
  V.row(0) << 0,0,0;
  for(int i = 0;i<6;i++)
  {
    const double angle = two_pi*static_cast<double>(i)/6.0;
    V.row(1+i) << std::cos(angle),std::sin(angle),0;
  }
  Eigen::MatrixXi F(6,3);
  for(int i = 0;i<6;i++) { F.row(i) << 0,1+i,1+(i+1)%6; }

  std::set<Edge> seams;
  for(int i = 0;i<6;i++) { seams.insert(edge_key(0,1+i)); }

  int num_sectors = 0;
  const std::vector<int> sectors = corner_sectors(F,seams,num_sectors);
  Eigen::MatrixXd TC;
  Eigen::MatrixXi FTC;
  texture_from_sectors(V,F,sectors,num_sectors,TC,FTC);

  Eigen::MatrixXi seam_rows,boundary_rows,foldover_rows;
  igl::seam_edges(V,TC,F,FTC,seam_rows,boundary_rows,foldover_rows);
  REQUIRE(seam_rows.rows() == 6);
  REQUIRE(foldover_rows.rows() == 0);
  require_same_edges(edge_set_from_rows(F,seam_rows),seams);

  const CutMesh cut = apply_cut(V,F,flags_for_edges(F,seams));
  REQUIRE(cut.V.rows() == 18); // 7 original + 5 extra copies of vertex 0 + 6 rim copies
  REQUIRE(cut.F.rows() == 6);

  require_geometric_provenance(V,F,cut);
  require_attribute_consistency(F,FTC,sectors,cut);
  const auto copies = copies_by_original(cut);
  REQUIRE(copies.at(0).size() == 6);
  for(int v = 1;v<7;v++) { REQUIRE(copies.at(v).size() == 2); }

  // Each copy of the centre vertex carries a distinct texture index, and it is
  // the index of the face that owns it.
  std::set<int> centre_textures;
  for(const int nv : copies.at(0))
  {
    std::set<int> corner_textures;
    for(int f = 0;f<cut.F.rows();f++)
    {
      for(int k = 0;k<3;k++)
      {
        if(cut.F(f,k) == nv) { corner_textures.insert(FTC(f,k)); }
      }
    }
    REQUIRE(corner_textures.size() == 1);
    centre_textures.insert(*corner_textures.begin());
  }
  REQUIRE(centre_textures.size() == 6);

  const int num_boundary = require_valid_boundary_loops(cut.F);
  REQUIRE(num_boundary == 18); // 6 original + 2 per seam
  std::vector<std::vector<int> > loops;
  igl::boundary_loop(cut.F,loops);
  REQUIRE(loops.size() == 6);
  for(const auto & loop : loops) { REQUIRE(loop.size() == 3); }
  REQUIRE(igl::is_edge_manifold(cut.F));
  Eigen::VectorXi components;
  igl::facet_components(cut.F,components);
  REQUIRE(components.maxCoeff() + 1 == 6);
  require_seam_sides_separated(F,seam_rows,cut);
  require_unzip_corners_consistent(cut.F,FTC,cut.V.rows());

  // Cutting only three alternating spokes merges pairs of triangles into three
  // charts, giving three copies of the centre vertex.
  std::set<Edge> alternating = {edge_key(0,1),edge_key(0,3),edge_key(0,5)};
  int alt_sectors = 0;
  const std::vector<int> alt_labels =
    corner_sectors(F,alternating,alt_sectors);
  Eigen::MatrixXd alt_TC;
  Eigen::MatrixXi alt_FTC;
  texture_from_sectors(V,F,alt_labels,alt_sectors,alt_TC,alt_FTC);
  const CutMesh alt_cut = apply_cut(V,F,flags_for_edges(F,alternating));
  require_geometric_provenance(V,F,alt_cut);
  require_attribute_consistency(F,alt_FTC,alt_labels,alt_cut);
  REQUIRE(alt_cut.V.rows() == 12);
  REQUIRE(copies_by_original(alt_cut).at(0).size() == 3);
  REQUIRE(require_valid_boundary_loops(alt_cut.F) == 12);
  std::vector<std::vector<int> > alt_loops;
  igl::boundary_loop(alt_cut.F,alt_loops);
  REQUIRE(alt_loops.size() == 3);
  for(const auto & loop : alt_loops) { REQUIRE(loop.size() == 4); }
}

TEST_CASE("cut_mesh: closed texture seam", "[igl]")
{
  // Closed seam loop around the cell (1,1) of a 4x4 grid. Cutting it must not
  // leak into the surrounding mesh: every untouched interior edge stays an
  // interior edge and every genuine boundary edge stays on the boundary.
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  make_grid_mesh(4,4,V,F);

  // Cell (1,1) has corners 5=(1,1), 6=(2,1), 9=(1,2), 10=(2,2).
  const std::set<Edge> seams = {
    edge_key(5,6), edge_key(6,10), edge_key(10,9), edge_key(9,5)};

  int num_sectors = 0;
  const std::vector<int> sectors = corner_sectors(F,seams,num_sectors);
  Eigen::MatrixXd TC;
  Eigen::MatrixXi FTC;
  texture_from_sectors(V,F,sectors,num_sectors,TC,FTC);

  Eigen::MatrixXi seam_rows,boundary_rows,foldover_rows;
  igl::seam_edges(V,TC,F,FTC,seam_rows,boundary_rows,foldover_rows);
  REQUIRE(seam_rows.rows() == 4);
  REQUIRE(foldover_rows.rows() == 0);
  require_same_edges(edge_set_from_rows(F,seam_rows),seams);
  require_same_edges(edge_set_from_rows(F,boundary_rows),boundary_edge_set(F));
  REQUIRE(boundary_rows.rows() == 12);

  const CutMesh cut = apply_cut(V,F,flags_for_edges(F,seams));
  REQUIRE(cut.V.rows() == 20);
  REQUIRE(cut.F.rows() == 18);

  require_geometric_provenance(V,F,cut);
  require_attribute_consistency(F,FTC,sectors,cut);
  const auto copies = copies_by_original(cut);
  REQUIRE(copies.at(5).size() == 2);
  REQUIRE(copies.at(6).size() == 2);
  REQUIRE(copies.at(9).size() == 2);
  REQUIRE(copies.at(10).size() == 2);

  const int num_boundary = require_valid_boundary_loops(cut.F);
  REQUIRE(num_boundary == 20); // 12 original + 2 per seam
  std::vector<std::vector<int> > loops;
  igl::boundary_loop(cut.F,loops);
  REQUIRE(loops.size() == 3);
  std::vector<int> loop_lengths;
  for(const auto & loop : loops) { loop_lengths.push_back(loop.size()); }
  std::sort(loop_lengths.begin(),loop_lengths.end());
  REQUIRE(loop_lengths[0] == 4);
  REQUIRE(loop_lengths[1] == 4);
  REQUIRE(loop_lengths[2] == 12);
  REQUIRE(igl::is_edge_manifold(cut.F));
  Eigen::MatrixXi E;
  igl::edges(cut.F,E);
  REQUIRE(cut.V.rows() - E.rows() + cut.F.rows() == 1);

  require_seam_sides_separated(F,seam_rows,cut);
  require_uncut_interior_edges_preserved(F,seams,cut);

  const BoundaryOrigin origins = classify_boundary_edges(F,cut,seams);
  REQUIRE(origins.real_boundary == 12);
  REQUIRE(origins.seam == 8);
  REQUIRE(origins.uncut_interior == 0);

  require_unzip_corners_consistent(cut.F,FTC,cut.V.rows());
}

TEST_CASE("cut_mesh: real texture coordinates reach every corner", "[igl]")
{
  // tet.obj carries genuine per-corner texture indices: three of its edges are
  // UV seams on a closed surface. Every corner of the cut mesh must keep its
  // original texture index, and the geometric vertex it points at must be a
  // copy of the corner's original vertex.
  Eigen::MatrixXd V,TC,CN;
  Eigen::MatrixXi F,FTC,FN;
  REQUIRE(igl::readOBJ(test_common::data_path("tet.obj"),V,TC,CN,F,FTC,FN));

  Eigen::MatrixXi seam_rows,boundary_rows,foldover_rows;
  igl::seam_edges(V,TC,F,FTC,seam_rows,boundary_rows,foldover_rows);
  REQUIRE(seam_rows.rows() == 3);
  REQUIRE(boundary_rows.rows() == 0); // closed surface
  REQUIRE(foldover_rows.rows() == 0);
  const std::set<Edge> seams = edge_set_from_rows(F,seam_rows);

  const CutMesh cut = apply_cut(V,F,flags_for_edges(F,seams));
  REQUIRE(cut.V.rows() == 6); // vertex 0 is copied three times
  REQUIRE(cut.F.rows() == 4);

  // Geometric provenance.
  require_geometric_provenance(V,F,cut);

  // Per-corner texture provenance: each new vertex has one texture index, and
  // that index is the one its corners had in the input.
  std::map<int,std::set<int> > textures_of_new_vertex;
  for(int f = 0;f<cut.F.rows();f++)
  {
    for(int k = 0;k<3;k++)
    {
      textures_of_new_vertex[cut.F(f,k)].insert(FTC(f,k));
    }
  }
  for(const auto & kv : textures_of_new_vertex)
  {
    REQUIRE(kv.second.size() == 1);
  }
  const auto copies = copies_by_original(cut);
  REQUIRE(copies.at(0).size() == 3);
  std::set<int> copied_textures;
  for(const int nv : copies.at(0))
  {
    copied_textures.insert(*textures_of_new_vertex.at(nv).begin());
  }
  std::set<int> original_textures;
  for(int f = 0;f<F.rows();f++)
  {
    for(int k = 0;k<3;k++)
    {
      if(F(f,k) == 0) { original_textures.insert(FTC(f,k)); }
    }
  }
  REQUIRE(copied_textures == original_textures);

  // Every face keeps a consistent attribute mapping.
  require_unzip_corners_consistent(cut.F,FTC,cut.V.rows());

  // The surface is closed, so the cut opens it into one disk-like boundary.
  const int num_boundary = require_valid_boundary_loops(cut.F);
  REQUIRE(num_boundary == 6);
  std::vector<std::vector<int> > loops;
  igl::boundary_loop(cut.F,loops);
  REQUIRE(loops.size() == 1);
  REQUIRE(loops[0].size() == 6);
  REQUIRE(igl::is_edge_manifold(cut.F));
  Eigen::MatrixXi E;
  igl::edges(cut.F,E);
  REQUIRE(cut.V.rows() - E.rows() + cut.F.rows() == 1);

  require_seam_sides_separated(F,seam_rows,cut);
  require_uncut_interior_edges_preserved(F,seams,cut);
  const BoundaryOrigin origins = classify_boundary_edges(F,cut,seams);
  REQUIRE(origins.real_boundary == 0);
  REQUIRE(origins.seam == 6);
  REQUIRE(origins.uncut_interior == 0);
}

TEST_CASE("cut_mesh: seam, boundary and foldover are distinguished", "[igl]")
{
  // Two pairs of triangles: the first pair is folded over in UV space, the
  // second pair is separated by a texture seam. Every edge of both pairs is on
  // the mesh boundary.
  Eigen::MatrixXd V(8,3);
  V << 0,0,0,
       1,0,0,
       0.5,1,0,
       0.5,-0.6,0,
       3,0,0,
       4,0,0,
       3.5,1,0,
       4.5,-0.6,0;
  Eigen::MatrixXi F(4,3);
  F << 0,1,2,
       1,0,3,
       4,5,6,
       5,4,7;
  Eigen::MatrixXd TC(12,2);
  TC << 0,0,       // 0: folded pair, corner v0
        1,0,       // 1: folded pair, corner v1
        0.5,1,     // 2: folded pair, corner v2
        0.5,0.6,   // 3: folded pair, corner v3 (same side of edge 0-1)
        3,0,       // 4: seam pair, v4 on face 2
        4,0,       // 5: seam pair, v5 on face 2
        3.5,1,     // 6: seam pair, v6 on face 2
        3.5,-0.6,  // 7: unused
        3,0,       // 8: seam pair, v5 on face 3 (distinct index)
        4,0,       // 9: seam pair, v4 on face 3 (distinct index)
        3.5,-0.6;  // 10: seam pair, v7 on face 3
  Eigen::MatrixXi FTC(4,3);
  FTC << 0,1,2,
         1,0,3,
         4,5,6,
         8,9,10;

  Eigen::MatrixXi seam_rows,boundary_rows,foldover_rows;
  igl::seam_edges(V,TC,F,FTC,seam_rows,boundary_rows,foldover_rows);

  const std::set<Edge> seam_edges = edge_set_from_rows(F,seam_rows);
  const std::set<Edge> foldover_edges = edge_set_from_rows(F,foldover_rows);
  const std::set<Edge> boundary_edges = edge_set_from_rows(F,boundary_rows);
  const std::set<Edge> true_boundary_edges = boundary_edge_set(F);

  // 3. The three classes are disjoint and correct.
  REQUIRE(seam_edges.size() == 1);
  REQUIRE(foldover_edges.size() == 1);
  require_same_edges(seam_edges,std::set<Edge>{edge_key(4,5)});
  require_same_edges(foldover_edges,std::set<Edge>{edge_key(0,1)});
  require_same_edges(boundary_edges,true_boundary_edges);
  REQUIRE(boundary_edges.size() == 8);
  for(const Edge & e : seam_edges) { REQUIRE(foldover_edges.count(e) == 0); }
  for(const Edge & e : seam_edges) { REQUIRE(boundary_edges.count(e) == 0); }
  for(const Edge & e : foldover_edges) { REQUIRE(boundary_edges.count(e) == 0); }

  // Cutting the seam only.
  const CutMesh cut = apply_cut(V,F,flags_for_edges(F,seam_edges));
  REQUIRE(cut.V.rows() == 10);
  REQUIRE(cut.F.rows() == 4);

  require_geometric_provenance(V,F,cut);
  require_seam_sides_separated(F,seam_rows,cut);

  // The foldover pair is untouched, and the folded edge is still an interior
  // edge shared by the two triangles.
  REQUIRE(cut.F.row(0) == F.row(0));
  REQUIRE(cut.F.row(1) == F.row(1));
  const std::map<Edge,int> cut_mult = edge_multiplicity(cut.F);
  const Corner fold0 = edge_corners(F).at(edge_key(0,1))[0];
  const Corner fold1 = edge_corners(F).at(edge_key(0,1))[1];
  const Edge fold_edge0 = edge_key(cut.F(fold0.first,fold0.second),
                                   cut.F(fold0.first,(fold0.second+1)%3));
  const Edge fold_edge1 = edge_key(cut.F(fold1.first,fold1.second),
                                   cut.F(fold1.first,(fold1.second+1)%3));
  REQUIRE(fold_edge0 == fold_edge1);
  REQUIRE(cut_mult.at(fold_edge0) == 2);

  // The seam pair is split into two charts, and every genuine boundary edge is
  // still on the boundary.
  const int num_boundary = require_valid_boundary_loops(cut.F);
  REQUIRE(num_boundary == 10); // 8 original + 2 seam sides
  std::vector<std::vector<int> > loops;
  igl::boundary_loop(cut.F,loops);
  REQUIRE(loops.size() == 3);
  std::vector<int> loop_lengths;
  for(const auto & loop : loops) { loop_lengths.push_back(loop.size()); }
  std::sort(loop_lengths.begin(),loop_lengths.end());
  REQUIRE(loop_lengths[0] == 3);
  REQUIRE(loop_lengths[1] == 3);
  REQUIRE(loop_lengths[2] == 4);

  const auto copies = copies_by_original(cut);
  REQUIRE(copies.at(0).size() == 1); // foldover does not duplicate
  REQUIRE(copies.at(4).size() == 2); // seam does
  REQUIRE(copies.at(5).size() == 2);

  const BoundaryOrigin origins = classify_boundary_edges(F,cut,seam_edges);
  REQUIRE(origins.real_boundary == 8);
  REQUIRE(origins.seam == 2);
  REQUIRE(origins.uncut_interior == 0);

  std::map<int,std::set<int> > textures_of_new_vertex;
  for(int f = 0;f<cut.F.rows();f++)
  {
    for(int k = 0;k<3;k++)
    {
      textures_of_new_vertex[cut.F(f,k)].insert(FTC(f,k));
    }
  }
  for(const auto & kv : textures_of_new_vertex)
  {
    REQUIRE(kv.second.size() == 1);
  }
  require_unzip_corners_consistent(cut.F,FTC,cut.V.rows());
}

TEST_CASE("cut_mesh: foldover alone is not a cut", "[igl]")
{
  // Mirrored UVs over a shared edge: a foldover, not a seam. The cut must be a
  // no-op and the two triangles must keep sharing their edge.
  Eigen::MatrixXd V(4,3);
  V << 0,0,0,
       1,0,0,
       0.5,1,0,
       0.5,0.6,0;
  Eigen::MatrixXi F(2,3);
  F << 0,1,2,
       1,0,3;
  Eigen::MatrixXd TC(4,2);
  TC << 0,0,
        1,0,
        0.5,1,
        0.5,0.6;
  Eigen::MatrixXi FTC(2,3);
  FTC << 0,1,2,
         1,0,3;

  Eigen::MatrixXi seam_rows,boundary_rows,foldover_rows;
  igl::seam_edges(V,TC,F,FTC,seam_rows,boundary_rows,foldover_rows);
  REQUIRE(seam_rows.rows() == 0);
  REQUIRE(foldover_rows.rows() == 1);
  require_same_edges(edge_set_from_rows(F,boundary_rows),boundary_edge_set(F));

  const CutMesh cut = apply_cut(V,F,flags_for_edges(
        F,edge_set_from_rows(F,seam_rows)));
  REQUIRE(cut.V.rows() == V.rows());
  REQUIRE(cut.F == F);
  const std::map<Edge,int> mult = edge_multiplicity(cut.F);
  REQUIRE(mult.at(edge_key(0,1)) == 2); // still shared
  REQUIRE(require_valid_boundary_loops(cut.F) == 4);
}

