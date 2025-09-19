//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file orszag_tang.cpp
//! \brief Problem generator for the Orszag Tang vortex.
//!
//! REFERENCE: Orszag & Tang (J. Fluid Mech., 90, 129, 1998) and
//! https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html
//========================================================================================

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"

namespace bondi {
using namespace parthenon::driver::prelude;

Real rho_infty, cs_infty, ur_infty, pres_infty, en_den_infty ,  polytropic_constant,gamma, gm1, GN, MBH;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  Units units(pin);
  GN = units.gravitational_constant();
  gamma = pin->GetReal("hydro", "gamma");
  gm1 = (gamma - 1.0);
  MBH = pin->GetReal("problem/bondi","mbh");
  rho_infty = pin->GetReal("problem/bondi/gas", "rho");
  ur_infty = pin->GetReal("problem/bondi/gas", "ur");
  cs_infty = pin->GetReal("problem/bondi/gas", "cs");

  // For polytropic gas : P = k ρ^Γ
  //                    : cs^2 = k Γ ρ^(Γ-1)
  polytropic_constant = cs_infty*cs_infty / gamma / std::pow(rho_infty,gm1);
  pres_infty = polytropic_constant * std::pow(rho_infty,gamma);

  // Units of the polytropic constant in case it is useful
  Real k_units_cgs = units.dyne_cm2() / std::pow(units.g_cm3(),gamma);
  // Energy density at infinity
  // Energy density = P/(Γ-1) + 1/2 u^2 ρ
  Real en_den_infty = pres_infty /gm1 + 0.5 * ur_infty * ur_infty * rho_infty;

  std::stringstream msg;
  msg << std::setprecision(2);
  msg << "######################################" << std::endl;
  msg << "######  Bondi problem generator" << std::endl;
  msg << "#### Input parameters" << std::endl;
  msg << "## Density at infinity " << rho_infty / units.g_cm3() << " g/cm^3" << std::endl;
  msg << "## Radial velocity at infinity: " << ur_infty / units.km_s() << " km/s" << std::endl;
  msg << "## Sound speed at infinity: " << cs_infty / units.km_s() << " km/s" << std::endl;
  msg << "#### Derived parameters" << std::endl;
  msg << "## Pressure at infinity " << pres_infty / units.dyne_cm2() << " dyne/cm^2" << std::endl;
  msg << "## Energy density at infinity " << en_den_infty / units.erg() / (units.cm() * units.cm() * units.cm()) << " erg/cm^3" << std::endl;
  msg << "## Polytropic constant " << polytropic_constant / k_units_cgs << " [cgs]" << std::endl;
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto &coords = pmb->coords;
  auto &prim = mbd->Get("prim").data;


  pmb->par_for(
      "ProblemGenerator: Bondi", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, "Cluster::ProblemGenerator::UniformGas",
            parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                Real volume =  coords.CellVolume(k,j,i) ;
                // A simple setup for now. Needs to be improved
                u(IDN, k, j, i) = rho_infty;
                // TODO: Check if momentum or velocity should be updated or if momentum can be updated
                   // later based on the primitive::velocity
                u(IM1, k, j, i) = ur_infty;
                prim(IV1, k, j, i) = ur_infty;
                u(IEN, k, j, i) = en_den_infty * volume;
                prim(IPR,k,i,j) = pres_infty;
            });
        });
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg) {

  /************************************************************
   * Read Uniform Gas
   ************************************************************/

    hydro_pkg->AddParam<>("rho_infty",rho_infty);
    hydro_pkg->AddParam<>("ur_infty", ur_infty);
    hydro_pkg->AddParam<>("cs_infty", cs_infty);
}

void BondiOuter(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const auto nb = IndexRange{0, 0};
  const bool fine = false;
  pmb->par_for_bndry(
      "BondiOuter", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
      coarse, fine, KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        cons(IDN, k, j, i) = rho_infty;
        // TODO: Check if velocity can be used for BCs
        cons(IM1, k, j, i) = ur_infty;
      });
}

void GravitationalFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                               const parthenon::Real beta_dt) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GravitationalFieldSrcTerm", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        const Real r = coords.Xc<1>(i);
        const Real g_r = GN * MBH / (r*r);

        // Apply g_r as a source term
        const Real den = prim(IDN, k, j, i);
        const Real src = (r == 0) ? 0 : beta_dt * den * g_r / r;
        cons(IM1, k, j, i) -= src * coords.Xc<1>(i);
        cons(IEN, k, j, i) -= src * coords.Xc<1>(i) * prim(IV1, k, j, i) ;
      });
}

void SphericalSourceTerm(parthenon::MeshData<parthenon::Real> *md,
                         const parthenon::Real beta_dt) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GravitationalFieldSrcTerm", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        const Real r = coords.Xc<1>(i);
        const Real g_r = GN * MBH / (r*r);

        // Apply g_r as a source term
        const Real rhoC = cons(IDN, k, j, i);
        const Real uC = prim(IV1, k, j, i);
        const Real abr = 2/r;
        const Real E_c = cons(IEN,k,j,i);
        const Real p = prim(IPR,k,j,i);

         cons(IDN,i,0,0) += beta_dt *  rhoC * uC * abr;
         cons(IM1,i,0,0) += beta_dt * rhoC * (uC*uC) * abr;
         cons(IEN,i,0,0) += beta_dt *  uC * (E_c+p) * abr;
      });
}

void BondiUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                           const Real beta_dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  GravitationalFieldSrcTerm(md, beta_dt);
  SphericalSourceTerm(md,beta_dt);
};

} // namespace bondi

