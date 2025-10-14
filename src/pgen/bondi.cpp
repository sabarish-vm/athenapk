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
#include <pybind11/embed.h>  // for embedding Python
#include <pybind11/numpy.h>  // for numpy support
#include <tuple>
#include <vector>


// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"

namespace py = pybind11;

namespace bondi {
using namespace parthenon::driver::prelude;

Real rho_infty, cs_infty, ur_infty, pres_infty, en_den_infty ,
polytropic_constant,gamma, gm1, inv_gm1, GN, MBH, rB, cs2_infty,
mdot;

std::tuple<py::array_t<Real>, py::array_t<Real>> init_profile(MeshBlock *pmb){
    py::scoped_interpreter guard{};
    // Set paths
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("append")(PGEN_DIR);
    // Import Python module
    py::module_ mymodule = py::module_::import("bondi");
    // Get Python function
    py::object func = mymodule.attr("soln");

    auto &coords = pmb->coords;
    auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    std::vector<Real> vec;
    // Note that the loop is only over the interior domain
    for (int i = ib.s; i <= ib.e; i++) {
            const Real r = coords.Xc<1>(i);
            vec.push_back(r);
    }
    // Create a numpy array
    py::array_t<Real> input_arr(vec.size(),vec.data());
    // Python call
    py::tuple result = func(gamma,input_arr);
    // Untuple the python tuple
    mdot = result[0].cast<Real>();
    py::array_t<Real> rad_vel = result[1].cast<py::array_t<Real>>();
    py::array_t<Real> rho = result[2].cast<py::array_t<Real>>();
    // Create a cpp tyuple of density and velocity profile to return
    auto ret_tuple = std::make_tuple(rad_vel,rho);
    return ret_tuple;
};

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  Units units(pin);
  GN = units.gravitational_constant();
  gamma = pin->GetReal("hydro", "gamma");
  gm1 = (gamma - 1.0);
  inv_gm1 = 1/gm1;
  MBH = pin->GetReal("problem/bondi","mbh");
  const Real rho_infty_cgs = pin->GetReal("problem/bondi", "rho_infty_cgs");
  const Real cs_infty_cgs = pin->GetReal("problem/bondi", "cs_infty_cgs");
  const Real ur_infty_cgs = pin->GetReal("problem/bondi", "ur_infty_cgs");

  ur_infty = ur_infty_cgs  * units.cm_s();
  cs_infty = cs_infty_cgs  * units.cm_s();
  cs2_infty = cs_infty * cs_infty;
  rho_infty= rho_infty_cgs * units.g_cm3();

  rB = GN*MBH/cs2_infty;

  // For polytropic gas : P = k ρ^Γ
  //                    : cs^2 = k Γ ρ^(Γ-1)
  polytropic_constant = cs_infty*cs_infty / gamma / std::pow(rho_infty,gm1);
  pres_infty = polytropic_constant * std::pow(rho_infty,gamma);

  // Units of the polytropic constant in case it is useful
  Real k_units_cgs = units.dyne_cm2() / std::pow(units.g_cm3(),gamma);

  // Energy density at infinity
  // Energy density = P/(Γ-1) + 1/2 u^2 ρ
  en_den_infty = pres_infty /gm1 + 0.5 * ur_infty * ur_infty * rho_infty;

  std::stringstream msg;
  msg << std::setprecision(2) << '\n';
  msg << "######################################" << '\n';
  msg << "#############  Bondi problem generator" << '\n';
  msg << "###### Input parameters : " << '\n';
  msg << "## Mass of BH : " << MBH << " [code] , " << MBH/units.msun() << " [Msun]" << '\n';
  msg << "## Density at infinity : " << rho_infty << " [code] , " << rho_infty / units.g_cm3() << " [g/cm^3]" << '\n';
  msg << "## Radial velocity at infinity : " << ur_infty << " [code] , " << ur_infty / units.km_s() << " [km/s]" << '\n';
  msg << "## Sound speed at infinity : " << cs_infty << " [code] , " << cs_infty / units.km_s() << " km/s" << '\n';
  msg << "###### Derived parameters" << '\n';
  msg << "## Bondi radius : " << rB << " [code] , " << rB / units.cm() << " [cm]" << '\n';
  msg << "## Pressure at infinity : " << pres_infty << "[code] , " << pres_infty / units.dyne_cm2() << " dyne/cm^2" << '\n';
  msg << "## Energy density at infinity : " << en_den_infty << "[code] , " << en_den_infty / units.erg() / (units.cm() * units.cm() * units.cm()) << " erg/cm^3" << '\n';
  msg << "## Polytropic constant : " << polytropic_constant << "[code] , " << polytropic_constant / k_units_cgs << " [cgs]" << '\n';
  msg << "######################################" << '\n' << std::endl;
  std::cout << msg.str();
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto &coords = pmb->coords;
  auto &prim = mbd->Get("prim").data;
  auto profile_tuple = init_profile(pmb);
  auto rad_vel_vec = std::get<0>(profile_tuple);
  auto rho_vec = std::get<1>(profile_tuple);
 
  pmb->par_for(
      "ProblemGenerator::Bondi", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
            // This index offsetting ensures that we map index ib.s to 0 of the numpy array containing the profile
            const int vec_ind = i-ib.s;

            const Real r = coords.Xc<1>(i);
            const Real dr = coords.CellWidth<1>(k,j,i);
            const Real volume =  4*Kokkos::numbers::pi*r*r*dr ;
            // TODO : Verify volume calculation : CellWidth vs CellVolume

            const Real rho = rho_vec.at(vec_ind);
            const Real rad_vel   = rad_vel_vec.at(vec_ind);

            const Real pressure = polytropic_constant * std::pow(rho,gamma);
            const Real energy_den = pressure * inv_gm1 + 0.5 * rad_vel * rad_vel * rho ;

            u(IDN, k, j, i) = rho;
            u(IM1, k, j, i) = rho * rad_vel;
            u(IEN, k, j, i) = energy_den;

            //TODO: Check if pressure needs to be updated
            // prim(IPR,k,j,i) = pres_infty;
            // TODO: Check if momentum or velocity should be used for ICs
            // prim(IV1, k, j, i) = ur_infty;
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
  auto &coords = pmb->coords;
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  auto &prim = mbd->Get("prim").data;
  const auto nb = IndexRange{0, 0};
  const bool fine = false;
  pmb->par_for_bndry(
      "BondiOuter", nb, IndexDomain::outer_x1, parthenon::TopologicalElement::CC,
      coarse, fine, KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        const Real r = coords.Xc<1>(i);
        const Real dr = coords.CellWidth<1>(k,j,i);
        const Real volume =  4*Kokkos::numbers::pi*r*r*dr ;
        Real mass = rho_infty * volume;
        cons(IDN, k, j, i) = rho_infty;
        // TODO: Check and understand BCs in FVM better
        cons(IM1, k, j, i) = mass * ur_infty;
        prim(IPR,k,j,i) = pres_infty;
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

        const Real rhoC = cons(IDN, k, j, i);
        const Real uC = prim(IV1, k, j, i);
        const Real abr = 2/r;
        const Real E_c = cons(IEN,k,j,i);
        const Real p = prim(IPR,k,j,i);

         cons(IDN,0,0,i) += beta_dt *  rhoC * uC * abr;
         cons(IM1,0,0,i) += beta_dt * rhoC * (uC*uC) * abr;
         cons(IEN,0,0,i) += beta_dt *  uC * (E_c+p) * abr;
      });
}

void BondiUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                           const Real beta_dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  GravitationalFieldSrcTerm(md, beta_dt);
  SphericalSourceTerm(md,beta_dt);
};

} // namespace bondi

