
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
#include "./pgen.hpp"


namespace sedov_shock {
using namespace parthenon::driver::prelude;

const Real rho = 1;
const Real e0 = 10;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto &coords = pmb->coords;
  auto &prim = mbd->Get("prim").data;
 
  pmb->par_for(
      "ProblemGenerator::Bondi", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {

            const Real r = coords.Xc<1>(i);

            u(IDN, k, j, i) = rho;
            u(IM1, k, j, i) = 0.0;
            if(i==ib.s) {
                u(IEN, k, j, i) = e0;
            }
            else {
            u(IEN, k, j, i) = 1e-10;}
    });
}


void SphericalSourceTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                           const Real beta_dt) {
  bondi::SphericalSourceTerm(md,beta_dt);
};

} // namespace sedov

