.. _badcy_tutorial:

Bisphenol-A Dicyanate (BADCy) Thermoset
=======================================

This tutorial covers a **cyanate-ester thermoset** modelled topologically
as a step-growth aryl-ether network plus a postcure topology-repair
stage.  Two chemistry choices are unique to this example:

* **The triazine crosslink ring is pre-formed in the monomer**, not
  constructed during cure.  Real cured BADCy contains 1,3,5-triazine
  rings that arise from cyclotrimerization of three R-O-C#N cyanate
  end-groups during cure; modelling that mechanism literally requires
  closing a three-way C-N ring during the iterative CURE loop, which
  is awkward to drive deterministically.  Instead we use bare
  1,3,5-triazine as a trifunctional crosslinker monomer and pair it
  with bisphenol-A (BPA) as a difunctional bridge in a simple A2+B3
  step-growth ether substitution.  The cured-network *topology* is
  identical to that of a real cyclotrimerized BADCy thermoset, but the
  cure event is a pre-existing-ring aryl substitution rather than
  ring-forming cyclotrimerization.

* **A postcure topology-repair stage** converts the topological
  artifacts of incomplete cure into chemically realistic residuals.
  At any finite conversion the A2+B3 model leaves free BPA-OH groups
  and bare triazine C-H positions — species that don't exist in real
  undercured BADCy, where any cyanate that didn't cyclotrimerize stays
  as an intact -O-C#N end-group.  The new ``postcure_repair`` stage
  dismantles every incomplete triazine (fewer than 3 bonded BPAs) into
  three -C#N fragments and reattaches them to the BPAs they were
  already bonded to (in place) or transfers them to the nearest
  unreacted BPA-OH (free caps).  Atom conservation across the system
  is exact; the post-repair network has the BPA-O-C#N residual
  end-groups a real undercured BADCy thermoset carries.

The repair architecture itself is documented in detail on the
:ref:`postcure-repair user-guide page <postcure_repair>`; this
tutorial focuses on the BADCy-specific instance and what shows up in
the run log when you actually drive it.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   monomers
   reactions
   configuration
   run
   results
   postsim
