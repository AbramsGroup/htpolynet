.. _badcy_introduction:

Introduction
------------

Set up a clean working directory and pull the example YAML:

.. code-block:: console

   $ mkdir my_badcy
   $ cd my_badcy
   $ htpolynet fetch-example 6
   Fetched 6-cyanate-ester.yaml  (run with: htpolynet run 6-cyanate-ester.yaml)
   $ ls
   6-cyanate-ester.yaml

Self-contained YAML as in the earlier examples — all three constituents
are generated from SMILES with atom-mapping tokens at startup.

What's new in this example
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Topological cure, not mechanistic cure.**  Real BADCy thermosets cure
by `cyclotrimerization
<https://en.wikipedia.org/wiki/Cyanate_ester>`_: three cyanate
end-groups (R-O-C#N) on three different bisphenol-A-dicyanate monomers
come together and close into a 1,3,5-triazine ring with three aryl
ether arms — the characteristic crosslink of a cured cyanate ester.
Modelling that exactly means driving a 3-way C-N ring closure during
the iterative CURE loop, which is awkward to control and forces
``bondcycle_collective`` bypasses for the heteroatom ring.

We sidestep this by treating the triazine ring as **pre-formed**: the
crosslinker monomer is bare 1,3,5-triazine (``TAZ``), and BPA is a
bisphenol-A diol with two reactive phenolic O atoms.  The cure step is
a single A2+B3 aryl ether substitution (``BPA-OH + TAZ-CH ->
BPA-O-TAZ + H2``), which the existing cure machinery handles cleanly
without ring-closure tricks.  At full conversion the network topology
— triazine trifunctional nodes connected by BPA bridges through aryl
ether linkages — is structurally identical to a fully cyclotrimerized
BADCy thermoset.

**A postcure topology-repair stage** then bridges the one gap this
choice creates.  At finite cure conversion the topological model
leaves behind species that real BADCy doesn't carry:

* unreacted phenolic BPA-OH groups (real undercured BADCy has
  BPA-O-C#N instead — the cyanate just didn't cyclotrimerize);
* triazine rings with one or two BPA bonds and one or two bare ring
  C-H positions (real BADCy doesn't form partial triazines — three
  cyanates either come together to close a ring, or none of them do).

A short atom-conservation argument shows these two artefact counts are
**equal** across the whole system, and that for every "incomplete"
triazine ring with ``k`` bonded BPAs, the three ring-C + ring-N atoms
of that ring are exactly what we need to make ``k`` BPA-O-C#N caps
in place plus ``3 - k`` free -C#N fragments to donate to unreacted
BPA-OHs elsewhere.  The :ref:`postcure-repair user-guide page
<postcure_repair>` works through the bookkeeping.  In this tutorial
we'll just point at where the repair stage fires in the log and what
it does to the residue mix.

What you'll see in the build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At ``CURE.controls.desired_conversion: 0.90`` (the value the depot
YAML ships with), a representative run reaches ~90 % cure in 9
iterations, leaves on the order of 60-70 incomplete triazines, and the
postcure repair stage converts them into ~190 BPA-O-C#N caps with
exact heavy-atom conservation.  The repair stage takes about a minute
of wall time; the cure itself dominates the run at 15-20 minutes.

The remaining pages walk through the monomer SMILES, the cure and
repair reactions, the YAML in full, and what to look for in the
diagnostic log and plots.
