.. _ps_tutorial:

Polystyrene
===========

This tutorial walks through the first build that actually polymerizes
something: 1000 styrene monomers crosslinked by a single C–C cure
reaction, with a cap reaction that restores the vinyl double bond on
any monomer that did not react.  Compared to
:ref:`example 0 <liquid_styrene_tutorial>` it adds the ``CURE`` and
``reactions`` blocks; everything else (monomer-from-SMILES, the
densification + anneal cascade, the final-results bundle) carries over.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   monomer
   reactions
   configuration
   run
   results
   postsim
