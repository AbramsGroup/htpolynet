.. _bgs_tutorial:

BisGMA-Styrene Thermoset
========================

This tutorial covers a free-radical cure of bisGMA with styrene comonomers.
What makes it interesting beyond :ref:`example 1 <ps_tutorial>` is that
``htpolynet`` assembles the bisGMA molecule itself at runtime by reacting
2-hydroxypropyl isopropyl ester (HIE) onto each hydroxyl of bisphenol A
(BPA) via two **param**-stage reactions, then uses the assembled GMA
(and free STY) as the building blocks for the cure.  It also exercises
stereochemistry handling: HIE has two chiral carbons, so each GMA in the
initial liquid is drawn from a racemic pool of 16 diastereomers.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   reactions
   configuration
   run
   results
