.. _htpb_configuration:

The Configuration File
----------------------

The complete ``5-htpb-ipdi.yaml`` from the depot:

.. literalinclude:: ../../../../src/htpolynet/resources/example_depot/5-htpb-ipdi.yaml
   :language: yaml

The ``Title``, ``gromacs``, ``ambertools``, ``densification``,
``precure``, ``CURE``, and ``postcure`` blocks follow the same
conventions as the other tutorials.  Two YAML-level features are
distinctive to this example.

GAFF type-conflict resolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   GAFF:
     resolve_type_discrepancies:
       - typename: dihedraltypes
         funcidx: 4
         rule: stiffest

When two templates assign different parameters to the same atom-type
quartet (a real possibility when many small dimers are independently
parameterized by ``antechamber``), htpolynet would otherwise emit
both into the final ``[ dihedraltypes ]`` section and let GROMACS
pick the second.  This block tells htpolynet to resolve the conflict
by **keeping the stiffest dihedral** of each conflicting pair.  HTPB
is the only depot example that triggers this code path — the dimers
and trimers built by the param-stage reactions occasionally produce
inconsistent dihedral assignments around the inner C=C double bonds.

If you skip this block on this YAML, the build will warn loudly
about overlapping dihedral types; the resulting force field will run
but its torsional energies near the HTPB backbone double bonds are
not what the parameterization intended.

Low initial density
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   densification:
     initial_density: 50.0  # kg/m3
     equilibration:
       - ensemble: min
       - ensemble: nvt
         temperature: 600
         ps: 10
       - ensemble: npt
         temperature: 600
         pressure: 10
         ps: 100
         repeat: 20

The ``initial_density: 50.0`` is several times lower than the other
examples (200-300).  Long HTPB chains can interpenetrate
catastrophically if placed too close together in the initial liquid;
starting at a low density and using **20 NPT repeats** at 600 K / 10
bar to gradually densify avoids that.  The 20 repeats dominate the
precure stage and account for most of the ~1 hour precure wall-clock.

Generated templates
^^^^^^^^^^^^^^^^^^^

Combining the constituent declarations with the 16 reactions (6
param + 8 build + 2 cure):

* 4 small constituents (``OB``, ``TB``, ``TBO``, ``IPD``);
* 6 param-stage products (``A2``, ``AO``, ``OBT``, ``OB3``, ``A3``,
  ``A4``);
* 16 build-stage intermediates from the ``polymerization`` reaction's
  ``procession: count: 15`` (named ``A18_I0`` through ``A18_I15``);
* 7 other build-stage products (``O19``, ``DHT1``, ``DHT``,
  ``THT1``, ``THT2``, ``THT``, plus 1 stereoisomer);
* 2 cure-stage products (``UR1``, ``UR2``).

Total = 35 templates, as reported by the diagnostic log at startup:

.. code-block:: text

   INFO> 35 molecules detected in 5-htpb-ipdi.yaml
   INFO>                       explicit: 34
   INFO>     implied by stereochemistry: 1
   INFO>            implied by symmetry: 0

The 1 stereochemistry-implied molecule is the second diastereomer of
``IPD`` (its ``C3`` stereocenter; the chain assembly itself has no
stereocenters).  No symmetry-equivalent expansions because no
``symmetry_equivalent_atoms`` block is declared on any constituent.

The next page walks through what to expect when :ref:`actually running
the build <htpb_run>`.
