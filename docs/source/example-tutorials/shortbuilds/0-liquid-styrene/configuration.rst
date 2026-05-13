.. _liquid_styrene_configuration:

Configuration
-------------

The full ``0-liquid-styrene.yaml`` config is short.  There is no
``CURE``, no ``postcure``, no ``reactions`` block — just enough to
parameterize a single monomer and run a densification cascade.

.. literalinclude:: ../../../../../src/htpolynet/resources/example_depot/0-liquid-styrene.yaml
   :language: yaml

.. todo::

   - Walk through each block: ``Title``, ``gromacs``, ``ambertools``,
     ``constituents``, ``densification``, ``precure``.
   - Note that ``precure`` here is *not* a curing step in the chemical
     sense — it is the post-densification preequilibration + anneal +
     postequilibration cascade that runs even when no cure reactions are
     defined.  (If that name is misleading for a no-cure example, flag
     it as a candidate for renaming.)
   - Discuss the ``mdrun_options.gpu_id: 0`` line: it is left in by
     default but the runtime will strip it automatically with a warning
     if ``gmx`` is not GPU-capable or no GPU is visible.
   - Optionally call out the analysis snippet shown in the comment block
     at the top of the YAML (the ``ck.yaml`` template for
     ``htpolynet analyze``).
