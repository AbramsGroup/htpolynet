.. _molecular_structure_inputs:

Molecular Structure Inputs
--------------------------

``htpolynet`` requires molecular structures in order to generate systems.  It recognizes ``mol2`` and ``pdb`` format files, and it is up to you to make these for your system.  Any ``mol2`` or ``pdb`` file you wish to use as a monomer template should be in the ``./lib/molecules/inputs/`` relative to the current directory in which you issue ``htpolynet run`` or ``htpolynet parameterize``.  There are several ways to make such files from other (e.g., 2-dimensional) structural information, and we will cover two here.  There are also two **very important** things to keep in mind when generating your molecular structure files.

The first **very important** thing is this:  ``htpolynet`` requires **valence-conservation** when polymerizing.  This means that when two atoms are identified as bonding partners (each of which is on a separate molecule, most likely), they each must own at least one **sacrificial hydrogen** atom that is deleted when the bond is formed, thus keeping the valence of each atom constant.  We refer to the **valence-conserving** form of a monomer as its **active** form.  

To illustrate how to handle this, let's consider the simple monomer styrene.  The "inactive" form of styrene is its "actual" structure:

.. image:: pics/STYCC.png

However, for ``htpolynet``, styrene's **active** form is actually ethylbenzene:

.. image:: pics/STY.png

So we need to generate a ``mol2`` and/or ``pdb`` file for an ethylbenzene to use styrene as a monomer in ``htpolynet``.

One way to generate ``mol2`` files is with any one of a number of chemical "sketching" tools.  For example, the `ChemDoodle 2D sketcher <https://web.chemdoodle.com/demos/2d-sketcher>`_:

.. figure:: pics/chemdoodle-2dsketcher-emb.png

    Example of a Chemdoodle 2d-sketcher session for creating an input ``mol2`` file for styrene (ahem, actually ethylbenzene).

Another way is to use `OpenBabel <https://openbabel.org/wiki/Main_Page>`_'s ``obabel`` command.  For example, we can use the `SMILES string <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_ for ethylbenzene to tell ``openbabel`` to generate 3-D coordinates and save to a ``mol2`` (or, alternatively, a ``pdb``) file:

.. code-block:: console

    $ obabel -:"C1=CC=CC=C1CC" -ismi --gen-3d -h -omol2 -O STY.mol2

SMILES is a really great way to describe molecular structures, and it makes monomer structure generation simply a matter of expressing it as a string and using ``obabel`` to generate coordinates.

In-config SMILES (preferred)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can skip the standalone ``obabel`` invocation entirely by giving the
SMILES string directly to ``htpolynet`` inside the config file's
``constituents`` block.  ``htpolynet`` runs ``obabel`` (or RDKit, see below)
itself before parameterization, writes the resulting ``mol2`` into
``lib/molecules/inputs/<NAME>.mol2``, and proceeds as if you had generated it
by hand.  Two flavours of the spec are accepted:

* **obabel index path** (no extra dependencies).  Provide ``smiles`` and a
  ``rename_atoms`` map keyed by 1-based mol2 atom index::

    constituents:
      STY:
        smiles: "C1=CC=CC=C1CC"
        rename_atoms: {7: C1, 8: C2}

  You need to know which obabel-emitted indices to rename, which usually
  means running ``obabel`` once by hand to inspect the atom order.

* **RDKit atom-mapping path** (recommended; requires ``rdkit``).  Encode
  reactive atoms inline with SMILES atom-map labels (``[CH2:1]``) and a
  ``reactive_atoms`` map keyed by those labels::

    constituents:
      STY:
        smiles: "c1ccccc1[CH2:1][CH3:2]"
        reactive_atoms: {1: C1, 2: C2}

  This identifies the reactive atoms by chemical identity rather than by
  index, so the spec is robust to changes in ``obabel``'s output ordering.
  Install with ``pip install 'htpolynet[smiles]'`` or use the container,
  which ships RDKit by default.

  .. warning::

     SMILES bracket atoms (``[...]``) take an **explicit** hydrogen count.
     ``[C:1]`` means *zero implicit H* — the carbon stays at its explicit
     valence.  For an sp³ carbon you almost always want ``[CH:1]`` (one
     implicit H) or ``[CH2:1]`` / ``[CH3:1]`` as appropriate.  Mis-specified
     hydrogen counts typically show up as antechamber typing a saturated
     carbon as ``c2`` instead of ``c3``, propagating into a missing GAFF
     angle parameter in ``tleap``.

If an existing ``lib/molecules/inputs/<NAME>.mol2`` is present, it is left
alone and the SMILES regeneration is skipped — hand-edits survive a re-run.
Delete the file to force regeneration.

Atom-naming
^^^^^^^^^^^

Whether the ``mol2`` came from a sketcher, raw ``obabel``, or in-config
SMILES, ``htpolynet`` expects atoms that it must reference to have **unique
names** in each type of monomer.  It doesn't matter what the names are, but
they must be unique.  Not **all** atoms need to be uniquely named; only the
ones that ``htpolynet`` needs in order to make bonds happen need unique
names.  The in-config SMILES path handles this naming automatically via
``rename_atoms`` / ``reactive_atoms``; for hand-generated mol2 you must
edit the atom-name field yourself.  We provide several examples of atom
naming conventions in the :ref:`tutorials <example_tutorials>`.

