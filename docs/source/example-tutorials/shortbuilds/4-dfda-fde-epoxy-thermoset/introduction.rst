.. _dfe_introduction:

Introduction
------------

Set up a clean working directory and pull the example YAML:

.. code-block:: console

   $ mkdir my_dfa_fde
   $ cd my_dfa_fde
   $ htpolynet fetch-example 4
   Fetched 4-dfda-fde-epoxy-thermoset.yaml  (run with: htpolynet run 4-dfda-fde-epoxy-thermoset.yaml)
   $ ls
   4-dfda-fde-epoxy-thermoset.yaml

As with the earlier examples this is a single self-contained YAML.
Both FDE and DFA are generated from SMILES with atom-mapping at the
start of the run; no shell-script prep needed.

The cure mechanism is identical to :ref:`example 3 <pde_tutorial>`:

* The two primary amines of DFA each donate an H to an opened
  oxirane on FDE, forming a secondary amine and a pendant hydroxyl.
* Each secondary amine can react once more (slowly) to form a
  tertiary amine.
* Any FDE oxirane that remains unreacted gets its three-membered
  C–O–C ring re-formed by a cap reaction.

What is new is the chemistry of the *backbone*: instead of the
bisphenol-A diphenyl scaffold (DGEBA) or the bicyclohexyl-methane
scaffold (PACM), both species here are anchored on **furan** rings.
FDE has one furan ring attached to a tertiary amine bearing two
opened-oxirane arms; DFA has two furan rings joined by a central
methylene, with a furfurylmethylamine on each.

The :ref:`monomer page <dfe_monomers>` walks through the SMILES.
