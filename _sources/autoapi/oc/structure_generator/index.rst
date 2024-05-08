:py:mod:`oc.structure_generator`
================================

.. py:module:: oc.structure_generator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   oc.structure_generator.StructureGenerator



Functions
~~~~~~~~~

.. autoapisummary::

   oc.structure_generator.write_surface
   oc.structure_generator.parse_args
   oc.structure_generator.precompute_slabs
   oc.structure_generator.run_placements



Attributes
~~~~~~~~~~

.. autoapisummary::

   oc.structure_generator.args


.. py:class:: StructureGenerator(args, bulk_index, surface_index, adsorbate_index)


   A class that creates adsorbate/bulk/slab objects given specified indices,
   and writes vasp input files and metadata for multiple placements of the adsorbate
   on the slab. You can choose random, heuristic, or both types of placements.

   The output directory structure will have the following nested structure,
   where "files" represents the vasp input files and the metadata.pkl:
       outputdir/
           bulk0/
               surface0/
                   surface/files
                   ads0/
                       heur0/files
                       heur1/files
                       rand0/files
                       ...
                   ads1/
                       ...
               surface1/
                   ...
           bulk1/
               ...

   Precomputed surfaces will be calculated and saved out if they don't
   already exist in the provided directory.

   :param args: Contains all command line args
   :type args: argparse.Namespace
   :param bulk_index: Index of the bulk within the bulk db
   :type bulk_index: int
   :param surface_index: Index of the surface in the list of all possible surfaces
   :type surface_index: int
   :param adsorbate_index: Index of the adsorbate within the adsorbate db
   :type adsorbate_index: int

   .. py:method:: run()

      Create adsorbate/bulk/surface objects, generate adslab placements,
      and write to files.


   .. py:method:: _write_adslabs(adslab_obj, mode_str)

      Write one set of adslabs (called separately for random and heurstic placements)



.. py:function:: write_surface(args, slab, bulk_index, surface_index)

   Writes vasp inputs and metadata for a specified  slab


.. py:function:: parse_args()


.. py:function:: precompute_slabs(bulk_ind)


.. py:function:: run_placements(inputs)


.. py:data:: args

   

