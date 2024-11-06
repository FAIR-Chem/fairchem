data.oc.utils.geometry
======================

.. py:module:: data.oc.utils.geometry


Classes
-------

.. autoapisummary::

   data.oc.utils.geometry.Geometry
   data.oc.utils.geometry.PlaneBoundTriclinicGeometry
   data.oc.utils.geometry.BoxGeometry


Module Contents
---------------

.. py:class:: Geometry

   Bases: :py:obj:`abc.ABC`


   Base class for geometries


   .. py:method:: distance_point_plane(vec: numpy.array, point_plane: numpy.array, point_ext: numpy.array)
      :staticmethod:


      Returns the (shortest) distance between a plane with normal vector
      'vec' through point 'point_plane' and a point 'point_ext'.

      :param vec: normal vector of plane
      :type vec: np.array
      :param point_plane: point on line
      :type point_plane: np.array
      :param point_ext: external point(s)
      :type point_ext: np.array

      :returns: (np.array) Distance between plane and external point(s)



   .. py:method:: vec_and_point_to_plane(vec: numpy.array, point: numpy.array)
      :staticmethod:


      Returns the (unique) plane, given a normal vector 'vec' and a
      point 'point' in the plane. ax + by + cz - d = 0

      :param vec: normal vector of plane
      :type vec: np.array
      :param point: point in plane
      :type point: np.array

      :returns: (np.array) Paramterization of plane



   .. py:method:: cell2planes(cell: ase.cell.Cell, pbc: float)
      :staticmethod:


      Get the parameterization of the sizes of a ase.Atom cell

      cell: ase.cell.Cell
      pbc (float): shift of boundaries to be used with periodic boundary condition

      Return
          (List[np.array]) Parameterization of cell plane sides

      3 planes intersect the origin by ase design.



   .. py:method:: extract_box_properties(center: numpy.array, length: numpy.array, lo_corner: numpy.array, hi_corner: numpy.array)
      :staticmethod:


      Given two of the properties 'center', 'length', 'lo_corner',
      'hi_corner', return all the properties. The properties that
      are not given are expected to be 'None'.



   .. py:method:: packmol_structure(filename: str, number: int, side: str)
      :abstractmethod:


      How to write packmol input file. To be defined by inherited class.



.. py:class:: PlaneBoundTriclinicGeometry(cell: ase.cell.Cell, pbc: float = 0.0)

   Bases: :py:obj:`Geometry`


   Triclinic crystal geometry based on ase.Atom cell


   .. py:attribute:: planes


   .. py:attribute:: cell


   .. py:attribute:: ll_corner
      :value: [0, 0, 0]



   .. py:attribute:: ur_corner


   .. py:method:: packmol_structure(filename: str, number: int, side: str)

      Make file structure to be used in packmol input script

      :param filename: output filename to save structure
      :type filename: str
      :param number: number of solvent molecules
      :type number: int
      :param side: pack solvent inside/outside of geometry
      :type side: str

      :returns: String with information about the structure



.. py:class:: BoxGeometry(center=None, length=None, lo_corner=None, hi_corner=None, **kwargs)

   Bases: :py:obj:`Geometry`


   Box geometry for orthorhombic cells.


   .. py:attribute:: params


   .. py:attribute:: length


   .. py:method:: __repr__()


   .. py:method:: packmol_structure(filename: str, number: int, side: str)

      Make file structure to be used in packmol input script

      :param filename: output filename to save structure
      :type filename: str
      :param number: number of solvent molecules
      :type number: int
      :param side: pack solvent inside/outside of geometry
      :type side: str

      :returns: String with information about the structure



