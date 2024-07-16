from __future__ import annotations

import numpy as np

# Code adapted from https://github.com/henriasv/molecular-builder/tree/master


class Geometry:
    """Base class for geometries.

    :param periodic_boundary_condition: self-explanatory
    :type periodic_boundary_condition: array_like
    :param minimum_image_convention: use the minimum image convention for
                                     bookkeeping how the particles interact
    :type minimum_image_convention: bool
    """

    def __init__(
        self,
        periodic_boundary_condition=(False, False, False),
        minimum_image_convention=True,
    ):
        self.minimum_image_convention = minimum_image_convention
        self.periodic_boundary_condition = periodic_boundary_condition

    def __call__(self, atoms):
        """The empty geometry. False because we define no particle to be
        in the dummy geometry.

        :param atoms: atoms object from ase.Atom that is being modified
        :type atoms: ase.Atom obj
        :returns: ndarray of bools telling which atoms to remove
        :rtype: ndarray of bool
        """
        return np.zeros(len(atoms), dtype=np.bool)

    @staticmethod
    def distance_point_line(vec, point_line, point_ext):
        """Returns the (shortest) distance between a line parallel to
        a normal vector 'vec' through point 'point_line' and an external
        point 'point_ext'.

        :param vec: unit vector parallel to line
        :type vec: ndarray
        :param point_line: point on line
        :type point_line: ndarray
        :param point_ext: external points
        :type point_ext: ndarray
        :return: distance between line and external point(s)
        :rtype: ndarray
        """
        return np.linalg.norm(np.cross(vec, point_ext - point_line), axis=1)

    @staticmethod
    def distance_point_plane(vec, point_plane, point_ext):
        """Returns the (shortest) distance between a plane with normal vector
        'vec' through point 'point_plane' and a point 'point_ext'.

        :param vec: normal vector of plane
        :type vec: ndarray
        :param point_plane: point on line
        :type point_plane: ndarray
        :param point_ext: external point(s)
        :type point_ext: ndarray
        :return: distance between plane and external point(s)
        :rtype: ndarray
        """
        vec = np.atleast_2d(vec)  # Ensure n is 2d
        return np.abs(np.einsum("ik,jk->ij", point_ext - point_plane, vec))

    @staticmethod
    def vec_and_point_to_plane(vec, point):
        """Returns the (unique) plane, given a normal vector 'vec' and a
        point 'point' in the plane. ax + by + cz - d = 0

        :param vec: normal vector of plane
        :type vec: ndarray
        :param point: point in plane
        :type point: ndarray
        :returns: parameterization of plane
        :rtype: ndarray
        """
        return np.array((*vec, np.dot(vec, point)))

    @staticmethod
    def cell2planes(cell, pbc):
        """Get the parameterization of the sizes of a ase.Atom cell

        :param cell: ase.Atom cell
        :type cell: obj
        :param pbc: shift of boundaries to be used with periodic boundary condition
        :type pbc: float
        :returns: parameterization of cell plane sides
        :rtype: list of ndarray

        3 planes intersect the origin by ase design.
        """
        a = cell[0]
        b = cell[1]
        c = cell[2]

        n1 = np.cross(a, b)
        n2 = np.cross(c, a)
        n3 = np.cross(b, c)

        origin = np.array([0, 0, 0]) + pbc / 2
        top = (a + b + c) - pbc / 2

        plane1 = Geometry.vec_and_point_to_plane(n1, origin)
        plane2 = Geometry.vec_and_point_to_plane(n2, origin)
        plane3 = Geometry.vec_and_point_to_plane(n3, origin)
        plane4 = Geometry.vec_and_point_to_plane(-n1, top)
        plane5 = Geometry.vec_and_point_to_plane(-n2, top)
        plane6 = Geometry.vec_and_point_to_plane(-n3, top)

        return [plane1, plane2, plane3, plane4, plane5, plane6]

    @staticmethod
    def extract_box_properties(center, length, lo_corner, hi_corner):
        """Given two of the properties 'center', 'length', 'lo_corner',
        'hi_corner', return all the properties. The properties that
        are not given are expected to be 'None'.
        """
        # exactly two arguments have to be non-none
        if sum(x is None for x in [center, length, lo_corner, hi_corner]) != 2:
            raise ValueError("Exactly two arguments have to be given")

        # declare arrays to allow mathematical operations
        center, length = np.asarray(center), np.asarray(length)
        lo_corner, hi_corner = np.asarray(lo_corner), np.asarray(hi_corner)
        relations = [
            [
                "lo_corner",
                "hi_corner - length",
                "center - length / 2",
                "2 * center - hi_corner",
            ],
            [
                "hi_corner",
                "lo_corner + length",
                "center + length / 2",
                "2 * center - lo_corner",
            ],
            [
                "length / 2",
                "(hi_corner - lo_corner) / 2",
                "hi_corner - center",
                "center - lo_corner",
            ],
            [
                "center",
                "(hi_corner + lo_corner) / 2",
                "hi_corner - length / 2",
                "lo_corner + length / 2",
            ],
        ]

        # compute all relations
        relation_list = []
        for relation in relations:
            for i in relation:
                try:
                    relation_list.append(eval(i))
                except TypeError:
                    continue

        # keep the non-None relations
        for i, relation in enumerate(relation_list):
            if None in relation:
                del relation_list[i]
        return relation_list

    def packmol_structure(self, filename, number, side):
        """Make structure to be used in PACKMOL input script

        :param number: number of solvent molecules
        :type number: int
        :param side: pack solvent inside/outside of geometry
        :type side: str
        :returns: string with information about the structure
        :rtype: str
        """
        structure = ""
        structure += f"structure {filename}\n"
        structure += f"  number {number}\n"
        structure += f"  {side} {self.__repr__()} "
        for param in self.params:
            structure += f"{param} "
        structure += "\nend structure\n"
        return structure


class PlaneBoundTriclinicGeometry(Geometry):
    """Triclinic crystal geometry based on ase.Atom cell

    :param cell: ase.Atom cell
    :type cell: obj
    :param pbc: shift of boundaries to be used with periodic boundary condition
    :type pbc: float
    """

    def __init__(self, cell, pbc=0.0):
        self.planes = self.cell2planes(cell, pbc)
        self.cell = cell
        self.ll_corner = [0, 0, 0]
        a = cell[0, :]
        b = cell[1, :]
        c = cell[2, :]
        self.ur_corner = a + b + c

    def packmol_structure(self, filename, number, side):
        """Make structure to be used in PACKMOL input script"""
        structure = ""

        if side == "inside":
            side = "over"
        elif side == "outside":
            side = "below"
        structure += f"structure {filename}\n"
        structure += f"  number {number}\n"
        for plane in self.planes:
            structure += f"  {side} plane "
            for param in plane:
                structure += f"{param} "
            structure += "\n"
        structure += "end structure\n"
        return structure

    def __call__(self, position):
        raise NotImplementedError


class BoxGeometry(Geometry):
    """Box geometry.

    :param center: geometric center of box
    :type center: array_like
    :param length: length of box in all directions
    :type length: array_like
    :param lo_corner: lower corner
    :type lo_corner: array_like
    :param hi_corner: higher corner
    :type hi_corner: array_like
    """

    def __init__(
        self, center=None, length=None, lo_corner=None, hi_corner=None, **kwargs
    ):
        super().__init__(**kwargs)
        props = self.extract_box_properties(center, length, lo_corner, hi_corner)
        self.ll_corner, self.ur_corner, self.length_half, self.center = props
        self.params = list(self.ll_corner) + list(self.ur_corner)
        self.length = self.length_half * 2

    def __repr__(self):
        return "box"

    def __call__(self, atoms):
        positions = atoms.get_positions()
        dist = self.distance_point_plane(np.eye(3), self.center, positions)
        return np.all((np.abs(dist) <= self.length_half), axis=1)

    def volume(self):
        return np.prod(self.length)
