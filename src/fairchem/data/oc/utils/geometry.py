from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase.cell import Cell

# Code adapted from https://github.com/henriasv/molecular-builder/tree/master


class Geometry(ABC):
    """
    Base class for geometries
    """

    @staticmethod
    def distance_point_plane(vec: np.array, point_plane: np.array, point_ext: np.array):
        """
        Returns the (shortest) distance between a plane with normal vector
        'vec' through point 'point_plane' and a point 'point_ext'.

        Args:
            vec (np.array): normal vector of plane
            point_plane (np.array): point on line
            point_ext (np.array): external point(s)

        Returns:
            (np.array) Distance between plane and external point(s)
        """
        vec = np.atleast_2d(vec)  # Ensure n is 2d
        return np.abs(np.einsum("ik,jk->ij", point_ext - point_plane, vec))

    @staticmethod
    def vec_and_point_to_plane(vec: np.array, point: np.array):
        """
        Returns the (unique) plane, given a normal vector 'vec' and a
        point 'point' in the plane. ax + by + cz - d = 0

        Args:
            vec (np.array): normal vector of plane
            point (np.array): point in plane

        Returns:
            (np.array) Paramterization of plane
        """
        return np.array((*vec, np.dot(vec, point)))

    @staticmethod
    def cell2planes(cell: Cell, pbc: float):
        """
        Get the parameterization of the sizes of a ase.Atom cell

        cell: ase.cell.Cell
        pbc (float): shift of boundaries to be used with periodic boundary condition

        Return
            (List[np.array]) Parameterization of cell plane sides

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
    def extract_box_properties(
        center: np.array, length: np.array, lo_corner: np.array, hi_corner: np.array
    ):
        """
        Given two of the properties 'center', 'length', 'lo_corner',
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

    @abstractmethod
    def packmol_structure(self, filename: str, number: int, side: str):
        """
        How to write packmol input file. To be defined by inherited class.
        """


class PlaneBoundTriclinicGeometry(Geometry):
    """
    Triclinic crystal geometry based on ase.Atom cell
    """

    def __init__(self, cell: Cell, pbc: float = 0.0):
        """
        cell (ase.cell.Cell)
        pbc (float): shift of boundaries to be used with periodic boundary condition
        """
        self.planes = self.cell2planes(cell, pbc)
        self.cell = cell
        self.ll_corner = [0, 0, 0]
        a = cell[0, :]
        b = cell[1, :]
        c = cell[2, :]
        self.ur_corner = a + b + c

    def packmol_structure(self, filename: str, number: int, side: str):
        """
        Make file structure to be used in packmol input script

        Args:
            filename (str): output filename to save structure
            number (int): number of solvent molecules
            side (str): pack solvent inside/outside of geometry

        Returns:
            String with information about the structure
        """

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


class BoxGeometry(Geometry):
    """
    Box geometry for orthorhombic cells.
    """

    def __init__(
        self, center=None, length=None, lo_corner=None, hi_corner=None, **kwargs
    ):
        """
        Args:
            center (np.array): geometric center of box
            length (np.array): length of box in all directions
            lo_corner (np.array): lower corner
            hi_corner (np.array): higher corner
        """
        props = self.extract_box_properties(center, length, lo_corner, hi_corner)
        self.ll_corner, self.ur_corner, self.length_half, self.center = props
        self.params = list(self.ll_corner) + list(self.ur_corner)
        self.length = self.length_half * 2

    def __repr__(self):
        return "box"

    def packmol_structure(self, filename: str, number: int, side: str):
        """
        Make file structure to be used in packmol input script

        Args:
            filename (str): output filename to save structure
            number (int): number of solvent molecules
            side (str): pack solvent inside/outside of geometry

        Returns:
            String with information about the structure
        """
        structure = ""
        structure += f"structure {filename}\n"
        structure += f"  number {number}\n"
        structure += f"  {side} {self.__repr__()} "
        for param in self.params:
            structure += f"{param} "
        structure += "\nend structure\n"
        return structure
