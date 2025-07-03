"""Define classes to represent PDDL planning problem and domain files."""

from __future__ import annotations

from typing import Protocol


class PDDLable(Protocol):
    """A protocol for classes supporting conversion to and from PDDL strings."""

    @classmethod
    def from_pddl(cls, pddl: str) -> PDDLable:
        """Construct a PDDLable instance from a string of PDDL.

        :param pddl: PDDL string representation of a PDDLable instance
        :return: Constructed PDDLable instance
        """
        ...

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the PDDLable instance."""
        ...
