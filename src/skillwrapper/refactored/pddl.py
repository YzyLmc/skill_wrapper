"""Define classes to represent PDDL planning problem and domain files."""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar


class PDDLable(Protocol):
    """A protocol for classes that support conversion to PDDL strings."""

    def as_pddl(self) -> str:
        """Return a PDDL string representation of this class."""
        ...

    @property
    def pddl_requirements(self) -> set[str]:
        """Return the set of PDDL requirements (e.g., :strips) added by this class."""
        ...


PredicateT = TypeVar("PredicateT", bound=PDDLable)  # Accepts any type that can convert to PDDL
OperatorT = TypeVar("OperatorT", bound=PDDLable)
ObjectT = TypeVar("ObjectT", bound=PDDLable)
FactT = TypeVar("FactT", bound=PDDLable)
FormulaT = TypeVar("FormulaT", bound=PDDLable)


@dataclass(frozen=True)
class PDDLDomain(Generic[PredicateT, OperatorT]):
    """A PDDL domain defines the aspects of the planning model shared across situations."""

    name: str  # Name of the domain
    object_types: set[str]  # Set of possible object types
    predicates: set[PredicateT]
    operators: set[OperatorT]

    def find_requirements(self) -> set[str]:
        """Find the set of all PDDL requirements of the domain."""
        reqs = {":typing"} if self.object_types else set()
        for predicate in self.predicates:
            reqs.update(predicate.pddl_requirements)
        for operator in self.operators:
            reqs.update(operator.pddl_requirements)
        return reqs

    def as_pddl(self) -> str:
        """Convert the PDDL domain into its PDDL string representation."""
        pddl_reqs = self.find_requirements()
        types_str = f" {' '.join(t for t in self.object_types)}" if self.object_types else ""
        types_line = f"\t(:types{types_str})\n\n" if types_str else ""
        predicates_str = "\n\t\t".join(p.as_pddl() for p in self.predicates)
        operators_block = "\n\n\t".join(o.as_pddl() for o in self.operators)

        return (
            "(define\n"
            f"\t(domain {self.name})\n"
            f"\t(:requirements{' '.join(req for req in pddl_reqs)})\n"
            f"{types_line}"
            f"\t(:predicates\n\t\t{predicates_str}\n\t)\n\n"
            f"{operators_block}\n)"
        )


@dataclass(frozen=True)
class PDDLProblem(Generic[ObjectT, FactT, FormulaT]):
    """A PDDL problem defines the concrete details of a particular planning problem."""

    name: str  # Name of the problem
    domain: PDDLDomain  # PDDL domain corresponding to the problem
    objects: set[ObjectT]  # Concrete objects in the problem
    initial_state: set[FactT]  # Set of facts (i.e., predicate instances) true in the initial state
    goal_condition: FormulaT  # Logical expression defining goal states
