"""
FeatureIDE FM parser and normalized in-memory model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field


FM_NODE_TAGS = {"and", "alt", "or", "feature"}


def _local_tag(tag: str) -> str:
    """Return XML tag name without namespace prefix."""
    return tag.split("}", 1)[1] if "}" in tag else tag


class FMAttribute(BaseModel):
    """Feature attribute metadata."""
    name: str
    type: str = "string"
    unit: str = ""


class FMConstraint(BaseModel):
    """A node in a propositional constraint formula."""
    operator: str
    operands: List["FMConstraint"] = Field(default_factory=list)
    variable: Optional[str] = None


FMConstraint.model_rebuild()


class FMNode(BaseModel):
    """A node in the parsed FM tree."""
    id: str
    name: str
    path: str
    node_type: str
    group_type: str
    mandatory: bool = False
    abstract: bool = False
    parent_id: Optional[str] = None
    depth: int = 0
    attributes: List[FMAttribute] = Field(default_factory=list)
    children: List["FMNode"] = Field(default_factory=list)

    @property
    def is_string_feature(self) -> bool:
        return any(attr.type.lower() == "string" for attr in self.attributes)


FMNode.model_rebuild()


class FMDocument(BaseModel):
    """Canonical FM document used by backend and frontend."""
    source_path: str
    root: FMNode
    artefact_type: str
    index: Dict[str, Dict[str, Any]]
    constraints: List[FMConstraint] = Field(default_factory=list)


class FM:
    """Loads and exposes a FeatureIDE `fm.xml` as a canonical tree."""
    def __init__(self, xml_path: str):
        self._xml_path = Path(xml_path)
        self._document = self._parse()

    @property
    def constraints(self) -> List[FMConstraint]:
        return self._document.constraints

    @property
    def root(self) -> FMNode:
        return self._document.root

    @property
    def artefact_type(self) -> str:
        return self._document.artefact_type

    @property
    def source_path(self) -> str:
        return self._document.source_path

    def to_dict(self) -> Dict[str, Any]:
        return self._document.model_dump(mode="json")

    def iter_nodes(self) -> Iterable[FMNode]:
        stack: List[FMNode] = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def _parse(self) -> FMDocument:
        if not self._xml_path.exists():
            raise FileNotFoundError(f"FM file not found: {self._xml_path}")

        tree = ET.parse(self._xml_path)
        xml_root = tree.getroot()

        struct_elem = None
        for child in xml_root:
            if _local_tag(child.tag) == "struct":
                struct_elem = child
                break

        if struct_elem is None:
            raise ValueError("Invalid FeatureIDE XML: missing <struct> section")

        fm_root_elem = None
        for child in struct_elem:
            if _local_tag(child.tag) in FM_NODE_TAGS:
                fm_root_elem = child
                break

        if fm_root_elem is None:
            raise ValueError("Invalid FeatureIDE XML: missing FM root node under <struct>")

        root_node = self._parse_node(
            elem=fm_root_elem,
            parent_path="",
            parent_id=None,
            depth=0,
        )
        artefact_type = self._derive_artefact_type(root_node)
        index = self._build_index(root_node)
        constraints = self._parse_constraints(xml_root)

        return FMDocument(
            source_path=str(self._xml_path),
            root=root_node,
            artefact_type=artefact_type,
            index=index,
            constraints=constraints,
        )

    def _parse_node(
        self,
        elem: ET.Element,
        parent_path: str,
        parent_id: Optional[str],
        depth: int,
    ) -> FMNode:
        tag = _local_tag(elem.tag)
        if tag not in FM_NODE_TAGS:
            raise ValueError(f"Unsupported FM node tag: <{tag}>")

        node_name = elem.attrib.get("name", tag)
        node_path = f"{parent_path}.{node_name}" if parent_path else node_name

        attributes: List[FMAttribute] = []
        children: List[FMNode] = []
        for child in elem:
            child_tag = _local_tag(child.tag)
            if child_tag == "attribute":
                attributes.append(
                    FMAttribute(
                        name=child.attrib.get("name", ""),
                        type=child.attrib.get("type", "string"),
                        unit=child.attrib.get("unit", ""),
                    )
                )
            elif child_tag in FM_NODE_TAGS:
                children.append(
                    self._parse_node(
                        elem=child,
                        parent_path=node_path,
                        parent_id=node_path,
                        depth=depth + 1,
                    )
                )

        return FMNode(
            id=node_path,
            name=node_name,
            path=node_path,
            node_type=tag,
            group_type=tag,
            mandatory=elem.attrib.get("mandatory", "false").lower() == "true",
            abstract=elem.attrib.get("abstract", "false").lower() == "true",
            parent_id=parent_id,
            depth=depth,
            attributes=attributes,
            children=children,
        )

    def _derive_artefact_type(self, root_node: FMNode) -> str:
        """Convention: root is Artefact. First child is the artefact type."""
        if root_node.children:
            return root_node.children[0].name
        return root_node.name

    def _build_index(self, root_node: FMNode) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        stack: List[FMNode] = [root_node]
        while stack:
            node = stack.pop()
            index[node.id] = {
                "id": node.id,
                "name": node.name,
                "path": node.path,
                "node_type": node.node_type,
                "group_type": node.group_type,
                "mandatory": node.mandatory,
                "abstract": node.abstract,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "attributes": [attr.model_dump(mode="json") for attr in node.attributes],
                "children_ids": [child.id for child in node.children],
                "is_string_feature": node.is_string_feature,
            }
            stack.extend(reversed(node.children))

        return index

    def _parse_constraints(self, xml_root: ET.Element) -> List[FMConstraint]:
        """Parse the <constraints> section into propositional formula trees."""
        constraints: List[FMConstraint] = []
        for child in xml_root:
            if _local_tag(child.tag) == "constraints":
                for rule_elem in child:
                    if _local_tag(rule_elem.tag) == "rule":
                        for expr_elem in rule_elem:
                            parsed = self._parse_constraint_expr(expr_elem)
                            if parsed:
                                constraints.append(parsed)
        return constraints

    def _parse_constraint_expr(self, elem: ET.Element) -> Optional[FMConstraint]:
        """Recursively parse a constraint expression element."""
        tag = _local_tag(elem.tag)
        if tag == "graphics":
            return None

        if tag == "var":
            return FMConstraint(operator="var", variable=(elem.text or "").strip())

        if tag in {"imp", "or", "and", "eq"}:
            operands = [
                self._parse_constraint_expr(child)
                for child in elem
                if _local_tag(child.tag) != "graphics"
            ]
            return FMConstraint(
                operator=tag,
                operands=[op for op in operands if op is not None],
            )

        if tag == "not":
            for child in elem:
                if _local_tag(child.tag) != "graphics":
                    inner = self._parse_constraint_expr(child)
                    if inner:
                        return FMConstraint(operator="not", operands=[inner])
            return None

        raise ValueError(f"Unsupported constraint operator: <{tag}>")
