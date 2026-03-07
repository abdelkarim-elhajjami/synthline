"""
Tests for FeatureIDE FM parsing semantics in FM.
"""
from pathlib import Path

import pytest

from synthline.core.fm_parser import FM, FMNode

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FM_FIXTURE = str(FIXTURE_DIR / "fm.xml")


def _find_node(fm: FM, node_id: str) -> FMNode | None:
    return next((n for n in fm.iter_nodes() if n.id == node_id), None)


class TestParserCompleteness:
    """Parser handles FeatureIDE node types, attributes, and constraints."""

    def test_all_node_types_parsed(self):
        fm = FM(FM_FIXTURE)
        types = {node.node_type for node in fm.iter_nodes()}
        assert types == {"and", "alt", "or", "feature"}

    def test_mandatory_attribute_parsed(self):
        fm = FM(FM_FIXTURE)
        lang = _find_node(fm,"Artefact.Requirement.Description.Language")
        assert lang is not None
        assert lang.mandatory is True

    def test_abstract_attribute_parsed(self):
        fm = FM(FM_FIXTURE)
        ctx = _find_node(fm,"Artefact.Requirement.Context")
        assert ctx is not None
        assert ctx.abstract is True

    def test_and_group_with_feature_children(self):
        fm = FM(FM_FIXTURE)
        ctx = _find_node(fm,"Artefact.Requirement.Context")
        assert ctx is not None
        assert ctx.node_type == "and"
        assert len(ctx.children) == 8
        assert all(c.node_type == "feature" for c in ctx.children)

    def test_and_wrapper_structure(self):
        """Functional is an and-wrapper containing an alt subtype."""
        fm = FM(FM_FIXTURE)
        func = _find_node(fm,"Artefact.Requirement.RequirementType.Functional")
        assert func is not None
        assert func.node_type == "and"
        assert func.abstract is False
        assert len(func.children) == 1
        assert func.children[0].name == "FunctionalSubtype"
        assert func.children[0].node_type == "alt"

    def test_constraints_parsed(self):
        fm = FM(FM_FIXTURE)
        assert len(fm.constraints) == 1
        c = fm.constraints[0]
        assert c.operator == "imp"
        assert c.operands[0].variable == "ControlledNL"
        assert c.operands[1].variable == "WellFormed"

    def test_string_attribute_parsed(self):
        fm = FM(FM_FIXTURE)
        domain = _find_node(fm,"Artefact.Requirement.Domain")
        assert domain is not None
        assert domain.is_string_feature is True
        assert domain.attributes[0].name == "Value"
        assert domain.attributes[0].type == "string"


class TestUnsupportedConstraintOperator:
    """Parser should fail fast for unsupported operators."""

    def test_unsupported_operator_raises(self):
        xml = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<extendedFeatureModel>
  <struct>
    <and mandatory="true" name="Root">
      <feature name="A"/>
    </and>
  </struct>
  <constraints>
    <rule><xor><var>A</var></xor></rule>
  </constraints>
</extendedFeatureModel>"""
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".xml")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(xml)
            with pytest.raises(ValueError, match="Unsupported constraint operator"):
                FM(path)
        finally:
            os.unlink(path)
