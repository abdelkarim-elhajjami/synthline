"""
Tests for FeatureIDE FM semantics in FMResolver.
"""
from pathlib import Path
from typing import List, Optional

import pytest

from synthline.core.fm_parser import FM
from synthline.core.fm_resolver import FMResolver

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FM_FIXTURE = str(FIXTURE_DIR / "fm.xml")


# ─── Helpers ───────────────────────────────────────────────────────────────
def _resolver(fm_path: str = FM_FIXTURE) -> FMResolver:
    return FMResolver(FM(fm_path))


def _resolve(fm_configuration: dict, fm_path: str = FM_FIXTURE) -> list:
    return _resolver(fm_path).resolve(fm_configuration)


def _base_config(**overrides) -> dict:
    cfg = {
        "selected_options": {
            "Artefact.Requirement.Description.DescriptionType": ["ProseNL"],
        },
        "string_values": {
            "Artefact.Requirement.Description.Language": ["EN"],
        },
        "selected_features": [],
        "or_group_mode": {},
    }

    selected_options = overrides.pop("selected_options", None)
    if selected_options:
        cfg["selected_options"].update(selected_options)

    string_values = overrides.pop("string_values", None)
    if string_values:
        cfg["string_values"].update(string_values)

    or_group_mode = overrides.pop("or_group_mode", None)
    if or_group_mode:
        cfg["or_group_mode"].update(or_group_mode)

    cfg.update(overrides)
    return cfg


def _constraint_config(
    description_type: str,
    quality_profile: str,
    ears: str = "Ubiquitous",
    defect_children: Optional[List[str]] = None,
) -> dict:
    """Build common config used by constraint-focused tests."""
    desc_path = "Artefact.Requirement.Description.DescriptionType"
    qp_path = "Artefact.Requirement.QualityProfile"
    ears_path = "Artefact.Requirement.Description.DescriptionType.ControlledNL.EARS"

    selected_options = {
        desc_path: [description_type],
        qp_path: [quality_profile],
    }
    if description_type == "ControlledNL":
        selected_options[ears_path] = [ears]
    if quality_profile == "DefectInjection" and defect_children:
        selected_options[f"{qp_path}.{quality_profile}"] = defect_children

    return _base_config(selected_options=selected_options)


# ─── ALT group semantics — each selected child produces a separate variant ───
class TestAltGroupSemantics:
    """Alt groups produce one atomic config per selected child (split)."""

    def test_single_selection(self):
        """Selecting one child in an alt group → one variant."""
        cfg = _base_config(selected_options={
            "Artefact.Requirement.RequirementType": [
                "Artefact.Requirement.RequirementType.Functional",
            ],
        })
        configs = _resolve(cfg)
        rt_values = [c.get("Artefact.Requirement.RequirementType") for c in configs]
        assert rt_values.count("Functional") == 1

    def test_multiple_selections_split_into_variants(self):
        """Selecting multiple children → one separate variant per child."""
        cfg = _base_config(selected_options={
            "Artefact.Requirement.RequirementType": [
                "Artefact.Requirement.RequirementType.Functional",
                "Artefact.Requirement.RequirementType.Quality",
            ],
        })
        configs = _resolve(cfg)
        rt_values = {c.get("Artefact.Requirement.RequirementType") for c in configs}
        assert "Functional" in rt_values
        assert "Quality" in rt_values

    def test_nested_alt_multiplies_variants(self):
        """Alt inside alt → cross-product of variants."""
        cfg = _base_config(selected_options={
            "Artefact.Requirement.RequirementType": [
                "Artefact.Requirement.RequirementType.Functional",
            ],
            "Artefact.Requirement.AbstractionLevel": [
                "Artefact.Requirement.AbstractionLevel.HighLevel",
                "Artefact.Requirement.AbstractionLevel.DetailedLevel",
            ],
        })
        configs = _resolve(cfg)
        # Functional × (HighLevel, DetailedLevel) = 2 variants
        assert len(configs) == 2


# ─── OR group semantics — split vs combine modes ────────────────────────────
class TestOrGroupSemantics:
    """Or groups support split (one per child) and combine (all together)."""

    def test_or_split_mode(self):
        """Or split → one variant per selected child."""
        cfg = _base_config(
            selected_options={
                "Artefact.Requirement.QualityProfile": [
                    "Artefact.Requirement.QualityProfile.DefectInjection",
                ],
                "Artefact.Requirement.QualityProfile.DefectInjection": [
                    "Artefact.Requirement.QualityProfile.DefectInjection.Ambiguous",
                    "Artefact.Requirement.QualityProfile.DefectInjection.Incomplete",
                ],
            },
            or_group_mode={
                "Artefact.Requirement.QualityProfile.DefectInjection": "split",
            },
        )
        configs = _resolve(cfg)
        di_values = [
            c.get("Artefact.Requirement.QualityProfile.DefectInjection")
            for c in configs
        ]
        # Each defect child is a separate string variant
        assert "Ambiguous" in di_values
        assert "Incomplete" in di_values
        assert all(isinstance(v, str) for v in di_values if v is not None)

    def test_or_combine_mode(self):
        """Or combine → all selected children in a single list value."""
        cfg = _base_config(
            selected_options={
                "Artefact.Requirement.QualityProfile": [
                    "Artefact.Requirement.QualityProfile.DefectInjection",
                ],
                "Artefact.Requirement.QualityProfile.DefectInjection": [
                    "Artefact.Requirement.QualityProfile.DefectInjection.Ambiguous",
                    "Artefact.Requirement.QualityProfile.DefectInjection.Incomplete",
                ],
            },
            or_group_mode={
                "Artefact.Requirement.QualityProfile.DefectInjection": "combine",
            },
        )
        configs = _resolve(cfg)
        # Should be a single config with a list value
        di_values = [
            c.get("Artefact.Requirement.QualityProfile.DefectInjection")
            for c in configs
            if c.get("Artefact.Requirement.QualityProfile.DefectInjection") is not None
        ]
        assert len(di_values) >= 1
        assert isinstance(di_values[0], list)
        assert "Ambiguous" in di_values[0]
        assert "Incomplete" in di_values[0]

    def test_selected_or_group_without_children_is_invalid(self):
        """Selecting DefectInjection without picking any child is invalid."""
        cfg = _base_config(
            selected_options={
                "Artefact.Requirement.QualityProfile": [
                    "Artefact.Requirement.QualityProfile.DefectInjection",
                ],
            },
        )
        with pytest.raises(ValueError, match="No valid FM configurations"):
            _resolve(cfg)


# ─── AND group semantics — combine via selected_options ───────────────────────
class TestAndGroupSemantics:
    """And groups with selected_options use combine semantics."""

    def test_and_group_single_selection(self):
        """Selecting one child in an and group → value is a list with one name."""
        ctx_path = "Artefact.Requirement.Context"
        cfg = _base_config(selected_options={
            ctx_path: [f"{ctx_path}.Usage"],
        })
        configs = _resolve(cfg, FM_FIXTURE)
        ctx_values = [c.get(ctx_path) for c in configs if c.get(ctx_path) is not None]
        assert len(ctx_values) >= 1
        assert ctx_values[0] == ["Usage"]

    def test_and_group_multiple_selections_combine(self):
        """Selecting multiple children → all appear together in every variant."""
        ctx_path = "Artefact.Requirement.Context"
        cfg = _base_config(selected_options={
            ctx_path: [
                f"{ctx_path}.Usage",
                f"{ctx_path}.Business",
                f"{ctx_path}.Technical",
            ],
        })
        configs = _resolve(cfg, FM_FIXTURE)
        ctx_values = [c.get(ctx_path) for c in configs if c.get(ctx_path) is not None]
        assert len(ctx_values) >= 1
        # All three appear together (combine, not split)
        assert set(ctx_values[0]) == {"Usage", "Business", "Technical"}

    def test_and_group_does_not_split(self):
        """And group with 3 selections → does NOT produce 3 separate variants."""
        ctx_path = "Artefact.Requirement.Context"
        cfg = _base_config(selected_options={
            ctx_path: [
                f"{ctx_path}.Usage",
                f"{ctx_path}.Business",
            ],
        })
        configs = _resolve(cfg, FM_FIXTURE)
        # Only 1 variant (combined), not 2 (split)
        ctx_variants = [c for c in configs if c.get(ctx_path)]
        assert len(ctx_variants) == 1

    def test_and_group_empty_selection_is_valid_with_minimum_mandatory_defaults(self):
        """Baseline config with mandatory defaults remains valid."""
        cfg = _base_config()
        configs = _resolve(cfg, FM_FIXTURE)
        assert len(configs) >= 1

    def test_and_group_with_other_selections(self):
        """And group selections combine correctly with alt group selections."""
        ctx_path = "Artefact.Requirement.Context"
        abs_path = "Artefact.Requirement.AbstractionLevel"
        cfg = _base_config(selected_options={
            ctx_path: [f"{ctx_path}.Business"],
            abs_path: [
                f"{abs_path}.HighLevel",
                f"{abs_path}.DetailedLevel",
            ],
        })
        configs = _resolve(cfg, FM_FIXTURE)
        # Business context × (HighLevel, DetailedLevel) = 2 variants
        configs_with_ctx = [c for c in configs if c.get(ctx_path)]
        assert len(configs_with_ctx) == 2
        for c in configs_with_ctx:
            assert c[ctx_path] == ["Business"]


# ─── AND group with mandatory children ───────────────────────────────────────
class TestAndGroupMandatoryChildren:
    """And groups must include mandatory children alongside explicit selections."""

    def test_mandatory_children_included_in_and_resolution(self):
        """Build a minimal in-memory FM with an and group that has a
        mandatory child + optional children selected via selected_options.
        The mandatory child must appear in the resolved variant."""
        import os
        import tempfile

        # Build a tiny FM: Root(and) → Group(and) → [A(feature, mandatory), B(feature), C(feature)]
        xml = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<extendedFeatureModel>
  <struct>
    <and mandatory="true" name="Root">
      <and name="Group">
        <feature mandatory="true" name="AlwaysOn"/>
        <feature name="OptionalA"/>
        <feature name="OptionalB"/>
      </and>
    </and>
  </struct>
  <constraints/>
</extendedFeatureModel>"""

        fd, path = tempfile.mkstemp(suffix=".xml")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(xml)
            fm = FM(path)
            resolver = FMResolver(fm)

            cfg = {
                "selected_options": {
                    "Root.Group": ["Root.Group.OptionalA"],
                },
                "string_values": {},
                "selected_features": [],
                "or_group_mode": {},
            }
            configs = resolver.resolve(cfg)
            group_values = [c.get("Root.Group") for c in configs if c.get("Root.Group")]
            assert len(group_values) >= 1
            # Mandatory child must be included alongside selected optional
            assert "AlwaysOn" in group_values[0]
            assert "OptionalA" in group_values[0]
            # OptionalB was not selected → should NOT appear
            assert "OptionalB" not in group_values[0]
        finally:
            os.unlink(path)


# ─── AND wrapper nodes inside ALT groups ─────────────────────────────────────
class TestAndWrapperInAlt:
    """And nodes used as wrappers inside alt groups (e.g., Functional(and) → FunctionalSubtype(alt))."""

    def test_and_wrapper_resolves_inner_alt(self):
        """Selecting an and-wrapper child in an alt group → resolves its inner alt group."""
        rt_path = "Artefact.Requirement.RequirementType"
        func_path = f"{rt_path}.Functional"
        subtype_path = f"{func_path}.FunctionalSubtype"

        cfg = _base_config(selected_options={
            rt_path: [func_path],
            subtype_path: [f"{subtype_path}.DataPerspective"],
        })
        configs = _resolve(cfg, FM_FIXTURE)
        rt_values = [c.get(rt_path) for c in configs if c.get(rt_path)]
        assert "Functional" in rt_values
        sub_values = [c.get(subtype_path) for c in configs if c.get(subtype_path)]
        assert "DataPerspective" in sub_values

    def test_and_wrapper_multiple_subtypes_split(self):
        """Inner alt inside and-wrapper splits into separate variants."""
        rt_path = "Artefact.Requirement.RequirementType"
        func_path = f"{rt_path}.Functional"
        subtype_path = f"{func_path}.FunctionalSubtype"

        cfg = _base_config(selected_options={
            rt_path: [func_path],
            subtype_path: [
                f"{subtype_path}.DataPerspective",
                f"{subtype_path}.BehavioralPerspective",
            ],
        })
        configs = _resolve(cfg, FM_FIXTURE)
        sub_values = [c.get(subtype_path) for c in configs if c.get(subtype_path)]
        assert "DataPerspective" in sub_values
        assert "BehavioralPerspective" in sub_values
        # Alt inside and → split = 2 variants
        assert len(sub_values) == 2


# ─── Constraint enforcement — works across all group types ───────────────────
class TestConstraintEnforcementAcrossGroups:
    """Cross-tree constraints are enforced regardless of group type."""

    def test_constraint_valid_with_and_groups(self):
        """ControlledNL → WellFormed holds even with the and-group FM."""
        desc_path = "Artefact.Requirement.Description.DescriptionType"
        qp_path = "Artefact.Requirement.QualityProfile"
        ears_path = f"{desc_path}.ControlledNL.EARS"

        cfg = _base_config(selected_options={
            desc_path: ["ControlledNL"],
            ears_path: ["Ubiquitous"],
            qp_path: ["WellFormed"],
        })
        configs = _resolve(cfg, FM_FIXTURE)
        qp_values = [c.get(qp_path) for c in configs if c.get(qp_path)]
        assert "WellFormed" in qp_values

    def test_constraint_violation_pruned_with_and_groups(self):
        """ControlledNL + DefectInjection violates ControlledNL→WellFormed → pruned."""
        desc_path = "Artefact.Requirement.Description.DescriptionType"
        qp_path = "Artefact.Requirement.QualityProfile"
        ears_path = f"{desc_path}.ControlledNL.EARS"
        defective_path = f"{qp_path}.DefectInjection"

        cfg = _base_config(selected_options={
            desc_path: ["ControlledNL"],
            ears_path: ["Ubiquitous"],
            qp_path: ["DefectInjection"],
            defective_path: ["Ambiguous"],
        })
        cfg["allow_empty_fallback"] = False
        with pytest.raises(ValueError, match="No valid FM configurations"):
            _resolve(cfg, FM_FIXTURE)

    def test_constraint_violation_can_opt_into_empty_fallback(self):
        """When requested, fully pruned variants return explicit empty fallback."""
        desc_path = "Artefact.Requirement.Description.DescriptionType"
        qp_path = "Artefact.Requirement.QualityProfile"
        ears_path = f"{desc_path}.ControlledNL.EARS"
        defective_path = f"{qp_path}.DefectInjection"

        cfg = _base_config(
            selected_options={
                desc_path: ["ControlledNL"],
                ears_path: ["Ubiquitous"],
                qp_path: ["DefectInjection"],
                defective_path: ["Ambiguous"],
            },
        )
        cfg["allow_empty_fallback"] = True

        configs = _resolve(cfg, FM_FIXTURE)
        assert len(configs) == 1
        assert set(configs[0].keys()) == {"__fm_constraints__"}


class TestConstraintEnforcement:
    """Constraint pruning and implication behavior on the original FM fixture."""

    def test_controlled_nl_with_wellformed_is_valid(self):
        configs = _resolve(_constraint_config("ControlledNL", "WellFormed"), FM_FIXTURE)
        quality_values = [c.get("Artefact.Requirement.QualityProfile") for c in configs]
        assert "WellFormed" in quality_values

    def test_controlled_nl_with_defect_injection_is_pruned(self):
        cfg = _constraint_config(
            "ControlledNL",
            "DefectInjection",
            defect_children=["Ambiguous"],
        )
        cfg["allow_empty_fallback"] = False
        with pytest.raises(ValueError, match="No valid FM configurations"):
            _resolve(cfg, FM_FIXTURE)

    def test_prose_nl_with_defect_injection_is_valid(self):
        cfg = _constraint_config(
            "ProseNL",
            "DefectInjection",
            defect_children=["Ambiguous"],
        )
        configs = _resolve(cfg, FM_FIXTURE)
        quality_values = [c.get("Artefact.Requirement.QualityProfile") for c in configs]
        assert "DefectInjection" in quality_values

    def test_partial_pruning_mixed_description_types(self):
        desc_path = "Artefact.Requirement.Description.DescriptionType"
        qp_path = "Artefact.Requirement.QualityProfile"
        ears_path = "Artefact.Requirement.Description.DescriptionType.ControlledNL.EARS"
        di_path = "Artefact.Requirement.QualityProfile.DefectInjection"

        cfg = _base_config(
            selected_options={
                desc_path: ["ProseNL", "ControlledNL"],
                ears_path: ["Ubiquitous"],
                qp_path: ["WellFormed", "DefectInjection"],
                di_path: ["Ambiguous"],
            },
        )
        configs = _resolve(cfg, FM_FIXTURE)
        assert len(configs) >= 1

        for c in configs:
            desc = c.get(desc_path)
            qp = c.get(qp_path)
            if desc == "ControlledNL":
                assert qp != "DefectInjection"

        prose_qps = {c.get(qp_path) for c in configs if c.get(desc_path) == "ProseNL"}
        assert "WellFormed" in prose_qps
        assert "DefectInjection" in prose_qps

        ctrl_wf = [
            c for c in configs
            if c.get(desc_path) == "ControlledNL" and c.get(qp_path) == "WellFormed"
        ]
        assert len(ctrl_wf) >= 1


class TestConstraintOperators:
    """Parser and resolver should support all implemented boolean operators."""

    @staticmethod
    def _run(xml: str, cfg: dict):
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".xml")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(xml)
            return FMResolver(FM(path)).resolve(cfg)
        finally:
            os.unlink(path)

    def test_not_operator(self):
        xml = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<extendedFeatureModel>
  <struct>
    <and mandatory="true" name="Root">
      <alt mandatory="true" name="Pick">
        <feature name="A"/>
        <feature name="B"/>
      </alt>
    </and>
  </struct>
  <constraints>
    <rule><not><var>A</var></not></rule>
  </constraints>
</extendedFeatureModel>"""
        cfg = {
            "selected_options": {"Root.Pick": ["A", "B"]},
            "string_values": {},
            "selected_features": [],
            "or_group_mode": {},
        }
        configs = self._run(xml, cfg)
        picked = {c.get("Root.Pick") for c in configs}
        assert picked == {"B"}

    def test_and_operator(self):
        xml = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<extendedFeatureModel>
  <struct>
    <and mandatory="true" name="Root">
      <or mandatory="true" name="PickMany">
        <feature name="A"/>
        <feature name="B"/>
      </or>
    </and>
  </struct>
  <constraints>
    <rule><and><var>A</var><var>B</var></and></rule>
  </constraints>
</extendedFeatureModel>"""
        cfg = {
            "selected_options": {"Root.PickMany": ["A", "B"]},
            "string_values": {},
            "selected_features": [],
            "or_group_mode": {"Root.PickMany": "combine"},
        }
        configs = self._run(xml, cfg)
        assert len(configs) == 1
        assert set(configs[0]["Root.PickMany"]) == {"A", "B"}

    def test_or_operator(self):
        xml = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<extendedFeatureModel>
  <struct>
    <and mandatory="true" name="Root">
      <alt mandatory="true" name="PickOne">
        <feature name="A"/>
        <feature name="B"/>
      </alt>
    </and>
  </struct>
  <constraints>
    <rule><or><var>A</var><var>B</var></or></rule>
  </constraints>
</extendedFeatureModel>"""
        cfg = {
            "selected_options": {"Root.PickOne": ["A"]},
            "string_values": {},
            "selected_features": [],
            "or_group_mode": {},
        }
        configs = self._run(xml, cfg)
        assert len(configs) == 1
        assert configs[0]["Root.PickOne"] == "A"

    def test_eq_operator(self):
        xml = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<extendedFeatureModel>
  <struct>
    <and mandatory="true" name="Root">
      <or mandatory="true" name="PickMany">
        <feature name="A"/>
        <feature name="B"/>
      </or>
    </and>
  </struct>
  <constraints>
    <rule><eq><var>A</var><var>B</var></eq></rule>
  </constraints>
</extendedFeatureModel>"""
        cfg = {
            "selected_options": {"Root.PickMany": ["A", "B"]},
            "string_values": {},
            "selected_features": [],
            "or_group_mode": {"Root.PickMany": "combine"},
        }
        configs = self._run(xml, cfg)
        assert len(configs) == 1
        assert set(configs[0]["Root.PickMany"]) == {"A", "B"}

    def test_nested_constraint_expression(self):
        xml = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<extendedFeatureModel>
  <struct>
    <and mandatory="true" name="Root">
      <alt mandatory="true" name="Source">
        <feature name="A"/>
        <feature name="B"/>
      </alt>
      <alt mandatory="true" name="Target">
        <feature name="C"/>
        <feature name="D"/>
      </alt>
    </and>
  </struct>
  <constraints>
    <rule>
      <imp>
        <or><var>A</var><var>B</var></or>
        <and><var>C</var><var>D</var></and>
      </imp>
    </rule>
  </constraints>
</extendedFeatureModel>"""
        cfg = {
            "selected_options": {
                "Root.Source": ["A", "B"],
                "Root.Target": ["C", "D"],
            },
            "string_values": {},
            "selected_features": [],
            "or_group_mode": {},
            "allow_empty_fallback": False,
        }
        with pytest.raises(ValueError, match="No valid FM configurations"):
            self._run(xml, cfg)


class TestMandatoryGroupEnforcement:
    """Mandatory group/string semantics should fail when unsatisfied."""

    def test_mandatory_alt_group_without_selection_is_invalid(self):
        cfg = _base_config(
            selected_options={
                "Artefact.Requirement.Description.DescriptionType": [],
            },
        )
        cfg["allow_empty_fallback"] = False

        with pytest.raises(ValueError, match="No valid FM configurations"):
            _resolve(cfg, FM_FIXTURE)

    def test_mandatory_or_group_without_selection_is_invalid(self):
        xml = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<extendedFeatureModel>
  <struct>
    <and mandatory="true" name="Root">
      <or mandatory="true" name="PickAny">
        <feature name="A"/>
        <feature name="B"/>
      </or>
    </and>
  </struct>
  <constraints/>
</extendedFeatureModel>"""

        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".xml")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(xml)
            cfg = _base_config()
            cfg["allow_empty_fallback"] = False
            with pytest.raises(ValueError, match="No valid FM configurations"):
                _resolve(cfg, path)
        finally:
            os.unlink(path)

    def test_mandatory_string_feature_without_value_is_invalid(self):
        cfg = _base_config(
            selected_options={
                "Artefact.Requirement.Description.DescriptionType": ["ProseNL"],
            },
            string_values={
                "Artefact.Requirement.Description.Language": [],
            },
        )
        cfg["allow_empty_fallback"] = False
        with pytest.raises(ValueError, match="No valid FM configurations"):
            _resolve(cfg, FM_FIXTURE)


