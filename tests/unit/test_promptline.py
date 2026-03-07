import pytest
from unittest.mock import MagicMock
from pathlib import Path

from synthline.core.fm_parser import FM, FMNode
from synthline.core.fm_resolver import FMResolver
from synthline.core.promptline import Promptline


def _find_node(fm: FM, node_id: str) -> FMNode | None:
    return next((n for n in fm.iter_nodes() if n.id == node_id), None)


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def fm_model():
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "fm.xml"
    return FM(str(fixture_path))


@pytest.fixture
def promptline(mock_llm, mock_logger, fm_model):
    return Promptline(mock_llm, mock_logger, fm_model)


def test_fm_parser_group_types_and_structure(fm_model):
    root = fm_model.root
    assert root.name == "Artefact"
    assert root.node_type == "and"

    description_type = _find_node(fm_model,"Artefact.Requirement.Description.DescriptionType")
    assert description_type is not None
    assert description_type.node_type == "alt"

    defect_injection = _find_node(fm_model,"Artefact.Requirement.QualityProfile.DefectInjection")
    assert defect_injection is not None
    assert defect_injection.node_type == "or"

    language = _find_node(fm_model,"Artefact.Requirement.Description.Language")
    assert language is not None
    assert language.is_string_feature is True


def test_get_atomic_configurations_or_split(promptline):
    features = {
        "samples_per_prompt": 2,
        "fm_configuration": {
            "selected_options": {
                "Artefact.Requirement.RequirementType": [
                    "Artefact.Requirement.RequirementType.Functional",
                    "Artefact.Requirement.RequirementType.Quality",
                ],
                "Artefact.Requirement.Description.DescriptionType": [
                    "ProseNL",
                ],
                "Artefact.Requirement.AbstractionLevel": [
                    "Artefact.Requirement.AbstractionLevel.HighLevel",
                    "Artefact.Requirement.AbstractionLevel.DetailedLevel",
                ],
                "Artefact.Requirement.QualityProfile": [
                    "Artefact.Requirement.QualityProfile.DefectInjection",
                ],
                "Artefact.Requirement.QualityProfile.DefectInjection": [
                    "Artefact.Requirement.QualityProfile.DefectInjection.Ambiguous",
                    "Artefact.Requirement.QualityProfile.DefectInjection.Incomplete",
                ],
            },
            "string_values": {
                "Artefact.Requirement.Description.Language": ["English"],
            },
            "selected_features": [],
            "or_group_mode": {
                "Artefact.Requirement.QualityProfile.DefectInjection": "split",
            },
        },
    }

    configs = promptline.get_atomic_configurations(features)
    assert len(configs) == 8
    assert all(
        isinstance(
            config.get("Artefact.Requirement.QualityProfile.DefectInjection"),
            str,
        )
        for config in configs
    )


def test_get_atomic_configurations_or_combine(promptline):
    features = {
        "samples_per_prompt": 2,
        "fm_configuration": {
            "selected_options": {
                "Artefact.Requirement.RequirementType": [
                    "Artefact.Requirement.RequirementType.Functional",
                    "Artefact.Requirement.RequirementType.Quality",
                ],
                "Artefact.Requirement.Description.DescriptionType": [
                    "ProseNL",
                ],
                "Artefact.Requirement.AbstractionLevel": [
                    "Artefact.Requirement.AbstractionLevel.HighLevel",
                    "Artefact.Requirement.AbstractionLevel.DetailedLevel",
                ],
                "Artefact.Requirement.QualityProfile": [
                    "Artefact.Requirement.QualityProfile.DefectInjection",
                ],
                "Artefact.Requirement.QualityProfile.DefectInjection": [
                    "Artefact.Requirement.QualityProfile.DefectInjection.Ambiguous",
                    "Artefact.Requirement.QualityProfile.DefectInjection.Incomplete",
                ],
            },
            "string_values": {
                "Artefact.Requirement.Description.Language": ["English"],
            },
            "selected_features": [],
            "or_group_mode": {
                "Artefact.Requirement.QualityProfile.DefectInjection": "combine",
            },
        },
    }

    configs = promptline.get_atomic_configurations(features)
    assert len(configs) == 4
    assert all(
        isinstance(
            config.get("Artefact.Requirement.QualityProfile.DefectInjection"),
            list,
        )
        for config in configs
    )


def test_build_prompt_uses_artefact_type_and_constraints(promptline):
    prompt = promptline.build({
        "samples_per_prompt": 3,
        "__fm_constraints__": [
            {"id": "g1", "label": "DescriptionType", "value": "ProseNL"},
            {"id": "g2", "label": "Language", "value": "English"},
        ],
    })

    assert "Generate 3 diverse requirements" in prompt
    assert "DescriptionType: ProseNL" in prompt
    assert "Language: English" in prompt
    assert "JSON array" in prompt
