"""Tests for XAI Explainer."""
import os
import sys
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tgf_dosing.core.explainer import DosingExplainer


@dataclass
class MockRiskAssessment:
    lsi: float = 1.0
    rsi: float = 6.5
    scaling_risk: float = 0.5
    corrosion_risk: float = 0.1
    biofouling_risk: float = 0.1
    overall_risk: float = 0.3
    risk_level: str = "MODERATE"
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {"biofouling": "Adequate biocide"}


@dataclass
class MockDecision:
    continuous_doses_kg: dict = None
    slug_doses: dict = None
    blowdown_command: float = 0.2
    primary_risk: str = "scaling"
    risk_horizon_hours: float = 0.0
    preemptive: bool = False

    def __post_init__(self):
        if self.continuous_doses_kg is None:
            self.continuous_doses_kg = {"AQUATREAT-2501": 0.5, "AQUATREAT-1196": 0.3}
        if self.slug_doses is None:
            self.slug_doses = {}


@pytest.fixture
def explainer():
    return DosingExplainer()


@pytest.fixture
def tower_config():
    from tgf_dosing.config.tower_config import AQUATECH_850_TPD
    return AQUATECH_850_TPD


def test_explain_returns_string(explainer, tower_config):
    """explain() returns a non-empty string."""
    risk = MockRiskAssessment()
    decision = MockDecision()
    residuals = {"AQUATREAT-2501": 10.0, "AQUATREAT-1196": 30.0}

    explanation = explainer.explain(risk, residuals, decision, tower_config, coc=5.0)
    assert isinstance(explanation, str)
    assert len(explanation) > 0


def test_explain_mentions_risk(explainer, tower_config):
    """Explanation mentions scaling risk when it's elevated."""
    risk = MockRiskAssessment(scaling_risk=0.7, lsi=1.8)
    decision = MockDecision()
    residuals = {"AQUATREAT-2501": 10.0}

    explanation = explainer.explain(risk, residuals, decision, tower_config)
    assert "scaling" in explanation.lower()


def test_explain_preemptive(explainer, tower_config):
    """Preemptive decisions are labeled as such."""
    risk = MockRiskAssessment()
    decision = MockDecision(preemptive=True, risk_horizon_hours=6.0)
    residuals = {"AQUATREAT-2501": 10.0}

    explanation = explainer.explain(risk, residuals, decision, tower_config)
    assert "reemptive" in explanation


def test_explain_routine(explainer, tower_config):
    """Low-risk scenarios produce routine explanation."""
    risk = MockRiskAssessment(scaling_risk=0.1, corrosion_risk=0.1, biofouling_risk=0.1)
    decision = MockDecision()
    residuals = {"AQUATREAT-2501": 12.0, "AQUATREAT-1196": 35.0}

    explanation = explainer.explain(risk, residuals, decision, tower_config)
    assert isinstance(explanation, str)


def test_factor_breakdown(explainer, tower_config):
    """get_factor_breakdown returns structured dict."""
    risk = MockRiskAssessment(scaling_risk=0.6)
    decision = MockDecision()
    residuals = {"AQUATREAT-2501": 10.0}

    breakdown = explainer.get_factor_breakdown(risk, residuals, decision, tower_config)
    assert "explanation" in breakdown
    assert "factors" in breakdown
    assert isinstance(breakdown["factors"], list)
    assert "preemptive" in breakdown
    assert "primary_risk" in breakdown


def test_explain_with_slug_dose(explainer, tower_config):
    """Explanation mentions slug doses when present."""
    risk = MockRiskAssessment()
    decision = MockDecision(slug_doses={"AQUATREAT-3331": 120.0})
    residuals = {}

    explanation = explainer.explain(risk, residuals, decision, tower_config)
    assert "lug" in explanation


def test_explain_high_blowdown(explainer, tower_config):
    """Explanation mentions blowdown when high."""
    risk = MockRiskAssessment()
    decision = MockDecision(blowdown_command=0.8)
    residuals = {}

    explanation = explainer.explain(risk, residuals, decision, tower_config)
    assert "lowdown" in explanation
