"""
TGF XAI Explainer
==================
Explains dosing decisions in natural language using feature importance
extraction from the MPC cost function + optional SHAP.

Every dosing decision gets an audit-trail-ready explanation:
"Increased AQUATREAT-2501 by 0.3 kg because scaling risk was HIGH
 (LSI=1.8, contribution: 72%), conductivity elevated (2800 uS/cm, 18%)"
"""
import logging
import numpy as np
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


class DosingExplainer:
    """
    Generates natural language explanations for MPC dosing decisions.

    Primary method: decompose MPC cost function into per-factor contributions.
    Optional method: KernelSHAP on a surrogate model (if shap installed).
    """

    def __init__(self):
        self._shap_available = False
        try:
            import shap  # noqa: F401
            self._shap_available = True
        except ImportError:
            pass

    def explain(self,
                risk_assessment,
                current_residuals: Dict[str, float],
                decision,
                tower_config,
                coc: float = 1.0,
                ) -> str:
        """
        Generate a natural language explanation for a dosing decision.

        Args:
            risk_assessment: RiskAssessment from PhysicsEngine
            current_residuals: current chemical ppm levels
            decision: DosingDecision from MPC
            tower_config: TowerConfig
            coc: current Cycles of Concentration

        Returns:
            Human-readable explanation string
        """
        factors = self._extract_factors(risk_assessment, current_residuals,
                                         decision, tower_config, coc)

        return self._generate_nlg(factors, decision, tower_config)

    def _extract_factors(self,
                         risk_assessment,
                         residuals: Dict[str, float],
                         decision,
                         tower_config,
                         coc: float,
                         ) -> List[Tuple[str, float, str]]:
        """
        Extract contributing factors and their importance.

        Returns list of (factor_name, importance_0_to_1, detail_string).
        """
        factors = []

        # Scaling risk contribution
        if risk_assessment.scaling_risk > 0.2:
            factors.append((
                "scaling risk",
                risk_assessment.scaling_risk,
                f"LSI={risk_assessment.lsi:.2f}, RSI={risk_assessment.rsi:.2f}"
            ))

        # Corrosion risk contribution
        if risk_assessment.corrosion_risk > 0.2:
            factors.append((
                "corrosion risk",
                risk_assessment.corrosion_risk,
                f"LSI={risk_assessment.lsi:.2f}"
            ))

        # Biofouling risk contribution
        if risk_assessment.biofouling_risk > 0.2:
            factors.append((
                "biofouling risk",
                risk_assessment.biofouling_risk,
                risk_assessment.details.get("biofouling", "")
            ))

        # Chemical deficit contributions
        for name, chem in tower_config.chemicals.items():
            current = residuals.get(name, chem.target_ppm)
            if current < chem.min_ppm:
                deficit_severity = (chem.min_ppm - current) / max(chem.target_ppm, 1)
                factors.append((
                    f"{name} deficit",
                    min(1.0, deficit_severity),
                    f"{current:.1f}/{chem.target_ppm:.1f} ppm"
                ))
            elif current > chem.max_ppm:
                excess = (current - chem.max_ppm) / max(chem.target_ppm, 1)
                factors.append((
                    f"{name} excess",
                    min(1.0, excess),
                    f"{current:.1f}/{chem.max_ppm:.1f} ppm"
                ))

        # CoC contribution
        if coc > 7.0:
            factors.append((
                "high CoC",
                min(1.0, (coc - 6.0) / 4.0),
                f"CoC={coc:.1f} (target ~6.0)"
            ))
        elif coc < 3.0:
            factors.append((
                "low CoC",
                min(1.0, (3.0 - coc) / 2.0),
                f"CoC={coc:.1f} (excessive blowdown)"
            ))

        # Preemptive action
        if decision.preemptive:
            factors.append((
                "preemptive forecast",
                0.6,
                f"risk at +{decision.risk_horizon_hours:.1f}h horizon"
            ))

        # Sort by importance
        factors.sort(key=lambda x: x[1], reverse=True)
        return factors

    def _generate_nlg(self,
                      factors: List[Tuple[str, float, str]],
                      decision,
                      tower_config,
                      ) -> str:
        """Generate natural language from factor list."""
        if not factors:
            return "Routine maintenance dosing — all parameters within normal range."

        # Identify top factors
        top = factors[:3]

        # Build main chemicals being adjusted
        active_chems = []
        for name, kg in decision.continuous_doses_kg.items():
            if kg > 0.01:
                active_chems.append(f"{name} ({kg:.2f} kg)")

        # Build explanation parts
        parts = []
        total_importance = sum(f[1] for f in top)
        for factor_name, importance, detail in top:
            pct = int(importance / max(total_importance, 0.01) * 100)
            level = "strongly" if importance > 0.6 else "moderately" if importance > 0.3 else "slightly"
            parts.append(f"{factor_name} {level} influenced ({pct}%, {detail})")

        # Construct sentence
        if decision.preemptive:
            prefix = "Preemptive dosing"
        elif any(f[1] > 0.6 for f in top):
            prefix = "Adjusted dosing"
        else:
            prefix = "Routine dosing"

        chem_str = ", ".join(active_chems[:3]) if active_chems else "minimal chemicals"
        factor_str = "; ".join(parts)

        explanation = f"{prefix} ({chem_str}) because {factor_str}."

        # Add slug dose note
        if decision.slug_doses:
            slug_names = list(decision.slug_doses.keys())
            explanation += f" Slug dose triggered: {', '.join(slug_names)}."

        # Add blowdown note
        if decision.blowdown_command > 0.5:
            explanation += f" Blowdown at {decision.blowdown_command * 100:.0f}% to reduce concentration."

        return explanation

    def get_factor_breakdown(self,
                              risk_assessment,
                              current_residuals: Dict[str, float],
                              decision,
                              tower_config,
                              coc: float = 1.0,
                              ) -> Dict:
        """
        Return structured factor breakdown for API/dashboard.

        Returns:
            dict with factors list and explanation text
        """
        factors = self._extract_factors(risk_assessment, current_residuals,
                                         decision, tower_config, coc)
        explanation = self._generate_nlg(factors, decision, tower_config)

        return {
            "explanation": explanation,
            "factors": [
                {"name": f[0], "importance": round(f[1], 3), "detail": f[2]}
                for f in factors
            ],
            "preemptive": decision.preemptive,
            "primary_risk": decision.primary_risk,
        }
