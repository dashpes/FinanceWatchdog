"""Report formatter for Monte Carlo simulation results.

This module provides formatting utilities to convert SimulationOutput
into human-readable formats for email reports (markdown) and Slack (compact).
"""

from .models import SimulationOutput

# Default disclaimer text
DEFAULT_DISCLAIMER = (
    "Simulation based on historical returns. Not a prediction. "
    "Past performance does not guarantee future results."
)


class SimulationReportFormatter:
    """Formatter for Monte Carlo simulation results.

    This class converts SimulationOutput into formatted strings suitable
    for inclusion in research reports (markdown for email) or brief
    notifications (compact text for Slack).

    Attributes:
        _disclaimer: The disclaimer text to include in reports.

    Example:
        >>> formatter = SimulationReportFormatter()
        >>> markdown = formatter.format_markdown(simulation_output)
        >>> slack_text = formatter.format_compact(simulation_output)
    """

    def __init__(self, disclaimer: str | None = None) -> None:
        """Initialize with optional custom disclaimer.

        Args:
            disclaimer: Custom disclaimer text. If None, uses the default
                disclaimer about historical simulations not being predictions.
        """
        self._disclaimer = disclaimer if disclaimer is not None else DEFAULT_DISCLAIMER

    def format_markdown(self, output: SimulationOutput) -> str:
        """Format simulation results as markdown for email reports.

        Creates a detailed markdown report including:
        - Header: "## Risk Analysis (Monte Carlo Simulation)"
        - Entry info: price, simulations count, lookback years
        - Projected Price Ranges table by horizon
        - Stress Test Results table (longest horizon)
        - Risk Metrics (VaR, CVaR, probability of gain)
        - Sensitivity Check table
        - Disclaimer

        Args:
            output: The SimulationOutput from MonteCarloAnalyzer.

        Returns:
            Formatted markdown string suitable for email reports.
        """
        lines = []

        # Header
        lines.append("## Risk Analysis (Monte Carlo Simulation)")
        lines.append("")

        # Entry info
        lookback_years = output.lookback_days / 252
        lines.append(
            f"**Entry Point:** ${output.entry_price:,.2f} | "
            f"**Simulations:** {output.num_simulations:,} paths | "
            f"**Data:** {lookback_years:.1f} years"
        )
        lines.append("")

        # Projected Price Ranges table
        lines.append("### Projected Price Ranges")
        lines.append("")
        lines.append("| Horizon | Expected | 80% Confidence | Worst 5% |")
        lines.append("|---------|----------|----------------|----------|")

        # Sort horizons to ensure consistent order
        sorted_horizons = sorted(output.results.keys())
        for days in sorted_horizons:
            horizon_result = output.results[days]
            label = self._format_horizon_label(days)
            expected = f"${horizon_result.base_mean:.0f}"
            ci_80 = f"${horizon_result.base_ci_80[0]:.0f} - ${horizon_result.base_ci_80[1]:.0f}"
            worst_5 = f"Below ${horizon_result.base_percentiles.get(5, horizon_result.base_ci_95[0]):.0f}"
            lines.append(f"| {label} | {expected} | {ci_80} | {worst_5} |")

        lines.append("")

        # Stress Test Results (use longest horizon)
        longest_horizon = max(output.results.keys())
        longest_result = output.results[longest_horizon]

        if longest_result.scenarios:
            horizon_label = self._format_horizon_label(longest_horizon)
            lines.append(f"### Stress Test Results ({horizon_label} Horizon)")
            lines.append("")
            lines.append("| Scenario | Expected | 80% Range | Chance of >20% Loss |")
            lines.append("|----------|----------|-----------|---------------------|")

            for scenario_name, scenario in longest_result.scenarios.items():
                expected = f"${scenario.mean:.0f}"
                ci_80 = f"${scenario.ci_80[0]:.0f} - ${scenario.ci_80[1]:.0f}"
                prob_loss = f"{scenario.prob_loss_20pct * 100:.0f}%"
                lines.append(f"| {scenario_name} | {expected} | {ci_80} | {prob_loss} |")

            lines.append("")

        # Risk Metrics
        lines.append("### Risk Metrics")
        lines.append("")
        var_pct = longest_result.base_var_95 * 100
        cvar_pct = longest_result.base_cvar_95 * 100

        # Calculate probability of gain from base case
        # If mean > entry_price, estimate prob of gain from distribution
        prob_gain = self._estimate_prob_gain(output.entry_price, longest_result)

        lines.append(f"- **Value at Risk (95%):** {var_pct:.1f}%")
        lines.append(f"- **Conditional VaR (95%):** {cvar_pct:.1f}%")
        lines.append(f"- **Base Case Probability of Gain:** {prob_gain:.0f}%")
        lines.append("")

        # Sensitivity Check
        lines.append("### Sensitivity Check")
        lines.append("")
        lines.append("| Input Assumption | Impact on Results |")
        lines.append("|------------------|-------------------|")

        sensitivity = output.sensitivity

        # Volatility impact
        vol_impact_label = self._impact_label(sensitivity.volatility_impact)
        vol_swing = self._calculate_swing(sensitivity.volatility_range)
        lines.append(f"| Volatility | {vol_impact_label} - {vol_swing} swing |")

        # Drift impact
        drift_impact_label = self._impact_label(sensitivity.drift_impact)
        drift_swing = self._calculate_swing_from_dict(sensitivity.drift_range)
        lines.append(f"| Return Assumption | {drift_impact_label} - {drift_swing} swing |")

        # Lookback impact
        lookback_impact_label = self._impact_label(sensitivity.lookback_impact)
        lookback_swing = self._calculate_swing_from_dict(sensitivity.lookback_range)
        lines.append(f"| Lookback Period | {lookback_impact_label} - {lookback_swing} swing |")

        lines.append("")
        lines.append(f"**Bottom Line:** Projections are most sensitive to {sensitivity.primary_driver} assumptions.")
        lines.append("")

        # Disclaimer
        lines.append("---")
        lines.append(f"*{self._disclaimer}*")

        return "\n".join(lines)

    def format_compact(self, output: SimulationOutput) -> str:
        """Format simulation results as compact text for Slack.

        Creates a brief summary suitable for Slack notifications.
        The output is guaranteed to be under 1000 characters.

        Args:
            output: The SimulationOutput from MonteCarloAnalyzer.

        Returns:
            Compact text string under 1000 characters.
        """
        lines = []

        # Title with ticker
        lines.append(f"*Monte Carlo Risk Analysis: {output.ticker}*")
        lines.append("")

        # Entry info (compact)
        lines.append(f"Entry: ${output.entry_price:,.2f} | {output.num_simulations:,} sims")
        lines.append("")

        # Quick price projections for longest horizon
        longest_horizon = max(output.results.keys())
        longest_result = output.results[longest_horizon]
        horizon_label = self._format_horizon_label(longest_horizon)

        lines.append(f"*{horizon_label} Outlook:*")
        lines.append(
            f"Expected: ${longest_result.base_mean:.0f} | "
            f"Range: ${longest_result.base_ci_80[0]:.0f}-${longest_result.base_ci_80[1]:.0f}"
        )
        lines.append("")

        # Key risk metric
        var_pct = longest_result.base_var_95 * 100
        lines.append(f"VaR (95%): {var_pct:.1f}%")

        # Worst stress scenario
        if longest_result.scenarios:
            worst_scenario = min(
                longest_result.scenarios.values(),
                key=lambda s: s.mean
            )
            lines.append(
                f"Worst Case ({worst_scenario.name}): "
                f"${worst_scenario.mean:.0f} ({worst_scenario.prob_loss_20pct * 100:.0f}% chance >20% loss)"
            )

        result = "\n".join(lines)

        # Ensure under 1000 chars (truncate if somehow over)
        if len(result) > 995:
            result = result[:995] + "..."

        return result

    def _format_horizon_label(self, days: int) -> str:
        """Convert days to human-readable label.

        Args:
            days: Number of trading days.

        Returns:
            Formatted label: "30 days", "90 days", or "1 year" for 252 days.
        """
        if days == 252:
            return "1 year"
        return f"{days} days"

    def _impact_label(self, impact: float) -> str:
        """Convert impact score to HIGH/MEDIUM/LOW label.

        Args:
            impact: Impact score from 0-100.

        Returns:
            "HIGH" if >= 70, "MEDIUM" if >= 40, else "LOW".
        """
        if impact >= 70:
            return "HIGH"
        elif impact >= 40:
            return "MEDIUM"
        else:
            return "LOW"

    def _estimate_prob_gain(self, entry_price: float, horizon_result) -> float:
        """Estimate probability of gain from distribution characteristics.

        Uses a simple approximation based on mean, std, and entry price.

        Args:
            entry_price: The entry price for the position.
            horizon_result: HorizonResult with distribution statistics.

        Returns:
            Estimated probability of gain (0-100).
        """
        mean = horizon_result.base_mean
        std = horizon_result.base_std

        if std == 0:
            return 100.0 if mean > entry_price else 0.0

        # Z-score for entry price
        z = (entry_price - mean) / std

        # Approximate using empirical rule / normal CDF approximation
        # P(X > entry) = P(Z > z)
        # Simple sigmoid approximation for normal CDF
        from math import erf, sqrt

        # CDF approximation: 0.5 * (1 + erf(z / sqrt(2)))
        prob_below = 0.5 * (1 + erf(z / sqrt(2)))
        prob_gain = (1 - prob_below) * 100

        return max(0.0, min(100.0, prob_gain))

    def _calculate_swing(self, range_dict: dict[float, float]) -> str:
        """Calculate swing percentage from volatility multiplier range.

        Args:
            range_dict: Dict mapping multipliers to prices.

        Returns:
            Formatted swing string like "+-16%".
        """
        if not range_dict:
            return "N/A"

        values = list(range_dict.values())
        if len(values) < 2:
            return "N/A"

        mid_value = values[len(values) // 2]
        max_value = max(values)
        min_value = min(values)

        swing_up = (max_value - mid_value) / mid_value * 100 if mid_value != 0 else 0
        swing_down = (mid_value - min_value) / mid_value * 100 if mid_value != 0 else 0
        avg_swing = (swing_up + swing_down) / 2

        return f"+-{avg_swing:.0f}%"

    def _calculate_swing_from_dict(self, range_dict: dict) -> str:
        """Calculate swing percentage from a generic range dictionary.

        Args:
            range_dict: Dict mapping any keys to price values.

        Returns:
            Formatted swing string like "+-6%".
        """
        if not range_dict:
            return "N/A"

        values = list(range_dict.values())
        if len(values) < 2:
            return "N/A"

        mid_value = sum(values) / len(values)
        max_value = max(values)
        min_value = min(values)

        if mid_value == 0:
            return "N/A"

        swing_up = (max_value - mid_value) / mid_value * 100
        swing_down = (mid_value - min_value) / mid_value * 100
        avg_swing = (swing_up + swing_down) / 2

        return f"+-{avg_swing:.0f}%"
