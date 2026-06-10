"""Diagnostics ("doctor") for AI/LLM configuration and hardware.

Produces a human-readable report covering detected RAM, the RAM-based model
recommendation, the resolved configuration (honoring "auto" and overrides), the
selected tier-2 provider, and live Ollama reachability/model availability. Used
by ``investment-monitor --doctor``.
"""

from __future__ import annotations

from .analysis.hardware import recommend_models, total_ram_gb
from .analysis.ollama_client import model_matches, model_names
from .config import Settings, get_settings


def _probe_ollama(host: str) -> tuple[bool, list[str], str | None]:
    """Probe an Ollama server for reachability and installed models.

    Args:
        host: The Ollama server URL.

    Returns:
        A tuple ``(reachable, installed_model_tags, error_message)``. When the
        server is unreachable or the package is missing, ``reachable`` is False
        and ``error_message`` explains why.
    """
    try:
        import ollama
    except ImportError:
        return False, [], "ollama package not installed (pip install 'investment-monitor[ai]')"
    try:
        client = ollama.Client(host=host)
        return True, model_names(client.list()), None
    except Exception as e:  # noqa: BLE001 - report any connection/runtime error
        return False, [], str(e)


def build_doctor_report(settings: Settings | None = None) -> str:
    """Build the diagnostics report string.

    Args:
        settings: Settings to inspect (defaults to the loaded application settings).

    Returns:
        A multi-line, human-readable diagnostics report.
    """
    settings = settings or get_settings()
    lines: list[str] = []
    lines.append("Investment Monitor - AI / LLM diagnostics")
    lines.append("=" * 44)

    # Hardware + recommendation
    ram = total_ram_gb()
    rec = recommend_models(ram)
    ram_str = f"{ram:.1f} GiB" if ram is not None else "unknown (using conservative fallback)"
    lines.append("")
    lines.append("Hardware")
    lines.append(f"  Detected RAM      : {ram_str}")
    lines.append(f"  Capability tier   : {rec.tier}")
    lines.append(f"  Suggested fast    : {rec.fast}")
    lines.append(f"  Suggested synth   : {rec.synthesis}")

    # Resolved configuration
    fast = settings.resolved_ollama_model()
    synth = settings.resolved_synthesis_model()
    use_claude = settings.prefer_anthropic_synthesis()
    lines.append("")
    lines.append("Resolved configuration")
    lines.append(f"  OLLAMA_HOST            : {settings.ollama_host}")
    lines.append(f"  OLLAMA_MODEL           : {settings.ollama_model} -> {fast}")
    lines.append(f"  OLLAMA_SYNTHESIS_MODEL : {settings.ollama_synthesis_model} -> {synth}")
    lines.append(f"  LLM_PROVIDER           : {settings.llm_provider}")
    lines.append(
        f"  Tier-2 provider        : {'Claude (Anthropic)' if use_claude else 'local Ollama (free)'}"
    )
    if use_claude:
        lines.append(
            f"  ANTHROPIC_API_KEY      : {'set' if settings.anthropic_api_key else 'MISSING'}"
        )

    # Live Ollama probe
    reachable, installed, err = _probe_ollama(settings.ollama_host)
    lines.append("")
    lines.append("Ollama")
    if reachable:
        lines.append(f"  Server                 : reachable at {settings.ollama_host}")
        lines.append(f"  Installed models       : {', '.join(installed) or '(none)'}")
        for label, model in (("Fast", fast), ("Synth", synth)):
            if model_matches(installed, model):
                lines.append(f"  {label} model             : OK ({model})")
            else:
                lines.append(
                    f"  {label} model             : MISSING ({model}) -> ollama pull {model}"
                )
    else:
        lines.append(f"  Server                 : NOT reachable ({err})")
        lines.append("  Fix                    : install Ollama, run `ollama serve`, then:")
        lines.append(f"                             ollama pull {fast}")
        if synth != fast:
            lines.append(f"                             ollama pull {synth}")

    return "\n".join(lines)
