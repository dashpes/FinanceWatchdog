"""The Study: calibration and adaptation — win rate, Brier, weight tilts."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from investment_monitor.storage.learning_models import (
    LEARNING_KIND_OUTCOME,
    LEARNING_KIND_WEIGHT_ADAPTATION,
    LearningEvent,
)

from ..deps import get_read_session
from ._serialize import iso

router = APIRouter(tags=["learning"])


@router.get("/learning/summary")
def summary(session: Session = Depends(get_read_session)) -> dict:
    from investment_monitor.storage.learning_operations import (
        accuracy_stats_for_symbol,
        get_outcome_symbols,
    )

    outcomes = (
        session.query(LearningEvent)
        .filter(LearningEvent.kind == LEARNING_KIND_OUTCOME)
        .order_by(LearningEvent.id.asc())
        .all()
    )
    hits = [int(e.direction_correct or 0) for e in outcomes]
    briers = [float(e.brier) for e in outcomes if e.brier is not None]

    per_symbol = [
        {"symbol": s, **accuracy_stats_for_symbol(session, s)}
        for s in get_outcome_symbols(session)
    ]

    adaptations = (
        session.query(LearningEvent)
        .filter(LearningEvent.kind == LEARNING_KIND_WEIGHT_ADAPTATION)
        .order_by(LearningEvent.id.desc())
        .limit(50)
        .all()
    )

    return {
        "totals": {
            "n_outcomes": len(outcomes),
            "win_rate": (sum(hits) / len(hits)) if hits else None,
            "mean_brier": (sum(briers) / len(briers)) if briers else None,
        },
        "outcome_series": [
            {
                "as_of_date": iso(e.as_of_date),
                "symbol": e.symbol,
                "conviction": e.conviction,
                "realized_return": e.realized_return,
                "direction_correct": e.direction_correct,
                "brier": e.brier,
            }
            for e in outcomes
        ],
        "per_symbol": per_symbol,
        "adaptations": [
            {
                "as_of_date": iso(e.as_of_date),
                "symbol": e.symbol,
                "before_value": e.before_value,
                "after_value": e.after_value,
                "applied": e.applied,
                "note": e.note,
            }
            for e in adaptations
        ],
    }
