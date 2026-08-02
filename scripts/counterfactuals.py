from __future__ import annotations

import random
from typing import Any, Literal

CounterfactualMode = Literal["reflect", "discrete"]

SITE_VOCAB_7 = [
    "anterior torso",
    "posterior torso",
    "lower extremity",
    "upper extremity",
    "head/neck",
    "palms/soles",
    "oral/genital",
]

DISCRETE_AGES = [20, 35, 50, 65, 80, "unknown"]
DISCRETE_SEX = ["male", "female", "unknown"]
DISCRETE_SITES = SITE_VOCAB_7 + ["unknown"]


def _different_choice(
    values: list[Any],
    original: Any,
    rng: random.Random,
) -> Any:
    candidates = [value for value in values if value != original]
    if not candidates:
        raise ValueError("No alternative counterfactual value is available.")
    return rng.choice(candidates)


def make_task_c_counterfactual(
    *,
    age: Any,
    sex: Any,
    site: Any,
    rng: random.Random,
    mode: CounterfactualMode,
) -> tuple[Any, str, str]:
    sex_norm = str(sex).strip().lower()
    site_norm = str(site).strip().lower()

    if mode == "reflect":
        try:
            age_cf = max(5.0, min(95.0, 95.0 - float(age)))
            if age_cf.is_integer():
                age_cf = int(age_cf)
        except (TypeError, ValueError):
            age_cf = "unknown"

        if sex_norm == "male":
            sex_cf = "female"
        elif sex_norm == "female":
            sex_cf = "male"
        else:
            sex_cf = rng.choice(["male", "female"])

        site_cf = _different_choice(SITE_VOCAB_7, site_norm, rng)
        return age_cf, sex_cf, site_cf

    if mode == "discrete":
        age_cf = _different_choice(DISCRETE_AGES, age, rng)
        sex_cf = _different_choice(DISCRETE_SEX, sex_norm, rng)
        site_cf = _different_choice(DISCRETE_SITES, site_norm, rng)
        return age_cf, sex_cf, site_cf

    raise ValueError(f"Unsupported Task C counterfactual mode: {mode}")
