from __future__ import annotations

import re
from dataclasses import dataclass, field


CATEGORIES = [
    "Аммиак",
    "Фосфаты",
    "Калий",
    "Энергоносители",
    "Логистика",
    "Спрос/агрорынок",
    "Макро/валюта",
    "Прочее",
]

CATEGORY_PRODUCT_PRIORS: dict[str, dict[str, float]] = {
    "Аммиак": {"urea": 1.0, "dap": 0.2, "mop": 0.1},
    "Фосфаты": {"urea": 0.2, "dap": 1.0, "mop": 0.2},
    "Калий": {"urea": 0.1, "dap": 0.2, "mop": 1.0},
    "Энергоносители": {"urea": 1.0, "dap": 0.4, "mop": 0.2},
    "Логистика": {"urea": 0.5, "dap": 0.5, "mop": 0.8},
    "Спрос/агрорынок": {"urea": 0.6, "dap": 0.7, "mop": 0.6},
    "Макро/валюта": {"urea": 0.5, "dap": 0.5, "mop": 0.5},
    "Прочее": {"urea": 0.0, "dap": 0.0, "mop": 0.0},
}

SOURCE_IMPORTANCE: dict[str, float] = {
    "Reuters": 0.85,
    "Коммерсантъ": 0.9,
    "ICIS": 0.75,
    "CRU": 0.75,
    "Argus": 0.7,
    "Bloomberg": 0.8,
    "Интерфакс": 0.65,
    "Профильный телеграм-канал": 0.7,
}

PROCHEE_PATTERNS = re.compile(
    r"ESG|конференц|меморандум|день инвестора|научн\w+ сотруднич",
    re.IGNORECASE,
)

BEARISH_PATTERNS = re.compile(
    r"дешеве|снижа|пада|слаб\w+ спрос|идут вниз|давлени\w+ вниз|перепроизвод",
    re.IGNORECASE,
)

BULLISH_PATTERNS = re.compile(
    r"дефицит|вверх|подорож|рост цен|сокращ\w+ мощност|сорван|взлетел|толка\w+ цен",
    re.IGNORECASE,
)

CATEGORY_KEYWORDS: dict[str, re.Pattern] = {
    "Аммиак": re.compile(r"аммиак|ammonia|мочевин|urea", re.IGNORECASE),
    "Фосфаты": re.compile(r"DAP|MAP|фосфат|phosphate", re.IGNORECASE),
    "Калий": re.compile(r"кали|MOP|potash|хлорист", re.IGNORECASE),
    "Энергоносители": re.compile(r"газ|нефт|Brent|энерг|LNG|СПГ", re.IGNORECASE),
    "Логистика": re.compile(r"фрахт|логист|порт|контейнер|поставк", re.IGNORECASE),
    "Спрос/агрорынок": re.compile(r"агро|фермер|урожа|спрос|культур", re.IGNORECASE),
    "Макро/валюта": re.compile(r"рубл|валют|USD|макро|санкц|ставк", re.IGNORECASE),
}


@dataclass
class NewsHints:
    likely_prochee: bool = False
    direction: float = 0.0
    suggested_category: str | None = None
    source_importance: float = 0.5
    product_priors: dict[str, float] = field(default_factory=lambda: {"urea": 0.5, "dap": 0.5, "mop": 0.5})
    notes: list[str] = field(default_factory=list)


def analyze_headline(source: str, headline: str) -> NewsHints:
    hints = NewsHints(source_importance=SOURCE_IMPORTANCE.get(source, 0.5))

    if PROCHEE_PATTERNS.search(headline):
        hints.likely_prochee = True
        hints.suggested_category = "Прочее"
        hints.product_priors = CATEGORY_PRODUCT_PRIORS["Прочее"].copy()
        hints.notes.append("likely_Прочее: шаблоны ESG/конференция/меморандум")

    bearish = len(BEARISH_PATTERNS.findall(headline))
    bullish = len(BULLISH_PATTERNS.findall(headline))
    if bearish > bullish:
        hints.direction = -0.7
        hints.notes.append("lexicon: медвежий тон (давление на снижение цен)")
    elif bullish > bearish:
        hints.direction = 0.7
        hints.notes.append("lexicon: бычий тон (давление на рост цен)")

    best_cat = None
    best_score = 0
    for cat, pattern in CATEGORY_KEYWORDS.items():
        if pattern.search(headline):
            score = len(pattern.findall(headline))
            if score > best_score:
                best_score = score
                best_cat = cat

    if best_cat and not hints.likely_prochee:
        hints.suggested_category = best_cat
        hints.product_priors = CATEGORY_PRODUCT_PRIORS[best_cat].copy()
        hints.notes.append(f"keyword hint: {best_cat}")

    if hints.source_importance >= 0.85:
        hints.notes.append(f"source prior: высокая ожидаемая амплитуда ({source})")

    return hints


def format_hints(hints: NewsHints) -> str:
    lines = [
        f"likely_prochee: {hints.likely_prochee}",
        f"direction_hint: {hints.direction:+.2f}",
        f"suggested_category: {hints.suggested_category or 'none'}",
        f"source_importance: {hints.source_importance:.2f}",
        f"product_priors: urea={hints.product_priors['urea']}, dap={hints.product_priors['dap']}, mop={hints.product_priors['mop']}",
    ]
    if hints.notes:
        lines.append("notes: " + "; ".join(hints.notes))
    return "\n".join(lines)
