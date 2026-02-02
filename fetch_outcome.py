import csv
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

GAMMA_MARKET_BY_SLUG_URL = "https://gamma-api.polymarket.com/markets/slug/{slug}"

# -------------------------
# Config
# -------------------------
@dataclass
class LabelConfig:
    # Consideramos “resuelto” por precio si max(p) >= HIGH y min(p) <= LOW
    HIGH: float = 0.99
    LOW: float = 0.01
    # rate limiting suave
    SLEEP_S: float = 0.12
    TIMEOUT_S: int = 30


CFG = LabelConfig()


# -------------------------
# Parsing del input (categorías + slugs)
# -------------------------
def parse_category_slug_list(raw_text: str) -> List[Tuple[str, str]]:
    """
    Formato soportado:
      - Líneas con una sola palabra (p.ej. politics, tech, world, Science, culture:) => categoría
      - Otras líneas => slug
    """
    pairs: List[Tuple[str, str]] = []
    current_cat: Optional[str] = None

    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue

        # "culture:" -> "culture"
        if s.endswith(":") and len(s) > 1:
            current_cat = s[:-1].strip().lower()
            continue

        # Categoría: una palabra, sin guiones, corta
        is_category = ("-" not in s) and (len(s) <= 40) and re.match(r"^[A-Za-z_]+$", s) is not None
        if is_category:
            current_cat = s.lower()
            continue

        if current_cat is None:
            current_cat = "unknown"

        pairs.append((current_cat, s))

    return pairs


# -------------------------
# Helpers de parsing robusto (Gamma a veces devuelve arrays como strings)
# -------------------------
def _maybe_json_loads(x: Any) -> Any:
    if isinstance(x, str):
        xs = x.strip()
        # intenta parsear JSON solo si parece JSON
        if (xs.startswith("[") and xs.endswith("]")) or (xs.startswith("{") and xs.endswith("}")):
            try:
                return json.loads(xs)
            except Exception:
                return x
    return x


def normalize_outcomes(market: Dict[str, Any]) -> Optional[List[str]]:
    outcomes = market.get("outcomes") or market.get("shortOutcomes")
    outcomes = _maybe_json_loads(outcomes)
    if outcomes is None:
        return None

    if isinstance(outcomes, list):
        return [str(o).strip() for o in outcomes]

    # Algunos schemas antiguos ponen outcomes como string con separador
    if isinstance(outcomes, str):
        # intenta separar por coma si aplica
        if "," in outcomes:
            return [s.strip() for s in outcomes.split(",") if s.strip()]
        return [outcomes.strip()]

    return None


def normalize_outcome_prices(market: Dict[str, Any]) -> Optional[List[float]]:
    prices = market.get("outcomePrices")
    prices = _maybe_json_loads(prices)
    if prices is None:
        return None

    if isinstance(prices, list):
        out: List[float] = []
        for p in prices:
            try:
                out.append(float(p))
            except Exception:
                return None
        return out

    if isinstance(prices, str):
        # try comma-separated
        if "," in prices:
            try:
                return [float(x.strip()) for x in prices.split(",")]
            except Exception:
                return None
        # single value -> not useful
        return None

    return None


def fetch_market_by_slug(slug: str, timeout_s: int) -> Dict[str, Any]:
    url = GAMMA_MARKET_BY_SLUG_URL.format(slug=slug)
    r = requests.get(url, timeout=timeout_s)
    if not r.ok:
        raise requests.HTTPError(f"{r.status_code} {r.reason} for {url}")
    return r.json()


# -------------------------
# Inferencia del outcome
# -------------------------
def infer_outcome(market: Dict[str, Any], cfg: LabelConfig) -> Tuple[Optional[int], Optional[str], str]:
    """
    Retorna: (outcome_int, outcome_label, label_source)
      - outcome_int: 1 (YES) / 0 (NO) / None
      - outcome_label: 'YES'/'NO'/None
      - label_source: explicación breve
    """

    # 1) Intento por campo explícito (si existiera)
    for k in ["resolvedOutcome", "winningOutcome", "outcome", "resolution", "result", "finalOutcome"]:
        v = market.get(k)
        if isinstance(v, str):
            vu = v.strip().upper()
            if vu in {"YES", "NO"}:
                return (1 if vu == "YES" else 0, vu, f"explicit:{k}")

    # 2) Fallback robusto: outcomePrices colapsan a {1,0} cuando está cerrado/archivado
    closed = bool(market.get("closed")) if "closed" in market else None
    archived = bool(market.get("archived")) if "archived" in market else None
    is_terminal = (closed is True) or (archived is True)

    outcomes = normalize_outcomes(market)
    prices = normalize_outcome_prices(market)

    if not outcomes or not prices or len(outcomes) < 2 or len(prices) < 2:
        return (None, None, "no_outcomes_or_prices")

    # Tomamos solo las dos primeras (binario YES/NO)
    outcomes2 = outcomes[:2]
    prices2 = prices[:2]

    # Identifica cuál índice corresponde a YES/NO por el texto (suele ser "Yes"/"No")
    yes_idx = None
    no_idx = None
    for i, o in enumerate(outcomes2):
        ou = o.strip().upper()
        if ou == "YES":
            yes_idx = i
        elif ou == "NO":
            no_idx = i

    # Si no vienen como YES/NO literal, asumimos orden típico ["Yes","No"] pero lo marcamos
    assumed_mapping = False
    if yes_idx is None or no_idx is None:
        assumed_mapping = True
        yes_idx, no_idx = 0, 1

    p_yes = prices2[yes_idx]
    p_no = prices2[no_idx]

    # Condición “colapsado” a {1,0}
    pmax = max(prices2)
    pmin = min(prices2)
    collapsed = (pmax >= cfg.HIGH) and (pmin <= cfg.LOW)

    if is_terminal and collapsed:
        if p_yes > p_no:
            src = "price_proxy_terminal"
            if assumed_mapping:
                src += ":assumed_yes_no_order"
            return (1, "YES", src)
        elif p_no > p_yes:
            src = "price_proxy_terminal"
            if assumed_mapping:
                src += ":assumed_yes_no_order"
            return (0, "NO", src)
        else:
            return (None, None, "price_proxy_tie")

    # Si no está terminal o no colapsó, no damos label
    if not is_terminal:
        return (None, None, "not_terminal")
    return (None, None, "terminal_but_not_collapsed")


# -------------------------
# Main: genera CSV
# -------------------------
def build_outcomes_csv_from_text(raw_text: str, out_csv_path: str, cfg: LabelConfig) -> None:
    pairs = parse_category_slug_list(raw_text)

    rows: List[Dict[str, Any]] = []
    for category, slug in pairs:
        category_url = f"{category}-{slug}"
        try:
            market = fetch_market_by_slug(slug, timeout_s=cfg.TIMEOUT_S)
            y, ylab, src = infer_outcome(market, cfg)

            rows.append({
                "categoria-url": category_url,
                "category": category,
                "slug": slug,
                "resolved_outcome": y,          # 1/0/None
                "resolved_label": ylab,         # YES/NO/None
                "label_source": src,
                "closed": market.get("closed"),
                "archived": market.get("archived"),
                "conditionId": market.get("conditionId"),
                "id": market.get("id"),
                "question": market.get("question"),
            })

        except Exception as e:
            rows.append({
                "categoria-url": category_url,
                "category": category,
                "slug": slug,
                "resolved_outcome": None,
                "resolved_label": None,
                "label_source": f"error:{type(e).__name__}:{e}",
                "closed": None,
                "archived": None,
                "conditionId": None,
                "id": None,
                "question": None,
            })

        time.sleep(cfg.SLEEP_S)

    fieldnames = [
        "categoria-url", "category", "slug",
        "resolved_outcome", "resolved_label", "label_source",
        "closed", "archived", "conditionId", "id", "question",
    ]

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Resumen rápido
    n_total = len(rows)
    n_labeled = sum(1 for r in rows if r["resolved_outcome"] in (0, 1))
    print(f"Wrote {n_total} rows to {out_csv_path}. Labeled: {n_labeled}/{n_total}")


if __name__ == "__main__":
    # OPCIÓN A (recomendada): pon tu lista en un archivo slugs.txt y déjala tal cual (categorías + slugs)
    # Ejemplo: politics\nslug1\nslug2\n\ntech\nslug3...
    INPUT_TXT_PATH = "slugs.txt"
    OUTPUT_CSV_PATH = "market_outcomes.csv"

    with open(INPUT_TXT_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    build_outcomes_csv_from_text(raw, OUTPUT_CSV_PATH, CFG)
