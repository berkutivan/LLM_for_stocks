from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT, load_config, resolve_path
from src.data.split import temporal_split
from src.eval.days_metrics import run_days_analysis, run_flat_threshold_comparison
from src.eval.flat_class_metrics import run_flat_analysis
from src.eval.next_day_metrics import compute_next_day_metrics, save_next_day_comparison
from src.eval.streak_metrics import compute_streak_metrics
from src.llm.category_prioritizer import CategoryPrioritizer
from src.llm.client import create_client
from src.llm.predict_graph import PredictGraphRunner
from src.finbert.next_day_predictor import FinBERTNextDayPredictor
from src.prices.align import PRODUCTS
from src.prices.next_day import compute_next_day_ground_truth, compute_next_day_returns
from src.prices.streaks import compute_day_streaks_for_news_df, compute_streaks_for_news_df
from src.rag.build import cmd_build_rag_from_config
from src.rag.retriever import HistoricalFactRetriever


def _load_news(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _prioritizer_cache_key(source: str, headline: str) -> str:
    return f"{source}|||{headline}|||prioritizer"


def _cache_key(source: str, headline: str, variant: int) -> str:
    return f"{source}|||{headline}|||v{variant}"


def _parse_variants(raw: str) -> list[int]:
    if raw == "all":
        return [1, 2, 3]
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def _row_to_prediction(
    row: pd.Series,
    pred: dict,
    variant: int,
    category: str | None = None,
    graph_state: dict | None = None,
) -> dict:
    out = {
        "date": row["date"].strftime("%Y-%m-%d"),
        "source": row["source"],
        "headline": row["headline"],
        "variant": variant,
        "pred_category": category or "",
        "reasoning": pred["reasoning"],
    }
    if graph_state is not None:
        totals = PredictGraphRunner.totals(graph_state)
        timings = graph_state.get("step_timings_ms", {})
        out["elapsed_ms"] = round(graph_state.get("total_elapsed_ms", 0.0), 1)
        out["timing_categorize_ms"] = round(timings.get("categorize", 0.0), 1)
        out["timing_predict_ms"] = round(
            timings.get("predict_next_day", timings.get("predict_streak", 0.0)), 1
        )
        out["prompt_tokens"] = totals["prompt_tokens"]
        out["completion_tokens"] = totals["completion_tokens"]
        out["total_tokens"] = totals["total_tokens"]
    for product in PRODUCTS:
        out[f"pred_dir_{product}"] = pred[f"dir_{product}"]
        out[f"pred_weeks_{product}"] = pred[f"weeks_{product}"]
        out[f"pred_days_{product}"] = pred[f"days_{product}"]
    return out


def _graph_state_to_trace(row: pd.Series, variant: int, state: dict) -> dict:
    totals = PredictGraphRunner.totals(state)
    return {
        "date": row["date"].strftime("%Y-%m-%d"),
        "source": row["source"],
        "headline": row["headline"],
        "variant": variant,
        "pred_category": state.get("category", ""),
        "category_reasoning": state.get("category_reasoning", ""),
        "actionable": state.get("actionable", False),
        "skipped": state.get("skipped", False),
        "skip_reason": state.get("skip_reason"),
        "elapsed_ms": state.get("total_elapsed_ms", 0.0),
        "step_timings_ms": state.get("step_timings_ms", {}),
        "token_usage": state.get("token_usage", {}),
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "total_tokens": totals["total_tokens"],
        "prediction": state.get("prediction"),
    }


def _save_cache(cache_path: Path, cache: dict[str, dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_retriever(config: dict) -> HistoricalFactRetriever:
    facts_path = resolve_path(config, config["rag"]["facts_path"])
    if not facts_path.exists():
        raise RuntimeError(
            f"RAG facts not found at {facts_path}. Run: python -m src.cli build-rag"
        )
    index_dir = resolve_path(config, config["rag"]["index_dir"])
    retriever = HistoricalFactRetriever(
        facts_path=facts_path,
        index_dir=index_dir,
        embedding_model=config["rag"]["embedding_model"],
        top_k=config["rag"]["top_k"],
        retrieve_candidates=config["rag"].get("retrieve_candidates", 20),
    )
    retriever.initialize()
    return retriever


def _build_prioritizer(config: dict) -> CategoryPrioritizer:
    client, model, cfg = create_client()
    cfg["project_root"] = config["project_root"]
    return CategoryPrioritizer(client, model, cfg)


def _build_finbert_predictor(config: dict, news_df: pd.DataFrame) -> FinBERTNextDayPredictor:
    cfg = dict(config)
    cfg["project_root"] = config["project_root"]
    retriever = _build_retriever(cfg)
    prices_path = str(PROJECT_ROOT / "new_prices.csv")
    return FinBERTNextDayPredictor(cfg, retriever, news_df, prices_path)


def _select_news_for_split(news_df: pd.DataFrame, split: str, train_ratio: float) -> pd.DataFrame:
    train, test = temporal_split(news_df, train_ratio)
    if split == "train":
        return train
    if split == "test":
        return test
    return news_df


def _ground_truth_days_df(
    news_df: pd.DataFrame,
    config: dict,
    prices_path: str,
    flat_threshold: float | None = None,
) -> pd.DataFrame:
    streak_cfg = config.get("streak", {})
    threshold = (
        flat_threshold
        if flat_threshold is not None
        else streak_cfg.get("flat_threshold", 0.01)
    )
    computed = compute_day_streaks_for_news_df(
        news_df,
        prices_path,
        max_days=streak_cfg.get("max_days", 4),
        flat_threshold=threshold,
    )
    rows: list[dict] = []
    for _, row in computed.iterrows():
        item = {
            "date": pd.Timestamp(row["date"]),
            "source": row["source"],
            "headline": row["headline"],
        }
        for product in PRODUCTS:
            s = row["streaks"][product]
            item[f"true_dir_{product}"] = s["direction"]
            item[f"true_days_{product}"] = s["days"]
            item[f"true_weeks_{product}"] = 0
        rows.append(item)
    return pd.DataFrame(rows)


def _ground_truth_df(news_df: pd.DataFrame, config: dict, prices_path: str) -> pd.DataFrame:
    streak_cfg = config.get("streak", {})
    computed = compute_streaks_for_news_df(
        news_df,
        prices_path,
        max_weeks=streak_cfg.get("max_weeks", 7),
        flat_threshold=streak_cfg.get("flat_threshold", 0.01),
    )
    rows: list[dict] = []
    for _, row in computed.iterrows():
        item = {
            "date": pd.Timestamp(row["date"]),
            "source": row["source"],
            "headline": row["headline"],
        }
        for product in PRODUCTS:
            s = row["streaks"][product]
            item[f"true_dir_{product}"] = s["direction"]
            item[f"true_weeks_{product}"] = s["weeks"]
            item[f"true_days_{product}"] = s["days"]
        rows.append(item)
    return pd.DataFrame(rows)


def _save_variant_predictions(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant in sorted(df["variant"].unique()):
        sub = df[df["variant"] == variant]
        path = output_dir / f"predictions_v{int(variant)}.csv"
        sub.to_csv(path, index=False, encoding="utf-8-sig")


def _save_variant_comparison(report: dict, output_path: Path) -> None:
    rows: list[dict] = []
    for variant, vrep in report.get("by_variant", {}).items():
        rows.append(
            {
                "variant": int(variant),
                "n": vrep["n"],
                "direction_accuracy_mean": vrep["direction_accuracy_mean"],
                "weeks_mae_mean": vrep["weeks_mae_mean"],
                **{
                    f"{product}_dir_acc": vrep["products"][product]["direction_accuracy"]
                    for product in PRODUCTS
                },
                **{
                    f"{product}_weeks_mae": vrep["products"][product]["weeks_mae"]
                    for product in PRODUCTS
                },
            }
        )
    comparison = pd.DataFrame(rows).sort_values("variant")
    comparison.to_csv(output_path, index=False, encoding="utf-8-sig")


def cmd_build_rag(args: argparse.Namespace) -> None:
    cmd_build_rag_from_config(
        config_path=args.config,
        no_cache=args.no_cache,
        quiet=args.quiet,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    news_df = _load_news(Path(args.input))
    full_news = news_df.copy()
    news_df = _select_news_for_split(news_df, args.split, config["eval"]["train_ratio"])
    if args.limit:
        news_df = news_df.head(args.limit)

    variants = _parse_variants(args.variant)
    max_workers = args.max_workers or config["llm"].get("max_workers", 5)
    prioritizer = _build_prioritizer(config)
    predictor = _build_finbert_predictor(config, full_news)

    cache_path = PROJECT_ROOT / "outputs" / "streak_prediction_cache.json"
    cache: dict[str, dict] = {}
    if cache_path.exists() and not args.no_cache:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    runners = {
        variant: PredictGraphRunner(
            prioritizer,
            predictor,
            variant,
            cache=cache,
            no_cache=args.no_cache,
            prioritizer_key=_prioritizer_cache_key,
            prediction_key=_cache_key,
        )
        for variant in variants
    }

    jobs: list[tuple[pd.Series, int]] = []
    for _, row in news_df.iterrows():
        for variant in variants:
            jobs.append((row, variant))

    if jobs and not args.quiet:
        print(
            f"Split={args.split}  variants={variants}  langgraph jobs={len(jobs)}, "
            f"workers={max_workers}"
        )

    rows: list[dict] = []
    traces: list[dict] = []
    skipped = 0

    def _run_job(job: tuple[pd.Series, int]) -> tuple[dict | None, dict]:
        row, variant = job
        runner = runners[variant]
        state = runner.invoke(row["source"], row["headline"], pd.Timestamp(row["date"]))
        trace = _graph_state_to_trace(row, variant, state)
        if state.get("skipped") or state.get("prediction") is None:
            return None, trace
        pred = state["prediction"]
        out = _row_to_prediction(
            row,
            pred,
            variant,
            category=state.get("category"),
            graph_state=state,
        )
        return out, trace

    iterator = jobs
    if jobs and not args.quiet:
        iterator = tqdm(jobs, desc="LangGraph predict", unit="job")

    if max_workers <= 1:
        for job in iterator:
            out, trace = _run_job(job)
            traces.append(trace)
            if out is None:
                skipped += 1
            else:
                rows.append(out)
            if not args.no_cache:
                _save_cache(cache_path, cache)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_run_job, job): job for job in jobs}
            progress = as_completed(future_map)
            if not args.quiet:
                progress = tqdm(progress, total=len(jobs), desc="LangGraph predict", unit="job")
            for future in progress:
                out, trace = future.result()
                traces.append(trace)
                if out is None:
                    skipped += 1
                else:
                    rows.append(out)
        if not args.no_cache:
            _save_cache(cache_path, cache)

    prior_rows = [
        {
            "date": t["date"],
            "source": t["source"],
            "headline": t["headline"],
            "pred_category": t["pred_category"],
            "category_reasoning": t["category_reasoning"],
            "actionable": t["actionable"],
        }
        for t in traces
        if t["variant"] == variants[0]
    ]
    prior_path = Path(args.output).parent / "prioritizer_results.csv"
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prior_rows).drop_duplicates(
        subset=["date", "source", "headline"]
    ).to_csv(prior_path, index=False, encoding="utf-8-sig")

    trace_path = Path(args.output).parent / "graph_traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as fh:
        for trace in traces:
            fh.write(json.dumps(trace, ensure_ascii=False) + "\n")

    rows.sort(key=lambda r: (r["date"], r["variant"]))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    _save_variant_predictions(df, out_path.parent)

    if traces and not args.quiet:
        sample = traces[0]
        print(f"GraphState sample: {PredictGraphRunner.format_summary(sample)}")

    print(
        f"Saved {len(rows)} predictions to {out_path} "
        f"({skipped} skipped as Прочее across variants)"
    )
    print(f"Graph traces saved to {trace_path}")
    print(f"Categories saved to {prior_path}")
    for variant in sorted(df["variant"].unique()) if len(df) else []:
        n = len(df[df["variant"] == variant])
        print(f"  v{int(variant)}: {n} rows -> {out_path.parent / f'predictions_v{int(variant)}.csv'}")


def cmd_eval(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    predictions = pd.read_csv(args.predictions, parse_dates=["date"])
    news_df = _load_news(Path(args.news))

    if args.split == "test":
        _, scope = temporal_split(news_df, config["eval"]["train_ratio"])
    elif args.split == "train":
        scope, _ = temporal_split(news_df, config["eval"]["train_ratio"])
    else:
        scope = news_df

    scope_keys = scope[["date", "source", "headline"]]
    predictions = predictions.merge(scope_keys, on=["date", "source", "headline"], how="inner")

    ground_truth = _ground_truth_df(scope, config, args.prices)
    merged = predictions.merge(
        ground_truth,
        on=["date", "source", "headline"],
        how="inner",
    )

    eval_split = args.split if args.split != "all" else "all"
    report = compute_streak_metrics(merged, ground_truth)
    results = {eval_split: report}

    streak_cfg = config.get("streak", {})
    next_day_truth = compute_next_day_ground_truth(
        scope,
        args.prices,
        flat_threshold=streak_cfg.get("flat_threshold", 0.01),
    )
    next_day_report = compute_next_day_metrics(predictions, next_day_truth)
    next_day_results = {eval_split: next_day_report}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_path = out_path.parent / "variant_comparison.csv"
    _save_variant_comparison(report, comparison_path)

    next_day_path = out_path.parent / "next_day_metrics.json"
    next_day_path.write_text(
        json.dumps(next_day_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    next_day_comparison_path = out_path.parent / "next_day_comparison.csv"
    save_next_day_comparison(next_day_report, next_day_comparison_path)

    print(f"Split={eval_split}  n={report['n']}  (streak ground truth)")
    for variant, vrep in report.get("by_variant", {}).items():
        print(
            f"  v{variant} streak: dir_acc={vrep['direction_accuracy_mean']:.3f}  "
            f"weeks_mae={vrep['weeks_mae_mean']:.2f}"
        )
    print(f"Metrics saved to {out_path}")
    print(f"Comparison saved to {comparison_path}")

    print(f"\nNext-day price move  n={next_day_report['n']}")
    for variant, vrep in next_day_report.get("by_variant", {}).items():
        u = vrep["products"]["urea"]["direction_accuracy"]
        d = vrep["products"]["dap"]["direction_accuracy"]
        m = vrep["products"]["mop"]["direction_accuracy"]
        print(
            f"  v{variant}: urea={u:.3f}  dap={d:.3f}  mop={m:.3f}  "
            f"mean={vrep['direction_accuracy_mean']:.3f}"
        )
    print(f"Next-day metrics saved to {next_day_path}")
    print(f"Next-day comparison saved to {next_day_comparison_path}")


def cmd_eval_days(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    predictions = pd.read_csv(args.predictions, parse_dates=["date"])
    news_df = _load_news(Path(args.news))

    if args.split == "test":
        _, scope = temporal_split(news_df, config["eval"]["train_ratio"])
    elif args.split == "train":
        scope, _ = temporal_split(news_df, config["eval"]["train_ratio"])
    else:
        scope = news_df

    scope_keys = scope[["date", "source", "headline"]]
    predictions = predictions.merge(scope_keys, on=["date", "source", "headline"], how="inner")

    variant = args.variant
    if "variant" in predictions.columns:
        predictions = predictions[predictions["variant"] == variant]

    streak_cfg = config.get("streak", {})
    flat_threshold = streak_cfg.get("flat_threshold", 0.01)
    ground_truth = _ground_truth_days_df(scope, config, args.prices, flat_threshold=flat_threshold)
    merged = predictions.merge(
        ground_truth,
        on=["date", "source", "headline"],
        how="inner",
    )

    exclude_categories = getattr(args, "exclude_category", None) or []
    if exclude_categories:
        category_map = news_df[["date", "source", "headline", "category"]]
        merged = merged.merge(category_map, on=["date", "source", "headline"], how="left")
        before = len(merged)
        merged = merged[~merged["category"].isin(exclude_categories)].drop(columns=["category"])
        print(
            f"Excluded categories {exclude_categories}: {before - len(merged)} rows "
            f"({len(merged)} remaining)"
        )

    eval_split = args.split if args.split != "all" else "all"
    result = run_days_analysis(
        merged, Path(args.output), variant=variant, flat_threshold=flat_threshold
    )
    report = result["report"]
    if exclude_categories:
        report["excluded_categories"] = exclude_categories

    filter_note = f"  exclude={exclude_categories}" if exclude_categories else ""
    print(
        f"Split={eval_split}  variant={variant}  n={report['n']}  "
        f"flat_threshold={flat_threshold * 100:.1f}%  (days ground truth){filter_note}"
    )
    vrep = report.get("by_variant", {}).get(str(variant), {})
    if vrep:
        print(
            f"  dir_acc={vrep['direction_accuracy_mean']:.3f}  "
            f"days_class_acc={vrep['days_class_accuracy_mean']:.3f}"
        )
        for product in PRODUCTS:
            p = vrep["products"][product]
            print(
                f"    {product}: dir={p['direction_accuracy']:.3f}  "
                f"days={p['days_class_accuracy']:.3f}"
            )
        print("\nDirection precision (flat / up / down) by true_days class:")
        prec_df = pd.read_csv(result["direction_precision_csv"])
        for product in PRODUCTS:
            print(f"  {product}:")
            sub = prec_df[prec_df["product"] == product]
            for _, row in sub.iterrows():
                def _fmt(val: float | None) -> str:
                    return f"{val:.1%}" if pd.notna(val) else "n/a"

                print(
                    f"    days={int(row['true_days'])} (n={int(row['n'])}): "
                    f"flat={_fmt(row['precision_flat'])} (pred n={int(row['n_pred_flat'])})  "
                    f"up={_fmt(row['precision_up'])} (pred n={int(row['n_pred_up'])})  "
                    f"down={_fmt(row['precision_down'])} (pred n={int(row['n_pred_down'])})"
                )
        print("\nDirection recall (flat / up / down) by true_days class:")
        rec_df = pd.read_csv(result["direction_precision_csv"])
        for product in PRODUCTS:
            print(f"  {product}:")
            sub = rec_df[rec_df["product"] == product]
            for _, row in sub.iterrows():
                def _fmt(val: float | None) -> str:
                    return f"{val:.1%}" if pd.notna(val) else "n/a"

                print(
                    f"    days={int(row['true_days'])} (n={int(row['n'])}): "
                    f"flat={_fmt(row['recall_flat'])} (true n={int(row['n_true_flat'])})  "
                    f"up={_fmt(row['recall_up'])} (true n={int(row['n_true_up'])})  "
                    f"down={_fmt(row['recall_down'])} (true n={int(row['n_true_down'])})"
                )
        print("\nDays count precision (class 0–4):")
        days_pr_df = pd.read_csv(result["days_count_precision_recall_csv"])
        for product in PRODUCTS:
            print(f"  {product}:")
            sub = days_pr_df[days_pr_df["product"] == product]
            for _, row in sub.iterrows():
                def _fmt(val: float | None) -> str:
                    return f"{val:.1%}" if pd.notna(val) else "n/a"

                print(
                    f"    class={int(row['day_class'])}: "
                    f"precision={_fmt(row['precision'])} (pred n={int(row['n_pred'])})  "
                    f"recall={_fmt(row['recall'])} (true n={int(row['n_true'])})"
                )
    print(f"Metrics saved to {result['metrics']}")
    print(f"Comparison saved to {result['comparison']}")
    print(f"Scatter plot saved to {result['plot']}")
    print(f"Class accuracy plot saved to {result['class_accuracy_plot']}")
    print(f"Direction accuracy plot saved to {result['direction_accuracy_plot']}")
    print(f"Direction precision plot saved to {result['direction_precision_plot']}")
    print(f"Direction recall plot saved to {result['direction_recall_plot']}")
    print(f"Days count precision plot saved to {result['days_count_precision_plot']}")
    print(f"Days count recall plot saved to {result['days_count_recall_plot']}")


def cmd_eval_days_compare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    predictions = pd.read_csv(args.predictions, parse_dates=["date"])
    news_df = _load_news(Path(args.news))

    if args.split == "test":
        _, scope = temporal_split(news_df, config["eval"]["train_ratio"])
    elif args.split == "train":
        scope, _ = temporal_split(news_df, config["eval"]["train_ratio"])
    else:
        scope = news_df

    scope_keys = scope[["date", "source", "headline"]]
    predictions = predictions.merge(scope_keys, on=["date", "source", "headline"], how="inner")

    variant = args.variant
    if "variant" in predictions.columns:
        predictions = predictions[predictions["variant"] == variant]

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    streak_cfg = config.get("streak", {})
    max_days = streak_cfg.get("max_days", 4)

    result = run_flat_threshold_comparison(
        predictions=predictions,
        scope=scope,
        prices_path=args.prices,
        output_dir=Path(args.output),
        thresholds=thresholds,
        variant=variant,
        max_days=max_days,
    )

    eval_split = args.split if args.split != "all" else "all"
    print(f"Split={eval_split}  variant={variant}  compare flat thresholds: {[t*100 for t in thresholds]}%")
    print("\nSummary:")
    print(result["summary_df"].to_string(index=False))

    print("\nDirection precision by true_days (flat / up / down):")
    df = result["comparison_df"]
    for th_pct in sorted(df["flat_threshold_pct"].unique()):
        print(f"\n  === flat threshold {th_pct:.0f}% ===")
        sub = df[np.isclose(df["flat_threshold_pct"], th_pct)]
        for product in PRODUCTS:
            print(f"  {product}:")
            ps = sub[sub["product"] == product]
            for _, row in ps.iterrows():
                def _fmt(val: float | None) -> str:
                    return f"{val:.1%}" if pd.notna(val) else "n/a"

                print(
                    f"    days={int(row['true_days'])} (n={int(row['n'])}): "
                    f"flat={_fmt(row['precision_flat'])}  "
                    f"up={_fmt(row['precision_up'])}  "
                    f"down={_fmt(row['precision_down'])}  "
                    f"| dir_acc={_fmt(row['direction_accuracy'])}  "
                    f"days_acc={_fmt(row['days_class_accuracy'])}"
                )

    print(f"\nComparison CSV: {result['comparison_csv']}")
    print(f"Summary CSV: {result['summary_csv']}")
    print(f"Precision comparison plot: {result['comparison_plot']}")
    print(f"Recall comparison plot: {result['comparison_recall_plot']}")


def cmd_eval_flat(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    predictions = pd.read_csv(args.predictions, parse_dates=["date"])
    news_df = _load_news(Path(args.news))

    if args.split == "test":
        _, scope = temporal_split(news_df, config["eval"]["train_ratio"])
    elif args.split == "train":
        scope, _ = temporal_split(news_df, config["eval"]["train_ratio"])
    else:
        scope = news_df

    scope_keys = scope[["date", "source", "headline"]]
    predictions = predictions.merge(scope_keys, on=["date", "source", "headline"], how="inner")

    returns_df = compute_next_day_returns(scope, args.prices)
    streak_cfg = config.get("streak", {})
    current_threshold = streak_cfg.get("flat_threshold", 0.01)

    result = run_flat_analysis(
        predictions=predictions,
        returns_df=returns_df,
        output_dir=Path(args.output),
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_steps=args.threshold_steps,
        current_threshold=current_threshold,
    )

    print(f"Flat sweep saved to {result['sweep_csv']}")
    print(f"Plot saved to {result['plot']}")
    print(f"At threshold={current_threshold*100:.1f}% -> {result['at_threshold_csv']}")
    print(f"\nPrecision / Recall flat at {current_threshold*100:.1f}%:")
    for _, row in result["at_current"].iterrows():
        p = row["precision_flat"]
        r = row["recall_flat"]
        p_str = f"{p:.3f}" if pd.notna(p) else "n/a"
        r_str = f"{r:.3f}" if pd.notna(r) else "n/a"
        print(
            f"  v{int(row['variant'])} {row['product']}: "
            f"precision={p_str}  recall={r_str}  "
            f"(true_flat={int(row['n_true_flat'])}, pred_flat={int(row['n_pred_flat'])})"
        )


def cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    facts_path = resolve_path(config, config["rag"]["facts_path"])
    if not facts_path.exists() or args.rebuild_rag:
        build_args = argparse.Namespace(
            config=args.config,
            no_cache=args.no_cache,
            quiet=args.quiet,
        )
        cmd_build_rag(build_args)

    pred_out = args.output or str(PROJECT_ROOT / "outputs" / "predictions.csv")
    metrics_out = args.metrics or str(PROJECT_ROOT / "outputs" / "metrics.json")

    predict_args = argparse.Namespace(
        input=args.news,
        output=pred_out,
        config=args.config,
        split="test",
        variant=args.variant,
        limit=args.limit,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        no_cache=args.no_cache,
        quiet=args.quiet,
    )
    cmd_predict(predict_args)

    eval_args = argparse.Namespace(
        predictions=pred_out,
        news=args.news,
        prices=args.prices,
        output=metrics_out,
        config=args.config,
        split="test",
    )
    cmd_eval(eval_args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streak RAG + LLM predictor for fertilizer price news"
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))

    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-rag", help="Build streak facts corpus from train 70%")
    p_build.add_argument("--no-cache", action="store_true")
    p_build.add_argument("--quiet", action="store_true")
    p_build.set_defaults(func=cmd_build_rag)

    p_predict = sub.add_parser("predict", help="Predict streaks on news CSV via LLM")
    p_predict.add_argument("--input", default=str(PROJECT_ROOT / "market_news.csv"))
    p_predict.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "predictions.csv"))
    p_predict.add_argument("--split", choices=["all", "train", "test"], default="test")
    p_predict.add_argument(
        "--variant",
        default="all",
        help="1, 2, 3, comma-separated, or all (default: all)",
    )
    p_predict.add_argument("--limit", type=int, default=None)
    p_predict.add_argument("--batch-size", type=int, default=None)
    p_predict.add_argument("--max-workers", type=int, default=None)
    p_predict.add_argument("--no-cache", action="store_true")
    p_predict.add_argument("--quiet", action="store_true")
    p_predict.set_defaults(func=cmd_predict)

    p_eval = sub.add_parser("eval", help="Evaluate streak predictions against prices")
    p_eval.add_argument("--news", default=str(PROJECT_ROOT / "market_news.csv"))
    p_eval.add_argument("--predictions", default=str(PROJECT_ROOT / "outputs" / "predictions.csv"))
    p_eval.add_argument("--prices", default=str(PROJECT_ROOT / "new_prices.csv"))
    p_eval.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "metrics.json"))
    p_eval.add_argument("--split", choices=["all", "train", "test"], default="test")
    p_eval.set_defaults(func=cmd_eval)

    p_eval_days = sub.add_parser("eval-days", help="Days class accuracy + scatter plot (v3)")
    p_eval_days.add_argument("--news", default=str(PROJECT_ROOT / "market_news.csv"))
    p_eval_days.add_argument(
        "--predictions",
        default=str(PROJECT_ROOT / "outputs" / "predictions_v3.csv"),
    )
    p_eval_days.add_argument("--prices", default=str(PROJECT_ROOT / "new_prices.csv"))
    p_eval_days.add_argument("--output", default=str(PROJECT_ROOT / "outputs"))
    p_eval_days.add_argument("--split", choices=["all", "train", "test"], default="test")
    p_eval_days.add_argument("--variant", type=int, default=3)
    p_eval_days.add_argument(
        "--exclude-category",
        action="append",
        default=[],
        help="Exclude news with this category (repeatable, e.g. --exclude-category Прочее)",
    )
    p_eval_days.set_defaults(func=cmd_eval_days)

    p_eval_days_cmp = sub.add_parser(
        "eval-days-compare",
        help="Compare days metrics at different flat thresholds (e.g. 1% vs 2%)",
    )
    p_eval_days_cmp.add_argument("--news", default=str(PROJECT_ROOT / "market_news.csv"))
    p_eval_days_cmp.add_argument(
        "--predictions",
        default=str(PROJECT_ROOT / "outputs" / "predictions_v3.csv"),
    )
    p_eval_days_cmp.add_argument("--prices", default=str(PROJECT_ROOT / "new_prices.csv"))
    p_eval_days_cmp.add_argument("--output", default=str(PROJECT_ROOT / "outputs"))
    p_eval_days_cmp.add_argument("--split", choices=["all", "train", "test"], default="test")
    p_eval_days_cmp.add_argument("--variant", type=int, default=3)
    p_eval_days_cmp.add_argument(
        "--thresholds",
        default="0.01,0.02",
        help="Comma-separated flat thresholds as fractions (default: 0.01,0.02 = 1%%, 2%%)",
    )
    p_eval_days_cmp.set_defaults(func=cmd_eval_days_compare)

    p_eval_flat = sub.add_parser("eval-flat", help="Precision/recall flat vs flat threshold")
    p_eval_flat.add_argument("--news", default=str(PROJECT_ROOT / "market_news.csv"))
    p_eval_flat.add_argument("--predictions", default=str(PROJECT_ROOT / "outputs" / "predictions.csv"))
    p_eval_flat.add_argument("--prices", default=str(PROJECT_ROOT / "new_prices.csv"))
    p_eval_flat.add_argument("--output", default=str(PROJECT_ROOT / "outputs"))
    p_eval_flat.add_argument("--split", choices=["all", "train", "test"], default="test")
    p_eval_flat.add_argument("--threshold-min", type=float, default=0.001)
    p_eval_flat.add_argument("--threshold-max", type=float, default=0.05)
    p_eval_flat.add_argument("--threshold-steps", type=int, default=40)
    p_eval_flat.set_defaults(func=cmd_eval_flat)

    p_run = sub.add_parser("run", help="build-rag + predict test + eval test")
    p_run.add_argument("--news", default=str(PROJECT_ROOT / "market_news.csv"))
    p_run.add_argument("--prices", default=str(PROJECT_ROOT / "new_prices.csv"))
    p_run.add_argument("--output", default=None)
    p_run.add_argument("--metrics", default=None)
    p_run.add_argument("--variant", default="all")
    p_run.add_argument("--rebuild-rag", action="store_true")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--batch-size", type=int, default=None)
    p_run.add_argument("--max-workers", type=int, default=None)
    p_run.add_argument("--no-cache", action="store_true")
    p_run.add_argument("--quiet", action="store_true")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
