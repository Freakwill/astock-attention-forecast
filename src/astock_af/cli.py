"""Fire-based CLI: data fetch, tensor build, training, evaluation, LLM overlay."""

from __future__ import annotations

from pathlib import Path

import fire

from .config import load_config, load_data_config, load_feature_config
from .data import (
    available_sources,
    build_panel,
    coverage_report,
    load_panel,
    resolve_chain,
)
from .data.features import PanelTensors, build_tensors, tensor_summary


class CLI:
    """Command group for the A-share attention forecasting project."""

    def sources(self) -> None:
        """List registered data vendors and whether they are usable right now."""
        for name, reason in available_sources().items():
            print(f"  {name:9} {reason}")

    def fetch(
        self,
        data_config: str = "config/data.yaml",
        name: str = "panel",
        group: str = "cross_sector",
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        """Fetch the daily panel from the priority-chain sources and cache it.

        Args:
            data_config: Path to config/data.yaml.
            name: Cache name inside cache_dir (``panel`` by default).
            group: ``cross_sector`` or ``same_sector`` symbol group.
            start: Override the configured start date.
            end: Override the configured end date.
        """
        cfg = load_data_config(data_config)
        groups = {
            "cross_sector": cfg.symbols,
            "same_sector": cfg.same_sector_symbols,
            "etf": cfg.etf_symbols,
        }
        if group not in groups:
            raise ValueError(f"group must be one of {sorted(groups)}, got {group!r}")
        if not groups[group]:
            raise ValueError(f"group {group!r} has no symbols in {data_config}")
        symbols = groups[group]
        if cfg.benchmark is not None:
            symbols = [*symbols, cfg.benchmark]
        print(f"[fetch] group={group} adjust={cfg.adjust} align={cfg.align}")
        print(f"[fetch] symbols: {', '.join(s.ts_code for s in symbols)}")
        chain = resolve_chain(cfg.sources)
        try:
            panel, meta = build_panel(
                symbols,
                chain,
                start=start or cfg.start,
                end=end or cfg.end,
                adjust=cfg.adjust,
                align=cfg.align,
                name=name,
                cache_dir=cfg.cache_dir,
            )
        finally:
            for source in chain:
                source.close()
        print(f"[fetch] sources used: {meta.per_source}")
        print(coverage_report(panel).to_string(index=False))

    def coverage(self, cache_dir: str = "data/cache", name: str = "panel") -> None:
        """Show provenance plus per-symbol coverage / NaN counts for a cached panel."""
        panel, meta = load_panel(cache_dir, name)
        print(meta.to_json())
        print(coverage_report(panel).to_string(index=False))

    def tensors(
        self,
        data_config: str = "config/data.yaml",
        feature_config: str = "config/dataset.yaml",
        cache_dir: str = "data/cache",
        name: str = "panel",
        out: str = "data/cache/tensors.npz",
        section: str = "features",
    ) -> None:
        """Build the ``(time, assets, features)`` tensor from a cached panel."""
        panel, meta = load_panel(cache_dir, name)
        cfg = load_feature_config(feature_config, section=section)
        benchmark = None
        benchmark_code = next((code for code in meta.symbols if code.endswith("000300.SH")), None)
        if benchmark_code is not None:
            benchmark = panel[panel["ts_code"] == benchmark_code]
            panel = panel[panel["ts_code"] != benchmark_code]
        tensors = build_tensors(panel, cfg, benchmark=benchmark)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        tensors.save(out)
        print(tensor_summary(tensors))
        print(f"[tensors] saved -> {out}")

    def train(
        self,
        tensors_path: str = "data/cache/tensors.npz",
        config: str = "config/model.yaml",
        architecture: str = "transformer",
        epochs: int | None = None,
        device: str | None = None,
        seed: int | None = None,
        out_dir: str | None = None,
    ) -> None:
        """Train one architecture on cached tensors and write a run directory."""
        cfg = load_config(config)
        tensors = PanelTensors.load(tensors_path)
        print(f"[train] {architecture} on {tensors.x.shape} (assets x features), config={config}")
        train_architecture(
            tensors,
            cfg.model,
            cfg.train,
            architecture,
            out_dir=out_dir or cfg.train.out_dir,
            epochs=epochs,
            device=device,
            seed=seed,
        )

    def benchmark(
        self,
        tensors_path: str = "data/cache/tensors.npz",
        config: str = "config/model.yaml",
        architectures: str = "transformer,lstm,linear,var,naive:zero,naive:mean,naive:momentum",
        epochs: int | None = None,
        out_dir: str = "reports/runs",
        report: str = "reports/benchmark.md",
        seed: int | None = None,
    ) -> None:
        """Train every architecture with identical splits and tabulate metrics."""
        cfg = load_config(config)
        tensors = PanelTensors.load(tensors_path)
        runs = []
        for architecture in [item.strip() for item in architectures.split(",") if item.strip()]:
            outcome = train_architecture(
                tensors, cfg.model, cfg.train, architecture, out_dir=out_dir, epochs=epochs, seed=seed
            )
            runs.append(outcome["run_dir"])
        frame = benchmark_runs(runs)
        table = format_benchmark(frame)
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(f"# Benchmark\n\n{table}\n")
        print(table)
        for run_dir in runs:
            try:
                path = plot_run(run_dir)
                print(f"[plot] {path}")
            except Exception as exc:  # noqa: BLE001 - plotting must not kill the run
                print(f"[plot] skipped for {run_dir}: {type(exc).__name__}: {exc}")
        print(f"[benchmark] report -> {report}")

    def evaluate(self, run: str, tensors_path: str = "data/cache/tensors.npz", plot: bool = True) -> None:
        """Re-print a finished run's metrics, per-asset breakdown and attention profile."""
        metrics, arrays = load_run(run)
        table = metrics_table(arrays["test"], arrays["y_test"], arrays["names"], arrays["test"].shape[2])
        print(table.to_string(index=False))
        if plot:
            print(f"[plot] {plot_run(run)}")
        attention_path = Path(run) / "attention.npz"
        if attention_path.is_file():
            tensors = PanelTensors.load(tensors_path)
            print(attention_summary(run, tensors).to_string(index=False))

    def chart(
        self,
        cache_dir: str = "data/cache",
        name: str = "panel",
        tensors_path: str = "data/cache/tensors.npz",
        run: str | None = None,
        out: str = "reports/figures/price_panel.png",
        asset: int = 0,
    ) -> None:
        """Draw the README figure: indexed prices with the split layout, plus the model's output.

        Args:
            cache_dir: Where the cached panel lives.
            name: Panel cache name (``panel`` cross-sector, ``panel_same`` control group).
            tensors_path: Tensors used to mark the train/validation/test regions.
            run: Optional run directory whose out-of-sample forecast is overlaid.
            out: Destination PNG.
            asset: Asset index to show in the lower panel.
        """
        panel, _ = load_panel(cache_dir, name)
        tensors = PanelTensors.load(tensors_path)
        path = plot_price_panel(panel, tensors=tensors, run_dir=run, out=out, asset=asset)
        print(f"[chart] -> {path}")

    def forecast(
        self,
        run: str,
        tensors_path: str = "data/cache/tensors.npz",
        out: str = "reports/forecast.md",
    ) -> None:
        """Forecast the next H trading days from the last window in the panel.

        Args:
            run: A trained run directory (``reports/runs/transformer_<stamp>``).
            tensors_path: Tensors the model was trained on.
            out: Where to write the markdown report.
        """
        from .forecast import format_forecast, forward_forecast

        forecast = forward_forecast(run, tensors_path)
        markdown = format_forecast(forecast)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(markdown, encoding="utf-8")
        print(markdown)
        print(f"[forecast] -> {out}")

    def mi(
        self,
        codes: str = "159142,512760,513100",
        permutations: int = 200,
        neighbors: int = 3,
        out: str = "reports/mi_etf.md",
    ) -> None:
        """Estimate lag-1 mutual information between the ETFs and write the table.

        Answers "does yesterday tell you anything about today" with dependence
        rather than correlation, using a kNN estimator plus a shuffle null so the
        finite-sample bias is separated from the estimate.

        Args:
            codes: Comma-separated ETF codes.
            permutations: Shuffles used to build the null distribution.
            out: Where to write the markdown report.
        """
        import numpy as np

        from .mi import format_scores, lag1_scores
        from .strategy import load_history

        # fire splits `--codes=159142,512760` into a tuple; normalise it back.
        if isinstance(codes, (tuple, list)):
            codes = ",".join(str(item) for item in codes)
        symbols = [code.strip() for code in codes.split(",") if code.strip()]
        prices = load_history(symbols)
        returns = np.log(prices.astype(float)).diff()

        level = lag1_scores(returns, transform="level", n_permutations=permutations, n_neighbors=neighbors)
        magnitude = lag1_scores(returns, transform="abs", n_permutations=permutations, n_neighbors=neighbors)

        counts = returns.notna().sum().to_dict()
        lines = [
            "# Lag-1 mutual information: I(x_{t-1}; y_t)",
            "",
            f"Estimator: kNN (k={neighbors}) in nats, converted to bits. Null: {permutations} shuffles of the",
            "target series, which keeps the marginals and destroys the dependence - so `shuffle mean`",
            "is what the bias alone produces and `excess bits` is the estimate that matters.",
            "",
            f"Observations per series: {', '.join(f'{k}={v}' for k, v in counts.items())}. "
            "Pairs involving 159142 are limited by its 2025-12-12 listing.",
            "",
            f"Bonferroni threshold for 9 directed pairs: p < {0.05 / 9:.4f}.",
            "",
            format_scores(level, "Signed returns (direction)"),
            "",
            format_scores(magnitude, "Absolute returns (volatility clustering)"),
            "",
            "Reading notes:",
            "",
            "* `excess bits` is the number to trust; a raw MI that equals its shuffle mean is bias.",
            "* `Pearson` is there to show when MI and correlation agree - if `excess bits` is",
            "  positive while `Pearson` is ~0, the dependence is nonlinear (or the estimate is noise).",
            "* Volatility clustering is expected to be the *real* dependence: it is the one thing",
            "  daily equity data reliably has. Directional predictability is what the forecasting",
            "  model would need, and it is what this table tests.",
            "",
        ]
        markdown = "\n".join(lines)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(markdown + "\n", encoding="utf-8")
        print(markdown)
        print(f"[mi] -> {out}")

    def strategy(
        self,
        codes: str = "159142,512760,513100",
        target_vol: float = 0.15,
        cap: float = 0.5,
        out: str = "reports/etf_strategy.md",
    ) -> None:
        """Backtest trend / rotation / buy-and-hold and print the current target weights.

        Args:
            codes: Comma-separated ETF codes (bare or ``ts_code``).
            target_vol: Annualised volatility target used to size each sleeve.
            cap: Maximum weight per ETF (no leverage, the rest stays in cash).
            out: Where to write the markdown report.
        """
        from .strategy import (
            COST_PER_SIDE,
            buy_and_hold,
            current_view,
            load_history,
            rotation_strategy,
            trend_strategy,
            vol_target_strategy,
        )

        # fire splits `--codes=159142,512760` into a tuple; normalise it back.
        if isinstance(codes, (tuple, list)):
            codes = ",".join(str(item) for item in codes)
        symbols = [code.strip() for code in codes.split(",") if code.strip()]
        prices = load_history(symbols)
        sleeves = [
            buy_and_hold(prices),
            vol_target_strategy(prices, target_vol, cap),
            trend_strategy(prices, target_vol, cap),
            rotation_strategy(prices, target_vol=target_vol, cap=cap),
        ]
        view = current_view(prices, target_vol=target_vol, cap=cap)

        lines = [
            "# ETF strategy - trend, rotation, buy & hold",
            "",
            f"History: {prices.index[0].date()} .. {prices.index[-1].date()} "
            f"({len(prices)} trading days, {prices.shape[1]} ETFs).",
            f"Costs: {COST_PER_SIDE * 100:.2f}% per side on turnover. Long/cash only, no leverage.",
            f"Sizing: inverse volatility, target {target_vol:.0%} annualised, cap {cap:.0%} per ETF.",
            "",
            "## Backtest, net of costs",
            "",
            "| strategy | CAGR % | vol % | Sharpe | max DD % | Calmar | exposure % | turnover/yr |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for result in sleeves:
            s = result.stats
            lines.append(
                f"| {result.name} | {s['cagr_pct']} | {s['vol_pct']} | {s['sharpe']} | {s['max_dd_pct']} "
                f"| {s['calmar']} | {s['exposure_pct']} | {s['turnover_yr']} |"
            )

        lines += [
            "",
            "## Per-ETF trend sleeve vs buy & hold",
            "",
            "| ETF | trend CAGR % | trend max DD % | buy&hold CAGR % | buy&hold max DD % |",
            "|---|---|---|---|---|",
        ]
        for code in prices.columns:
            single = prices[[code]]
            trend = trend_strategy(single, target_vol, cap)
            hold = buy_and_hold(single)
            lines.append(
                f"| {code} | {trend.stats['cagr_pct']} | {trend.stats['max_dd_pct']} "
                f"| {hold.stats['cagr_pct']} | {hold.stats['max_dd_pct']} |"
            )

        lines += [
            "",
            "## The same idea at other MA lengths (sensitivity, not a menu to pick from)",
            "",
            "| fast | slow | CAGR % | Sharpe | max DD % | exposure % |",
            "|---|---|---|---|---|---|",
        ]
        for fast, slow in ((10, 30), (20, 60), (60, 120), (120, 200)):
            result = trend_strategy(prices, target_vol, cap, fast=fast, slow=slow)
            s = result.stats
            lines.append(
                f"| {fast} | {slow} | {s['cagr_pct']} | {s['sharpe']} | {s['max_dd_pct']} | {s['exposure_pct']} |"
            )

        lines += [
            "",
            f"## Current readings and target weights (as of {view['date'].iloc[0].date()}, for the next session)",
            "",
            "| asset | close | MA20 | MA60 | 20d mom % | 60d mom % | 20d vol % | dd from 60d high % | above MA20 | MA20>MA60 | core w | trend w | final w |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, row in view.iterrows():
            lines.append(
                f"| {row['asset']} | {row['close']:.3f} | {row['ma20']:.3f} | {row['ma60']:.3f} "
                f"| {row['mom20'] * 100:+.2f} | {row['mom60'] * 100:+.2f} | {row['vol20'] * 100:.1f} "
                f"| {row['dd60'] * 100:+.2f} | {bool(row['above_ma20'])} | {bool(row['ma20_above_ma60'])} "
                f"| {row['core_weight']:.1%} | {row['trend_weight']:.1%} | {row['final_weight']:.1%} |"
            )
        total = float(view["final_weight"].sum())
        lines += [
            "",
            f"Final allocation: invested {total:.1%}, cash {1 - total:.1%} (final = min(core, trend)).",
            "",
        ]

        markdown = "\n".join(lines)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(markdown + "\n", encoding="utf-8")
        print(markdown)
        print(f"[strategy] -> {out}")

    def llm(
        self,
        run: str | None = None,
        config: str = "config/llm.yaml",
        days: int = 7,
        symbols: str | None = None,
        out: str = "reports/llm_overlay.json",
    ) -> None:
        """Ask an LLM for a sentiment-driven view and score it against the model / realized returns."""
        from .llm.overlay import run_overlay

        # fire turns `--symbols=159142,512760` into a tuple, so normalise before
        # handing it to the overlay, which expects the comma-separated string.
        if isinstance(symbols, (tuple, list)):
            symbols = ",".join(str(item) for item in symbols)
        run_overlay(config_path=config, run_dir=run, days=days, symbols=symbols, out=out)


from .evaluate import (  # noqa: E402 - imported after CLI to keep the module import graph flat
    attention_summary,
    benchmark_runs,
    format_benchmark,
    load_run,
    metrics_table,
    plot_price_panel,
    plot_run,
)
from .train import train_architecture  # noqa: E402


def main() -> None:
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
