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
        symbols = cfg.symbols if group == "cross_sector" else cfg.same_sector_symbols
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
