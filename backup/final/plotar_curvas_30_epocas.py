"""Gera duas curvas de validação do estudo de 30 épocas (sem dependências externas)."""

import csv
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "long_overfit"
DEST = Path(__file__).resolve().parent
COLORS = {3: "#2563eb", 6: "#d97706", 9: "#059669"}


def read(kind, task):
    with (SOURCE / kind / task / "comparison.csv").open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def chart(kind, normalized=False):
    width, height = 1120, 520
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
             '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:22px;font-weight:bold}.subtitle{font-size:13px;fill:#526075}.label{font-size:15px;font-weight:bold}.tick{font-size:12px;fill:#526075}.legend{font-size:13px}.grid{stroke:#dce2e9;stroke-width:1}.axis{stroke:#778494;stroke-width:1.3}</style>',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text class="title" x="50" y="39">{escape("Heads intermediários" if kind == "heads" else "Modelo base")} · validação até 30 épocas</text>',
             f'<text class="subtitle" x="50" y="61">Seed 42 · medições a cada 2 épocas · {"cada curva: (valor − mínimo) / (máximo − mínimo)" if normalized else "maior valor é melhor"}</text>']
    for index, task in enumerate(("rte", "sts")):
        x0 = 72 + index * 555
        y0, w, h = 105, 440, 315
        rows = read(kind, task)
        metric = "macro F1" if task == "rte" else "Pearson"
        field = "validation_primary" if kind == "base" else ("validation_macro_f1" if task == "rte" else "validation_pearson")
        series = [("Modelo base", "#7c3aed", rows)] if kind == "base" else [
            (f"Camada {layer}", COLORS[layer], [r for r in rows if int(r["layer"]) == layer])
            for layer in (3, 6, 9)]
        bounds = {label: (min(float(r[field]) for r in group), max(float(r[field]) for r in group))
                  for label, _, group in series}
        def value(row, label):
            raw = float(row[field])
            if not normalized:
                return raw
            minimum, maximum = bounds[label]
            return (raw - minimum) / (maximum - minimum) if maximum > minimum else 0.0
        values = [float(r[field]) for _, _, group in series for r in group]
        low = 0 if normalized else max(0, min(values) - (0.035 if kind == "base" else 0.06))
        high = 1 if normalized else min(1, max(values) + (0.02 if kind == "base" else 0.05))
        px = lambda epoch: x0 + (epoch - 2) / 28 * w
        py = lambda value: y0 + h - (value - low) / (high - low) * h
        parts.append(f'<text class="label" x="{x0}" y="91">{task.upper()} · {metric}</text>')
        for n in range(5):
            tick_value = low + (high - low) * n / 4
            y = py(tick_value)
            parts.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}"/>')
            parts.append(f'<text class="tick" x="{x0-9}" y="{y+4:.1f}" text-anchor="end">{tick_value:.2f}</text>')
        for epoch in (2, 6, 10, 14, 18, 22, 26, 30):
            x = px(epoch)
            parts.append(f'<text class="tick" x="{x:.1f}" y="{y0+h+20}" text-anchor="middle">{epoch}</text>')
        parts.append(f'<line class="axis" x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}"/>')
        for label, color, group in series:
            points = " ".join(f'{px(int(r["epoch"])):.1f},{py(value(r, label)):.1f}' for r in group)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.8" stroke-linejoin="round"/>')
            for r in group:
                parts.append(f'<circle cx="{px(int(r["epoch"])):.1f}" cy="{py(value(r, label)):.1f}" r="3.2" fill="{color}"/>')
            best = max(group, key=lambda r: float(r[field]))
            bx, by = px(int(best["epoch"])), py(value(best, label))
            parts.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="6" fill="white" stroke="{color}" stroke-width="2.5"/>')
            parts.append(f'<title>{escape(label)}: máximo {float(best[field]):.4f} na época {best["epoch"]}</title>')
        for j, (label, color, _) in enumerate(series):
            lx = x0 + j * 125
            parts.append(f'<line x1="{lx}" y1="470" x2="{lx+22}" y2="470" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text class="legend" x="{lx+28}" y="474">{escape(label)}</text>')
        parts.append(f'<text class="tick" x="{x0+w/2}" y="450" text-anchor="middle">Época</text>')
    parts.append('</svg>')
    output = DEST / f"curvas_30_epocas_{kind}{'_normalizadas' if normalized else ''}.svg"
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


if __name__ == "__main__":
    for kind in ("base", "heads"):
        for normalized in (False, True):
            print(chart(kind, normalized))
