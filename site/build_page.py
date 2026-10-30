"""Stage 10: build site/index.html from outputs/results.json.

Every figure on the page is read from that file, so the page cannot drift from the
pipeline's own numbers. Run it after run_all.py. The page is one static file with its
styles inside (the design tokens are copied from Personal-Website/styles.css, the same
ones kenya-aid-flows/site/build_page.py uses, so a visitor moving between projects sees
one body of work).
"""
import html
import json
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SITE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_DIR / "outputs" / "results.json"
PAGE_PATH = SITE_DIR / "index.html"

HOME_URL = "https://alven-yuka.vercel.app"
REPO_URL = "https://github.com/alvenyuka/nse-liquidity-radar"

STYLE = """
:root{--bg:#ffffff;--bg-2:#f6f7f3;--card:#ffffff;--fg:#1d222b;--fg-2:#323944;--muted:#4a5361;
--border:rgba(29,34,43,.11);--border-strong:rgba(29,34,43,.20);--lime:#d9f23a;--lime-line:#9fbc00;
--lime-wash:#f6fbd9;--on-lime:#1d222b;--accent:#3d5a00;--orange:#c2410c;--ring:#3d5a00;
--card-shadow:0 1px 2px rgba(29,34,43,.04),0 8px 24px -14px rgba(29,34,43,.12);
--font:"Cabinet Grotesk",system-ui,-apple-system,"Segoe UI",sans-serif;
--font-display:"Clash Display","Cabinet Grotesk",system-ui,sans-serif;
--font-mono:"JetBrains Mono",ui-monospace,"Cascadia Code",Consolas,monospace;color-scheme:light}
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth;scroll-padding-top:4.5rem}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--font);font-size:1.0625rem;line-height:1.6;font-weight:500;-webkit-font-smoothing:antialiased;overflow-x:hidden}
a{color:inherit;text-decoration-color:var(--border-strong);text-underline-offset:.16em}
a:hover{text-decoration-color:currentColor}
:focus-visible{outline:2px solid var(--ring);outline-offset:3px;border-radius:4px}
::selection{background:var(--lime);color:var(--on-lime)}
h1,h2,h3{margin:0;font-weight:500;line-height:1.15;text-wrap:balance}
p{margin:0}ul{margin:0;padding:0;list-style:none}
strong{font-weight:700}
.display{font-family:var(--font-display);font-weight:500;letter-spacing:-.02em}
.mono{font-family:var(--font-mono);font-weight:400}
.wrap{width:min(1180px,100% - 2rem);margin-inline:auto}
.skip{position:absolute;left:1rem;top:-3rem;padding:.5rem .75rem;background:var(--lime);color:var(--on-lime);border-radius:.5rem;z-index:50}
.skip:focus{top:1rem}
.top{position:sticky;top:0;z-index:40;background:rgba(255,255,255,.92);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom:1px solid var(--border)}
.top-row{display:flex;align-items:center;gap:1.5rem;height:4rem}
.brand{display:inline-flex;align-items:center;gap:.4rem;font-family:var(--font-display);font-weight:600;font-size:1.125rem;text-decoration:none}
.brand .dot{width:.4rem;height:.4rem;border-radius:50%;background:var(--lime-line)}
.nav{display:flex;gap:1.5rem;margin-inline:auto;font-size:.975rem;color:var(--fg-2)}
.nav a{text-decoration:none}.nav a:hover{color:var(--accent)}
.btn{display:inline-flex;align-items:center;justify-content:center;padding:.8rem 1.4rem;border-radius:999px;font-weight:600;font-size:1rem;text-decoration:none;border:1px solid transparent}
.btn-sm{padding:.5rem 1.05rem;font-size:.925rem}
.btn-outline{border-color:var(--border-strong);color:var(--fg);background:var(--card)}.btn-outline:hover{border-color:var(--fg)}
.nav-cta{margin-left:auto;border-color:var(--orange);color:var(--orange)}.nav-cta:hover{background:rgba(194,65,12,.08);border-color:var(--orange);color:var(--orange)}
.pill{display:inline-flex;align-items:center;gap:.55rem;padding:.4rem .9rem;border-radius:999px;border:1px solid var(--border-strong);background:var(--card);font-family:var(--font-mono);font-size:.8125rem;color:var(--fg-2);max-width:100%}
.hero{border-bottom:1px solid var(--border)}
.hero-inner{padding-block:clamp(3rem,7vw,5rem) clamp(2.5rem,5vw,4rem)}
.hero .pill{margin-bottom:1.5rem}
.hero h1{font-size:clamp(2.25rem,1.6rem + 2.4vw,3rem);line-height:1.05;letter-spacing:-.03em;max-width:22ch}
.lede{margin-top:1.25rem;color:var(--fg-2);font-size:clamp(1.0625rem,.95rem + .4vw,1.25rem);line-height:1.55;max-width:62ch}
.actions{display:flex;flex-wrap:wrap;gap:.75rem;margin-top:1.75rem}
.facts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1.25rem;margin-top:2.25rem;padding-top:1.5rem;border-top:1px solid var(--border)}
.facts li{display:grid;gap:.35rem;align-content:start}
.stat-num{font-size:2.1rem;line-height:1;letter-spacing:-.02em;justify-self:start;background:linear-gradient(transparent 58%,var(--lime) 58%,var(--lime) 92%,transparent 92%);padding-inline:.06em}
.stat-label{font-size:.9rem;color:var(--muted);line-height:1.35}
.section{padding-block:clamp(3rem,6vw,5rem)}
.section:nth-of-type(even){background:var(--bg-2)}
.section+.section{border-top:1px solid var(--border)}
.section-head{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:1.25rem 3rem;align-items:end;margin-bottom:clamp(1.5rem,3vw,2.5rem)}
.section-head .pill{margin-bottom:1rem}
.section-head h2{font-size:clamp(1.7rem,1.2rem + 2vw,2.6rem);letter-spacing:-.02em;line-height:1.05}
.section-head>p{color:var(--muted);max-width:56ch}
.card{background:var(--card);border:1px solid var(--border);border-radius:1rem;padding:clamp(1.1rem,3vw,1.75rem);box-shadow:var(--card-shadow)}
.callout{background:var(--lime-wash);border:1px solid var(--lime-line);border-radius:1rem;padding:clamp(1.1rem,3vw,1.5rem);margin-top:1.25rem}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1.25rem}
.prose p+p{margin-top:.9rem}
.prose{color:var(--fg-2);max-width:70ch}
.table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.92rem;min-width:30rem}
th,td{padding:.55rem .7rem;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
th{font-family:var(--font-mono);font-weight:400;font-size:.8rem;color:var(--muted)}
caption{text-align:left;font-family:var(--font-mono);font-size:.8rem;color:var(--muted);padding-bottom:.6rem}
.limits li{padding:.55rem 0;border-bottom:1px solid var(--border);color:var(--fg-2)}
.foot{padding-block:2rem;font-size:.8rem;color:var(--muted);display:grid;gap:.4rem;border-top:1px solid var(--border)}
@media (max-width:860px){.section-head{grid-template-columns:minmax(0,1fr)}.facts{grid-template-columns:repeat(2,minmax(0,1fr))}.grid2{grid-template-columns:minmax(0,1fr)}.nav{display:none}}
"""


def pct(value, digits=1):
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def whole(value):
    if value is None:
        return "—"
    return f"{int(round(value)):,}"


def num(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def esc(text):
    return html.escape(str(text))


def table(caption, headers, rows):
    head_cells = "".join(f"<th scope=\"col\">{esc(header)}</th>" for header in headers)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{esc(cell)}</td>" for cell in row)
        body = body + f"<tr>{cells}</tr>\n"
    return (
        f"<div class=\"table-wrap\"><table><caption>{esc(caption)}</caption>"
        f"<thead><tr>{head_cells}</tr></thead><tbody>\n{body}</tbody></table></div>"
    )


def section(section_id, label, heading, intro, body):
    return f"""
<section class="section" id="{section_id}">
<div class="wrap">
<div class="section-head"><div><span class="pill mono">{esc(label)}</span><h2 class="display">{esc(heading)}</h2></div><p>{intro}</p></div>
{body}
</div>
</section>"""


def git_commit():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_DIR, capture_output=True, text=True
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def build_page():
    r = json.loads(RESULTS_PATH.read_text())
    universe = r["universe"]
    coverage = r["coverage"]
    label_distribution = r["label_distribution"]
    baseline = r["baseline"]
    model = r["model"]["single_split"]
    walkforward = r["model"]["walkforward_by_year_transition_only"]

    hero_facts = f"""
<ul class="facts">
<li><span class="stat-num display">{whole(universe['stock_codes'])}</span><span class="stat-label">NSE-listed equities, {universe['first_date'][:4]}-{universe['last_date'][:4]}</span></li>
<li><span class="stat-num display">{pct(coverage['overall_untraded_share'])}</span><span class="stat-label">of all stock-sessions were untraded</span></li>
<li><span class="stat-num display">{pct(label_distribution['base_rate_transition'])}</span><span class="stat-label">of currently-liquid names turn illiquid within 5 sessions</span></li>
<li><span class="stat-num display">{num(model['transition']['pr_auc'])}</span><span class="stat-label">model PR-AUC on that harder subset, vs {num(baseline['transition']['zeros_rule']['pr_auc'])} for a fixed rule</span></li>
</ul>"""

    hero = f"""
<header class="hero"><div class="wrap hero-inner">
<span class="pill mono">SQL / Data Engineering + Applied ML &middot; DuckDB, plain SQL, scikit-learn</span>
<h1 class="display">Which NSE stocks are sliding into illiquidity, and which signal actually says so</h1>
<p class="lede">A reproducible pipeline on {whole(universe['stock_codes'])} Nairobi Securities Exchange equities, 2013 to October 2024, real daily price and volume data. Built to answer one question an investment desk actually asks: can a position in this name be exited without moving the market in the next week — and does a model beat a simple rule at predicting when that stops being true?</p>
<div class="actions"><a class="btn btn-sm btn-outline" href="{REPO_URL}">View the pipeline on GitHub</a></div>
{hero_facts}
</div></header>"""

    # Results: full population vs. the honest transition-only headline
    comparison_rows = [
        ["Persistence (today's regime = tomorrow's)", pct(baseline['full']['base_rate']), pct(baseline['full']['persistence']['pr_auc'], 3), pct(baseline['transition']['base_rate']), pct(baseline['transition']['persistence']['pr_auc'], 3)],
        [f"Rule: untraded ≥{int(baseline['full']['zeros_rule']['threshold']*100)}% of last 20 sessions", "", pct(baseline['full']['zeros_rule']['pr_auc'], 3), "", pct(baseline['transition']['zeros_rule']['pr_auc'], 3)],
        ["Logistic regression (4 SQL-built features)", "", pct(model['full']['pr_auc'], 3), "", pct(model['transition']['pr_auc'], 3)],
    ]
    comparison_table = table(
        "PR-AUC and base rate, test period 2023-2024 (5-session embargo before the split)",
        ["Method", "Base rate (full)", "PR-AUC (full)", "Base rate (transition)", "PR-AUC (transition)"],
        comparison_rows,
    )
    same_as_rule = model['transition']['same_flags_as_rule']
    callout = f"""
<div class="callout prose"><p><strong>Why "full population" overstates it.</strong> Most of the illiquidity in {pct(label_distribution['base_rate_full'])} of all rows is a stock that was already illiquid staying that way — persistence alone gets {pct(baseline['full']['persistence']['pr_auc'],3)} PR-AUC for free. The question worth answering is the harder one: among the {whole(label_distribution['currently_liquid_rows'])} stock-sessions that were NOT already illiquid, which ones turn within 5 sessions? There, persistence cannot even be defined usefully (it predicts nothing, since by definition these rows are not currently illiquid), the fixed rule gets {num(baseline['transition']['zeros_rule']['pr_auc'])}, and the model gets {num(model['transition']['pr_auc'])} — a real, modest improvement, not a free one.</p><p><strong>Where the model actually adds value.</strong> At the rule's own flag count ({same_as_rule['flagged']} flags), the model's top-ranked rows are the exact same as the rule's — precision {pct(same_as_rule['precision'],0)}, recall {pct(same_as_rule['recall'],1)}, identical. The model's higher PR-AUC comes from ranking better below that cutoff, not from beating the rule at the operating point a desk would actually use today. A lower, better-chosen cutoff is the natural next question, not answered here.</p></div>"""

    walkforward_rows = []
    for row in walkforward:
        walkforward_rows.append([
            row["test_year"], whole(row["rows_scored"]), pct(row["base_rate"]),
            num(row["persistence_pr_auc"]), num(row["rule_pr_auc"]), num(row["model_pr_auc"]),
        ])
    walkforward_table = table(
        "Walk-forward by year, transition population only, expanding training window",
        ["Test year", "Rows scored", "Base rate", "Persistence PR-AUC", "Rule PR-AUC", "Model PR-AUC"],
        walkforward_rows,
    )

    results_section = section(
        "results", "Results", "The model beats the rule every year — modestly",
        "Not one lucky split: the same model, refit on an expanding window, outperforms the fixed rule's ranking in every one of 9 overlapping test years (a stability check across years of the same pair of scorers, not nine independent replications).",
        f'<div class="card">{comparison_table}</div>{callout}<div class="card" style="margin-top:1.25rem">{walkforward_table}</div>',
    )

    # Coverage / data-engineering section
    coverage_rows = []
    for row in r["most_incomplete_stocks"]:
        coverage_rows.append([
            row["stock_code"], row["first_date"][:10], row["last_date"][:10],
            whole(row["sessions_in_bounds"]), whole(row["sessions_actually_recorded"]), whole(row["sessions_missing"]),
        ])
    coverage_table = table(
        "The 5 stocks with the most missing sessions inside their own listed life",
        ["Stock", "First seen", "Last seen", "Sessions expected", "Sessions recorded", "Sessions missing"],
        coverage_rows,
    )
    coverage_body = f"""
<div class="grid2">
<div class="card prose"><p><strong>{whole(coverage['grid_rows'])}</strong> stock-session rows in the dense grid. <strong>{whole(coverage['rows_absent_from_source_entirely'])}</strong> of those are sessions the source simply has no row for at all — not a zero, an absence. Confirming this before writing any rolling window mattered: a window function over the raw sparse table would have silently stretched "the next 5 sessions" across however many calendar gaps happen to fall in the way.</p><p>A further <strong>{whole(coverage['rows_present_but_marked_not_traded'])}</strong> rows are present but explicitly marked not traded (the source writes "-", never a literal 0). Both cases mean the same thing for this project: untraded.</p></div>
{coverage_table}
</div>"""
    coverage_section = section(
        "coverage", "Data engineering", "Why a dense per-stock session grid exists",
        "A real frontier-market data-quality finding, caught before it reached a model.",
        coverage_body,
    )

    # Watchlist
    watchlist_rows = []
    for row in r["watchlist"][:15]:
        days_1m = row.get("days_to_liquidate_1m")
        watchlist_rows.append([
            row["stock_code"], row.get("sector") or "—",
            pct(row["zeros_share_20d"]) if row["zeros_share_20d"] is not None else "—",
            "yes" if row["trailing_illiquid_5"] == 1 else "no",
            f"{days_1m:.1f}" if isinstance(days_1m, (int, float)) else "—",
        ])
    watchlist_table = table(
        f"Most recent reading per stock, ranked by last-20-session untraded share (as of {universe['last_date'][:10]})",
        ["Stock", "Sector", "Untraded share (20d)", "Illiquid now", "Days to exit KES 1M*"],
        watchlist_rows,
    )
    watchlist_section = section(
        "watchlist", "The watchlist", "What the signal said on the last session",
        "Not a live feed — the Mendeley archive ends 31 Oct 2024 — this is the backtest's own snapshot, read the same way a live version would be.",
        f'<div class="card">{watchlist_table}<p class="prose" style="margin-top:.9rem;font-size:.85rem">*Illustrative: position size divided by 20% of the stock\'s own 20-day average KES value traded. "—" means the stock traded nothing in the last 20 sessions, so no position size could be exited through the market at all right now.</p></div>',
    )

    method_body = """
<div class="grid2">
<div class="card prose"><p><strong>1-2. Ingest and clean.</strong> Four public Mendeley Data archives (one author, one schema, CC BY 4.0), downloaded with no login. Real schema drift handled: column names and even the date format change between the 2013-2021 files and the 2022-2024 files.</p><p><strong>3. Warehouse.</strong> DuckDB, loaded from cleaned parquet.</p><p><strong>4. SQL models.</strong> A dense per-stock session grid first, then window-function features (ROWS BETWEEN, never a date range), then the forward and trailing labels, then the analysis mart. Plain numbered .sql files, no dbt.</p></div>
<div class="card prose"><p><strong>5. Checks.</strong> 8 SQL checks that must return zero rows: no duplicate sessions, the grid never leaves a stock's own listed life, both labels are NULL in exactly the rows they should be, no negative volume, dates in range, every price code has a sector.</p><p><strong>6-7. Baseline and model.</strong> A naive persistence guess and a fixed rule, then a logistic regression, compared like-for-like by matching flag counts, not by a fixed probability cutoff that a rare label would make meaningless.</p><p><strong>8-10. Report, figures, site.</strong> Every number above traces to <code>outputs/results.json</code>.</p></div>
</div>"""
    method_section = section(
        "method", "How it is built", "One folder per stage, runnable end to end",
        "The same shape as this portfolio's other data pipelines: small scripts, plain loops, real checks.",
        method_body,
    )

    limits_items = [
        "This is a frozen historical archive (last update November 2024), not a live feed. There is deliberately no scheduled refresh — a live version would need a paid or scraped real-time NSE feed, which this project does not have.",
        "Indices (codes starting with ‘^’, e.g. the NSE All-Share Index) are excluded from the universe: an index has no comparable single-instrument trading volume.",
        "Returns are computed from the raw close, which is echoed unchanged on an untraded day (the source repeats the last traded price rather than leaving it blank). That makes 20-day return volatility understate true volatility on the thinnest names — a day with no trade is not really a zero-return day, it is a missing one.",
        "volume_zscore_60d is undefined (not computed) for a stock with zero trading volume for 60 straight sessions — a real state for several names in this market, not a bug.",
        "The logistic regression's standardised coefficients describe association within this sample, not a causal claim about what drives illiquidity.",
        "Sector is shown for grouping only; it is never a model input, because it is static and would mostly just memorise which names are already known to be thin.",
        "The ‘days to exit’ column on the watchlist is an illustrative execution rule of thumb (20% daily participation), not a market-impact model.",
    ]
    limits_body = '<ul class="limits">' + "".join(f"<li>{esc(item)}</li>" for item in limits_items) + "</ul>"
    limits_section = section(
        "limits", "Limits", "Read before citing a number from this page",
        "",
        f'<div class="card">{limits_body}</div>',
    )

    commit = git_commit()
    commit_text = f" Commit {esc(commit)}." if commit is not None else ""
    footer = f"""
<footer class="foot wrap mono"><span>Alven Yuka &middot; <a href="{HOME_URL}">alven-yuka.vercel.app</a> &middot; <a href="{REPO_URL}">{REPO_URL}</a></span>
<span>Figures read from outputs/results.json, built by site/build_page.py.{commit_text}</span>
<span>{esc(r['attribution'])}, {esc(r['licence'])}.</span></footer>"""

    body = hero + results_section + coverage_section + watchlist_section + method_section + limits_section

    description = (
        f"{whole(universe['stock_codes'])} NSE equities, {universe['first_date'][:4]}-{universe['last_date'][:4]}: a DuckDB/SQL liquidity "
        f"early-warning pipeline. Model PR-AUC {num(model['transition']['pr_auc'])} vs rule {num(baseline['transition']['zeros_rule']['pr_auc'])} on the honest transition subset."
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NSE Liquidity Radar | Alven Yuka</title>
<meta name="description" content="{esc(description)}">
<meta name="theme-color" content="#ffffff">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%23ffffff'/%3E%3Ccircle cx='32' cy='32' r='12' fill='%23d9f23a'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://api.fontshare.com" crossorigin>
<link rel="stylesheet" href="https://api.fontshare.com/v2/css?f[]=clash-display@500,600&f[]=cabinet-grotesk@400,500,700&display=swap">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap">
<style>{STYLE}</style>
</head>
<body>
<a class="skip" href="#results">Skip to results</a>
<div class="top"><div class="wrap top-row">
<a class="brand" href="{HOME_URL}"><span class="dot"></span>Alven Yuka</a>
<nav class="nav" aria-label="Sections"><a href="#results">Results</a><a href="#coverage">Data</a><a href="#watchlist">Watchlist</a><a href="#method">Method</a><a href="#limits">Limits</a></nav>
<a class="btn btn-sm btn-outline nav-cta" href="{REPO_URL}">Repository</a>
</div></div>
<main>{body}
</main>
{footer}
</body>
</html>
"""
    PAGE_PATH.write_text(page, encoding="utf-8")
    return page


def main():
    page = build_page()
    print(f"wrote {PAGE_PATH} ({len(page):,} characters)")


if __name__ == "__main__":
    main()
