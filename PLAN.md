# Engineering Spec: UI Upgrade & Discreet Diagnostics for `app.py`

**Project:** AI-Powered Customer Review Sentiment Analyzer (Streamlit + DistilBERT)
**Executor:** Antigravity (AI coding agent) — implement in one pass
**Scope:** UI/presentation changes in `app.py` only

---

## 0. Hard Constraints (read first)

1. **Modify exactly one file: `app.py`.** Do NOT touch `model.py`, `evaluate.py`, `requirements.txt`, `sample_data/`, or any inference logic.
2. Do NOT change how the model is called. The only inference calls in the app must remain exactly:
   - `analyzer.predict(review)` (Single Review tab)
   - `analyzer.predict(str(x)) if pd.notna(x) and str(x).strip() else {"sentiment":"EMPTY","confidence":0.0}` (Batch tab list comprehension)
   - `load_analyzer()` with `@st.cache_resource` and `SentimentAnalyzer()`
3. Keep intact: the quick-demo selector (`examples` dict + `chosen` selectbox), the review `text_area`, the "Analyze Sentiment" button, CSV upload, the `"review"` column validation, "Analyze all reviews", dataframe, metrics, matplotlib chart, both download buttons (`Download analyzed CSV`, `Download sample CSV`), the `ValueError` → `st.warning` handling, and the model-load `try/except` + `st.stop()`.
4. No new dependencies. Use only `streamlit`, `pandas`, `matplotlib`, and the stdlib (`html`) already available.
5. Result-dict keys consumed by the UI: `sentiment` (`"POSITIVE"`/`"NEGATIVE"`), `confidence`, `positive_score`, `negative_score` (all floats 0–1 except `sentiment`). Do not rename or add keys.

---

## 1. File to Modify

`Customer-Review-Sentiment-Analyzer/app.py` (currently 130 lines). Edit in place.

---

## 2. UI / Layout Changes

### Before
```
Hero banner
st.success("Model loaded…")
Tabs: [Single Review] [Batch Analysis] [Model Architecture] [Project & Viva]
Sidebar: Model information → Quick demo reviews
```

### After
```
Hero banner (refined)
Small "model loaded" status pill
Tabs: [Single Review] [Batch Analysis]            <-- ONLY these two
Sidebar (top → bottom):
  - Model information        (unchanged content)
  - Quick demo reviews       (unchanged widget + keys)
  - [very last] st.expander("System Diagnostics", expanded=False)
        └─ inner tabs: [Pipeline Specs] [Runtime Notes]
        (rendered only when URL has ?debug=true)
```

### Diagnostics visibility rule (implement exactly)
Define near the top of `app.py` (after imports):

```python
# Set True to always show the diagnostics expander (no URL flag needed).
ALWAYS_SHOW_DIAGNOSTICS = False
SHOW_DIAGNOSTICS = ALWAYS_SHOW_DIAGNOSTICS or str(st.query_params.get("debug", "")).lower() in ("true", "1")
```

- Normal URL (`http://localhost:8501/`) → no trace of diagnostics anywhere.
- Demo URL (`http://localhost:8501/?debug=true`) → "System Diagnostics" expander appears at the bottom of the sidebar, collapsed.
- `st.query_params` is available in the pinned `streamlit>=1.35.0`.

---

## 3. Component Mapping

| Existing element (current `app.py`) | Action | New location / form |
|---|---|---|
| `tab1` "🔍 Single Review" | Keep, restyle | Main tab 1 (`st.tabs(["🔍 Single Review", "📊 Batch Analysis"])`) |
| `tab2` "📊 Batch Analysis" | Keep, light restyle | Main tab 2 |
| `tab3` "🧠 Model Architecture" content (workflow code block, "Why DistilBERT?", "Core calculation", model-card link) | **Move + rename** | Sidebar → `System Diagnostics` → inner tab **"Pipeline Specs"** |
| `tab4` "📘 Project & Viva" content (problem, objectives, limitations, viva explanation) | **Move + rename** | Sidebar → `System Diagnostics` → inner tab **"Runtime Notes"** |
| `st.success("Model loaded successfully · Running on …")` | Replace presentation | Small status pill via `st.markdown` (same info: "Model loaded · Running on {analyzer.device}") |
| `c1/c2/c3` metrics + `st.progress(...)` in Single Review | Replace with card | `render_result_card()` helper (see §4 Step 6). Keep the `left/right` probability metrics or fold them into the card; keep the confidence caption |
| `st.markdown('<div class="hero">…')` | Keep, refine | Same markup, upgraded CSS |
| Sidebar "Model information" + "Quick demo reviews" | Keep unchanged | Unchanged; diagnostics expander appended after them |

**Naming rule:** the strings "Model Architecture", "Project & Viva", "Viva", "Architecture" must not appear in any UI-visible label. Use only "System Diagnostics", "Pipeline Specs", "Runtime Notes". Also change the sub-heading `st.subheader("Mini-project overview")` to `st.subheader("Runtime notes")` and the inline phrase `**Suggested viva explanation:**` to `**Pipeline walkthrough:**` so nothing visible gives away the viva context. (Content text otherwise unchanged.)

---

## 4. Implementation Steps (in order)

### Step 1 — Imports
Add `import html` next to the other stdlib imports. No other import changes.

### Step 2 — Replace the existing `<style>` block
Replace lines 11–16 (the current `st.markdown("""<style>…""")`) with the full CSS in §5.

### Step 3 — Diagnostics flag
Add the `ALWAYS_SHOW_DIAGNOSTICS` / `SHOW_DIAGNOSTICS` snippet from §2 right after `st.set_page_config(...)`.

### Step 4 — Hero + status pill
Keep the hero `st.markdown` call as is (class `hero`). Replace the `st.success(...)` line (currently line 48, after the model loads) with:

```python
st.markdown(f'<span class="status-pill"><span class="dot"></span>Model ready · running on {html.escape(str(analyzer.device))}</span>', unsafe_allow_html=True)
```
Keep the `try/except … st.stop()` model-loading block exactly as is.

### Step 5 — Tabs
Replace line 49 with:

```python
tab1, tab2 = st.tabs(["🔍 Single Review", "📊 Batch Analysis"])
```
Delete the `with tab3:` and `with tab4:` blocks from the main body (their content moves in Step 8). Delete nothing else.

### Step 6 — Result card helper (UI-only; place above the tabs)

```python
def render_result_card(result, word_count):
    label = str(result["sentiment"]).upper()
    is_pos = label == "POSITIVE"
    cls = "pos" if is_pos else "neg"
    icon = "😊" if is_pos else "😞"
    conf = float(result["confidence"]) * 100
    pos = float(result["positive_score"]) * 100
    neg = float(result["negative_score"]) * 100
    st.markdown(
        f'<div class="result-card {cls}">'
        f'<div class="rc-head"><span class="rc-icon">{icon}</span>'
        f'<div><div class="rc-eyebrow">Predicted sentiment</div>'
        f'<div class="rc-label">{html.escape(label.title())}</div></div>'
        f'<div class="rc-conf">{conf:.2f}<span>%</span><div class="rc-eyebrow">confidence</div></div></div>'
        f'<div class="conf-track"><div class="conf-fill" style="width:{conf:.1f}%"></div></div>'
        f'<div class="rc-grid">'
        f'<div class="rc-stat"><span>Positive probability</span><b>{pos:.2f}%</b></div>'
        f'<div class="rc-stat"><span>Negative probability</span><b>{neg:.2f}%</b></div>'
        f'<div class="rc-stat"><span>Input length</span><b>{word_count} words</b></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
```

**Gotcha:** build the HTML as a single string with **no blank lines and no leading 4-space indentation inside the string**, otherwise Streamlit's Markdown parser renders it as a code block.

Inside Single Review, keep everything up to and including `result = analyzer.predict(review)` and the `except ValueError` branch unchanged. Replace only the display lines (current `c1,c2,c3 … st.metric … st.progress … left,right …`) with:

```python
render_result_card(result, len(review.split()))
st.caption("Confidence is the model's softmax score, not a guarantee that the prediction is correct.")
```
Keep `st.subheader("Analyze one customer review")`, the `text_area`, and the button as they are.

### Step 7 — Batch Analysis polish (cosmetic only)
- Keep all logic identical. Do not wrap widgets in raw HTML `<div>`s (Streamlit can't nest widgets inside `st.markdown` HTML); style the file uploader, dataframe and metrics through the CSS in §5 only.
- Pass explicit colors to the matplotlib chart without changing data: `counts.reindex([...], fill_value=0).plot(kind="bar", ax=ax, color=["#16a34a", "#dc2626"], rot=0)`, remove top/right spines (`ax.spines[["top","right"]].set_visible(False)`), and use `fig.patch.set_alpha(0)` / `ax.set_facecolor("none")` so it sits cleanly on both light and dark themes. Axis label/title text color: leave default (acceptable) or set to `"#64748b"` for mid-tone legibility on both themes.
- Keep both `st.download_button` calls unchanged.

### Step 8 — Sidebar diagnostics (must be the LAST thing inside `with st.sidebar:`)
After the `chosen = st.selectbox(...)` line, still inside the `with st.sidebar:` block, append:

```python
    if SHOW_DIAGNOSTICS:
        st.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)
        with st.expander("System Diagnostics", expanded=False):
            d_specs, d_notes = st.tabs(["Pipeline Specs", "Runtime Notes"])
            with d_specs:
                # MOVE the former tab3 body here, verbatim, except:
                #   - drop its st.subheader or retitle it "Pipeline specs"
                #   - make the LaTeX line a RAW string with $...$ delimiters (see below)
                ...
            with d_notes:
                # MOVE the former tab4 body here, with the two label renames from §3
                ...
```

**Existing bug to fix while moving:** the "Core calculation" line uses `\(z_i\)` inside a normal string. That raises an invalid-escape warning and Streamlit doesn't render `\( \)`. Convert to a raw string with `$` delimiters:

```python
st.markdown(r"**Core calculation:** For logits $z_i$, softmax gives $P(y=i|x)=e^{z_i}/\sum_j e^{z_j}$. The class with the largest probability is selected.")
```
The `text` code-fence workflow diagram and the Hugging Face link stay as they are.

---

## 5. CSS Enhancements

Design principle: **theme-neutral.** Use `inherit` text color and translucent neutral surfaces (`rgba(128,128,128,x)`) so cards look right in both Streamlit light and dark themes without hardcoding backgrounds. Only the hero (dark-blue gradient with white text) and the semantic green/red are fixed colors; both read well on either theme.

```python
st.markdown("""<style>
:root{
  --accent:#2563eb; --accent-2:#1e3a8a;
  --pos:#16a34a; --neg:#dc2626;
  --surface:rgba(128,128,128,.07); --border:rgba(128,128,128,.25);
  --muted:rgba(128,128,128,.95);
  --radius:14px;
}
/* Layout & spacing */
.block-container{padding-top:2rem; padding-bottom:3rem; max-width:1200px;}
footer{visibility:hidden;}                      /* keep the header/menu: it holds the theme switcher */
h2,h3{letter-spacing:-.01em;}

/* Hero */
.hero{padding:1.5rem 1.8rem; border-radius:var(--radius); margin-bottom:.9rem;
  background:linear-gradient(115deg,#0f172a 0%,#1e3a8a 55%,#2563eb 100%); color:#fff;
  box-shadow:0 10px 30px rgba(37,99,235,.18);}
.hero h1{color:#fff; margin:0 0 .25rem 0; font-size:1.9rem; line-height:1.2;}
.hero p{margin:0; color:rgba(255,255,255,.82);}
.muted{color:var(--muted);}

/* Status pill */
.status-pill{display:inline-flex; align-items:center; gap:.45rem; padding:.28rem .75rem;
  border:1px solid var(--border); border-radius:999px; background:var(--surface);
  font-size:.82rem; margin:.2rem 0 1rem 0;}
.status-pill .dot{width:8px; height:8px; border-radius:50%; background:var(--pos);
  box-shadow:0 0 0 3px rgba(22,163,74,.22);}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:.4rem; border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{padding:.6rem 1.1rem; border-radius:10px 10px 0 0; font-weight:600;}

/* Primary buttons */
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"]{
  border-radius:10px; padding:.55rem 1.4rem; font-weight:600; border:0;
  background:linear-gradient(110deg,var(--accent-2),var(--accent)); color:#fff;
  transition:transform .12s ease, box-shadow .12s ease;}
.stButton>button[kind="primary"]:hover{transform:translateY(-1px); box-shadow:0 6px 16px rgba(37,99,235,.35);}
.stDownloadButton>button{border-radius:10px;}

/* Inputs */
.stTextArea textarea{border-radius:12px; line-height:1.55;}
[data-testid="stFileUploader"] section{border-radius:12px; border:1.5px dashed var(--border); background:var(--surface);}
[data-testid="stDataFrame"]{border:1px solid var(--border); border-radius:12px; overflow:hidden;}
[data-testid="stMetric"]{background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:.8rem 1rem;}

/* Result card */
.result-card{border:1px solid var(--border); border-left:6px solid var(--accent); border-radius:var(--radius);
  background:var(--surface); padding:1.25rem 1.4rem; margin:1rem 0 .5rem 0;
  animation:rise .35s ease both;}
.result-card.pos{border-left-color:var(--pos); background:linear-gradient(90deg,rgba(22,163,74,.10),var(--surface) 60%);}
.result-card.neg{border-left-color:var(--neg); background:linear-gradient(90deg,rgba(220,38,38,.10),var(--surface) 60%);}
.rc-head{display:flex; align-items:center; gap:1rem; margin-bottom:1rem;}
.rc-icon{font-size:2.2rem; line-height:1;}
.rc-eyebrow{font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);}
.rc-label{font-size:1.7rem; font-weight:700; line-height:1.15;}
.result-card.pos .rc-label{color:var(--pos);} .result-card.neg .rc-label{color:var(--neg);}
.rc-conf{margin-left:auto; text-align:right; font-size:2rem; font-weight:700; line-height:1;}
.rc-conf span{font-size:1rem; font-weight:600; margin-left:2px;}

/* Confidence bar */
.conf-track{height:12px; border-radius:999px; background:rgba(128,128,128,.22); overflow:hidden;}
.conf-fill{height:100%; border-radius:999px; animation:grow .8s cubic-bezier(.2,.7,.2,1) both;}
.result-card.pos .conf-fill{background:linear-gradient(90deg,#4ade80,var(--pos));}
.result-card.neg .conf-fill{background:linear-gradient(90deg,#f87171,var(--neg));}

.rc-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; margin-top:1.1rem;}
.rc-stat{border:1px solid var(--border); border-radius:10px; padding:.6rem .8rem;}
.rc-stat span{display:block; font-size:.74rem; color:var(--muted);}
.rc-stat b{font-size:1.05rem;}
@media (max-width:720px){.rc-grid{grid-template-columns:1fr;} .rc-head{flex-wrap:wrap;}}

@keyframes grow{from{width:0}} @keyframes rise{from{opacity:0; transform:translateY(8px)}}

/* Sidebar */
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarContent"]{padding-top:1.5rem;}
.sidebar-spacer{height:2.5rem;}
[data-testid="stSidebar"] [data-testid="stExpander"]{border:1px dashed var(--border); border-radius:10px; opacity:.85;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary{font-size:.82rem; color:var(--muted);}
</style>""", unsafe_allow_html=True)
```

Notes for the agent:
- Streamlit's `data-testid` / `data-baseweb` selectors can shift between versions. If a selector doesn't match, the page must still render correctly (CSS is purely additive); do not "fix" a non-matching selector by touching widget code.
- Do **not** hide `#MainMenu` or the header: the "Settings → Theme" switch is needed for light/dark demonstration.
- Do not use `!important` except where a Streamlit default demonstrably overrides a rule.
- Optional (only if requested): add `.streamlit/config.toml` with `[theme] primaryColor="#2563eb"` — not required.

---

## 6. Verification Steps

Run from the project folder.

**A. Static / safety checks**
1. Before editing, record hashes: `sha256sum model.py evaluate.py requirements.txt > /tmp/before.sha`. After editing: `sha256sum -c /tmp/before.sha` → all `OK`.
2. `python -m py_compile app.py` → no output (no syntax errors, no invalid-escape warnings: run `python -W error -c "import ast,sys; ast.parse(open('app.py').read())"`).
3. `grep -n "predict(\|load_analyzer\|SentimentAnalyzer" app.py` → the same three call sites as before; no new inference calls.
4. `grep -n -i "viva\|architecture" app.py` → no UI-visible strings containing these words (code comments are acceptable but prefer none).

**B. Runtime checks** (`streamlit run app.py`)
5. Default URL `http://localhost:8501/`: exactly **2** top tabs (Single Review, Batch Analysis); the sidebar shows Model information and Quick demo reviews only; no "System Diagnostics".
6. `http://localhost:8501/?debug=true`: a collapsed **System Diagnostics** expander is the last item in the sidebar. Expanding shows two inner tabs, **Pipeline Specs** and **Runtime Notes**, with the original content. The softmax formula renders as math. The code-block workflow diagram and the Hugging Face link render.
7. Single Review:
   - Pick each of the 4 quick-demo examples → text area fills correctly → "Analyze Sentiment" → result card shows correct label, colored accent (green for Positive, red for Negative), animated confidence bar with width equal to confidence %, three stat tiles, and the caption.
   - Empty text → the existing `st.warning` appears (no crash, no card).
8. Batch Analysis:
   - Download sample CSV → upload it → "Analyze all reviews" → dataframe, "Reviews processed", "Positive share", bar chart (green/red, transparent background) and "Download analyzed CSV" all work; downloaded CSV has `predicted_sentiment` and `confidence_percent` columns.
   - Upload `sample_data/sample_reviews.csv` and a CSV without a `review` column → the existing red error message appears.
   - A row with a blank review yields `EMPTY` and does not break the chart.
9. Theme: switch Settings → Theme between Light and Dark (and "Use system setting"). Cards, pills, uploader, and chart stay legible in both; hero remains dark-blue with white text.
10. Responsive: narrow the browser to ~600px → stat tiles stack into one column; no horizontal scrolling; sidebar still collapses.
11. Browser console and terminal show no new errors or warnings.

---

## 7. Acceptance Checklist

- [ ] Only `app.py` changed; `model.py` / `evaluate.py` hashes unchanged
- [ ] Main tabs = Single Review, Batch Analysis only
- [ ] "System Diagnostics" expander is the last sidebar element, collapsed, with "Pipeline Specs" and "Runtime Notes"; hidden unless `?debug=true` (or `ALWAYS_SHOW_DIAGNOSTICS = True`)
- [ ] Sentiment result card, styled confidence bar, status pill, refined hero and spacing in place
- [ ] Works in light and dark themes
- [ ] All original functionality (single review, demo selector, CSV batch, downloads, error handling) verified working
