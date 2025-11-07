"""
🎬 Movie Review Sentiment — Streamlit Advanced Portfolio Edition (English-only)

Highlights:
- English-only (DistilBERT SST-2)
- Single review: explanations, aspect hints, thresholding
- Evaluation: Confusion matrix, ROC, PR, Calibration, Threshold sweep, Slice metrics
- Figures auto-scaled smaller for better UI balance
- Lightweight (CPU optimized, no sklearn)
"""

import os, numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# NLTK bits with graceful data setup
import nltk
from nltk import pos_tag, word_tokenize
try:
    _ = nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    _ = nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger", quiet=True)

from rapidfuzz import fuzz

# === FIGURE SIZE CONTROLS (affects Evaluate tab plots) ===
FIGSIZE = (2.8, 2.2)   # inches: width, height
DPI = 160              # keeps text crisp at small size

# =============== SETUP =================
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
st.set_page_config(page_title="Movie Sentiment Lab", page_icon="🎬", layout="wide")

st.title("🎬 Movie Review Sentiment Analysis")
st.caption("DistilBERT SST-2 | Interpretability + Evaluation | Streamlit")

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
DEVICE = 0 if torch.cuda.is_available() else -1
MAX_TOKENS_EXPLAIN = 200

# =============== HELPERS =================
def aspect_hints(text, top_k=6):
    """Extract top noun/noun phrases."""
    try:
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        nouns = [w for w, t in tags if t.startswith("NN") and w.isalpha()]
        bigrams = [f"{nouns[i]} {nouns[i+1]}" for i in range(len(nouns)-1)]
        words = nouns + bigrams
        if not words:
            return []
        # frequency (simple) + fuzzy dedupe
        freq = {}
        for w in words:
            wl = w.lower()
            freq[wl] = freq.get(wl, 0) + 1
        uniq = []
        for w in sorted(freq, key=freq.get, reverse=True):
            if all(fuzz.QRatio(w, u) < 85 for u in uniq):
                uniq.append(w)
            if len(uniq) >= top_k:
                break
        return uniq
    except Exception:
        return []

def occlusion_explain(text, tokenizer, model, top_k=8):
    """Token-level occlusion sensitivity using [MASK] substitution."""
    model.eval()
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS_EXPLAIN)
        base = torch.softmax(model(**enc).logits, dim=-1)[0]
        pred_idx = int(torch.argmax(base))
        mask_id = tokenizer.mask_token_id
        if mask_id is None:
            return []
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        results = []
        for i, tok_id in enumerate(enc["input_ids"][0]):
            if tok_id in {tokenizer.cls_token_id, tokenizer.sep_token_id}:
                continue
            occluded = enc["input_ids"].clone()
            occluded[0, i] = mask_id
            logits = model(input_ids=occluded, attention_mask=enc["attention_mask"]).logits
            drop = (base[pred_idx] - torch.softmax(logits, dim=-1)[0][pred_idx]).item()
            results.append((toks[i].replace("##", ""), drop))
        # Top-k most influential (largest drop)
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

# ======== MODEL & INFERENCE ========
@st.cache_resource(show_spinner=False)
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, device=DEVICE)
    return pipe, tok, mdl

@st.cache_data(show_spinner=False)
def predict(_pipe, texts, max_len=256):
    outs = _pipe(texts, top_k=None, truncation=True, padding=True, max_length=max_len)
    return [{s["label"]: float(s["score"]) for s in o} for o in outs]

def prob_pos(score_dict): return score_dict.get("POSITIVE", 0.0)
def gt_bin(lbl): 
    s = str(lbl).lower()
    return "positive" if ("pos" in s or "4" in s or "5" in s or s == "1") else "negative"  # tolerant mapping
def pred_bin(prob, thr): return "positive" if prob >= thr else "negative"

# ======== METRICS ========
def confusion(y_true, y_pred):
    tp = sum(t=="positive" and p=="positive" for t,p in zip(y_true,y_pred))
    tn = sum(t=="negative" and p=="negative" for t,p in zip(y_true,y_pred))
    fp = sum(t=="negative" and p=="positive" for t,p in zip(y_true,y_pred))
    fn = sum(t=="positive" and p=="negative" for t,p in zip(y_true,y_pred))
    return tp, fp, tn, fn

def metrics(tp, fp, tn, fn):
    denom = max(1, tp+tn+fp+fn)
    acc = (tp+tn)/denom
    prec = tp/max(1, tp+fp)
    rec = tp/max(1, tp+fn)
    f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return acc, prec, rec, f1

def roc_curve_np(y_true, y_score):
    y_true, y_score = np.array(y_true, dtype=float), np.array(y_score, dtype=float)
    if len(y_true) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    desc = np.argsort(-y_score)
    y_true = y_true[desc]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    P = y_true.sum()
    N = len(y_true) - P
    P = max(P, 1.0)
    N = max(N, 1.0)
    tpr = np.concatenate([[0], tps/P])
    fpr = np.concatenate([[0], fps/N])
    return fpr, tpr

def pr_curve_np(y_true, y_score):
    y_true, y_score = np.array(y_true, dtype=float), np.array(y_score, dtype=float)
    if len(y_true) == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0])
    desc = np.argsort(-y_score)
    y_true = y_true[desc]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    P = max(1.0, y_true.sum())
    recall = tp / P
    precision = tp / np.maximum(1.0, tp + fp)
    return np.concatenate([[0], recall]), np.concatenate([[1], precision])

def reliability_curve(y_true, y_score, bins=10):
    y_true, y_score = np.array(y_true, dtype=float), np.array(y_score, dtype=float)
    edges = np.linspace(0, 1, bins+1)
    mids, accs, confs = [], [], []
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        m = (y_score>=lo)&(y_score<hi) if i<bins-1 else (y_score>=lo)&(y_score<=hi)
        if not m.any(): 
            continue
        conf = float(y_score[m].mean())
        acc = float(y_true[m].mean())
        mids.append((lo+hi)/2)
        accs.append(acc)
        confs.append(conf)
        ece += abs(acc - conf) * m.sum() / len(y_true)
    return np.array(mids), np.array(accs), np.array(confs), float(ece)

# ======== SIDEBAR ========
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Positive threshold", 0.5, 0.95, 0.6, 0.01)
    show_probs = st.checkbox("Show probabilities", True)
    show_explain = st.checkbox("Show token explanations", True)
    show_aspects = st.checkbox("Show aspect hints", True)
    max_len = st.slider("Max tokens (truncation)", 64, 512, 256, 32)

# ======== LOAD MODEL ========
pipe, tokenizer, model = load_model()
labels = [s["label"] for s in pipe(["ok"], top_k=None)[0]]
st.success(f"Loaded model: {MODEL_ID}\nLabels: {', '.join(labels)}")

# ======== TABS ========
tab1, tab2, tab3 = st.tabs(["🔎 Analyze", "📊 Evaluate", "ℹ️ About"])

# --- TAB 1 ---
with tab1:
    st.subheader("Single Review Analysis")
    text = st.text_area("Enter review:", "The movie had great visuals but a dull story.", height=150)
    if st.button("Analyze", type="primary"):
        with st.spinner("Analyzing..."):
            res = predict(pipe, [text], max_len=max_len)[0]
        p = prob_pos(res)
        pred = pred_bin(p, threshold)
        st.metric("Prediction", pred)
        st.metric("Confidence", f"{p:.3f}")
        if show_probs:
            # Bar chart expects index/column structure
            prob_df = pd.DataFrame({"label": list(res.keys()), "prob": list(res.values())}).set_index("label")
            st.bar_chart(prob_df)
        if show_aspects:
            aspects = aspect_hints(text)
            if aspects:
                st.caption("Aspect hints")
                st.markdown(", ".join(f"`{a}`" for a in aspects))
        if show_explain:
            st.caption("Top tokens (mask occlusion)")
            try:
                expl = occlusion_explain(text, tokenizer, model)
                if expl:
                    cols = st.columns(len(expl))
                    for i,(tok,drop) in enumerate(expl):
                        with cols[i]:
                            st.metric(tok, f"Δ{drop:.3f}")
            except Exception as e:
                st.warning(str(e))

# --- TAB 2 ---
with tab2:
    st.subheader("Evaluate Dataset")
    st.caption("Upload CSV with `text` and `label` columns (pos/neg or 1–5 stars).")
    f = st.file_uploader("Upload CSV", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
        if "text" not in df.columns or "label" not in df.columns:
            st.error("CSV must contain 'text' and 'label' columns.")
        elif st.button("Run Evaluation", type="primary"):
            with st.spinner("Evaluating..."):
                texts = df["text"].astype(str).tolist()
                scores = predict(pipe, texts, max_len=max_len)

            y_true = [gt_bin(l) for l in df["label"]]
            y_prob = [prob_pos(s) for s in scores]
            y_pred = [pred_bin(p, threshold) for p in y_prob]
            y_true_bin = [1 if t=="positive" else 0 for t in y_true]

            tp, fp, tn, fn = confusion(y_true, y_pred)
            acc, prec, rec, f1 = metrics(tp, fp, tn, fn)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Precision", f"{prec:.3f}")
            c3.metric("Recall", f"{rec:.3f}")
            c4.metric("F1", f"{f1:.3f}")

            # Confusion Matrix (compact)
            cm = np.array([[tn, fp], [fn, tp]])
            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            ax.imshow(cm)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["neg","pos"]); ax.set_yticklabels(["neg","pos"])
            ax.set_title("Confusion Matrix")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True, use_container_width=False)

            # ROC curve (compact)
            fpr, tpr = roc_curve_np(y_true_bin, y_prob)
            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            ax.plot(fpr, tpr, label="ROC")
            ax.plot([0,1],[0,1],'--',color='gray')
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title("ROC Curve"); ax.legend()
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True, use_container_width=False)

            # PR curve (compact)
            recall, precision = pr_curve_np(y_true_bin, y_prob)
            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            ax.plot(recall, precision)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title("PR Curve")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True, use_container_width=False)

            # Calibration curve (compact)
            mids, accs, confs, ece = reliability_curve(y_true_bin, y_prob, bins=10)
            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            ax.plot([0,1],[0,1],'--',color='gray')
            ax.scatter(confs, accs)
            ax.set_title(f"Calibration (ECE={ece:.3f})")
            ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True, use_container_width=False)

            # Threshold sweep (table)
            grid = np.linspace(0.3, 0.9, 25)
            sweep = []
            for th in grid:
                yp = [pred_bin(p, th) for p in y_prob]
                tp_,fp_,tn_,fn_ = confusion(y_true, yp)
                a,p_,r_,f_ = metrics(tp_,fp_,tn_,fn_)
                sweep.append((th, a, p_, r_, f_))
            df_sweep = pd.DataFrame(sweep, columns=["threshold","acc","prec","rec","f1"])
            st.dataframe(
                df_sweep.style.format({
                    "threshold": "{:.2f}",
                    "acc": "{:.3f}", "prec": "{:.3f}", "rec": "{:.3f}", "f1": "{:.3f}"
                }),
                use_container_width=False
            )
            best = df_sweep.iloc[df_sweep["f1"].idxmax()]
            st.info(f"Best F1={best['f1']:.3f} at threshold={best['threshold']:.2f}")

            # Slice metrics by text length
            st.subheader("Slice Metrics by Text Length")
            lens = df["text"].astype(str).str.len().values
            bins = [0, 80, 160, 320, 640, 9999]
            results = []
            for i in range(len(bins)-1):
                lo, hi = bins[i], bins[i+1]
                m = (lens>=lo) & (lens<hi)
                if not m.any():
                    continue
                idxs = np.where(m)[0]
                yt = [y_true[j] for j in idxs]
                yp = [y_pred[j] for j in idxs]
                tp_,fp_,tn_,fn_ = confusion(yt, yp)
                a,p_,r_,f_ = metrics(tp_, fp_, tn_, fn_)
                results.append((f"[{lo},{hi})", int(m.sum()), a, p_, r_, f_))
            df_slice = pd.DataFrame(results, columns=["len_range","n","acc","prec","rec","f1"])
            st.dataframe(
                df_slice.style.format({
                    "n": "{:,}",
                    "acc": "{:.3f}", "prec": "{:.3f}", "rec": "{:.3f}", "f1": "{:.3f}"
                }),
                use_container_width=False
            )

            # Error Explorer
            st.subheader("Error Explorer")
            errs = [{"text":t,"true":gt,"pred":pr,"prob":round(pp,3)}
                    for t,gt,pr,pp in zip(df["text"], y_true, y_pred, y_prob) if gt != pr]
            if errs:
                st.dataframe(pd.DataFrame(errs).head(100))
            else:
                st.success("No errors detected 🎉")

            # Download
            df_out = df.copy()
            df_out["true"] = y_true
            df_out["pred"] = y_pred
            df_out["prob_pos"] = y_prob
            st.download_button("Download evaluated CSV", df_out.to_csv(index=False).encode(), "evaluated.csv")

# --- TAB 3 ---
with tab3:
    st.markdown("""
    ### About
    - fast CPU sentiment analyzer for portfolio showcase  
    - Adds interpretability (aspect & token influence)  
    - Evaluation: Confusion matrix, ROC, PR, Calibration, Threshold sweep  
    - Clean compact visuals for presentations or demos
    """)
