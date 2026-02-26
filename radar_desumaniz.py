# enviado ///////////////////////////////////////////////////

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import base64
import io
import json
import streamlit.components.v1 as components
import requests
import plotly.express as px
from scipy.stats import entropy as scipy_entropy


# =========================
# Helpers
# =========================
def to_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace(" ", "")
    s = s.replace(".", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def robust_minmax(series: pd.Series, q_lo=0.02, q_hi=0.98) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo = s.quantile(q_lo)
    hi = s.quantile(q_hi)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    s2 = s.clip(lo, hi)
    return (s2 - lo) / (hi - lo)


def angle_from_cycle(dt: pd.Series, cycle: str) -> pd.Series:
    if cycle == "Por hora do Dia":
        seconds = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
        frac = seconds / 86400.0
        return 2 * np.pi * frac
    if cycle == "Por dia da Semana":
        frac = (dt.dt.dayofweek + (dt.dt.hour / 24.0)) / 7.0
        return 2 * np.pi * frac
    if cycle == "Por dia do Mês":
        dim = dt.dt.days_in_month.astype(float)
        frac = ((dt.dt.day - 1) + (dt.dt.hour / 24.0)) / dim
        return 2 * np.pi * frac
    if cycle == "Por mês do Ano":
        doy = dt.dt.dayofyear.astype(float)
        frac = (doy - 1 + (dt.dt.hour / 24.0)) / 365.25
        return 2 * np.pi * frac
    raise ValueError("Ciclo inválido")


def radial_from_intensity(int01: pd.Series, inner=0.10, outer=1.0) -> pd.Series:
    inv = 1.0 - int01
    return inner + inv * (outer - inner)


def make_cloud(theta0, r0, rep01, density=1.0, jitter_theta=0.06, jitter_r=0.035):
    rep01 = float(0.0 if pd.isna(rep01) else np.clip(rep01, 0, 1))
    rep01 = float(np.clip(rep01 * density, 0, 1))

    n_min, n_max = 6, 60
    n = int(round(n_min + rep01 * (n_max - n_min)))
    if n <= 0:
        return np.array([]), np.array([]), np.array([])

    jt = jitter_theta * (0.6 + 0.8 * rep01)
    jr = jitter_r * (0.6 + 0.8 * rep01)

    th = theta0 + np.random.normal(0, jt, size=n)
    rr = r0 + np.random.normal(0, jr, size=n)
    rr = np.clip(rr, 0.02, 1.05)

    base = 6
    sizes = base + (rep01 * 18) * np.clip(
        np.random.lognormal(mean=-0.2, sigma=0.55, size=n), 0.4, 2.8
    )
    return th, rr, sizes


def safe_text(s, max_len=240):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip()
    return s if len(s) <= max_len else s[:max_len].rstrip() + "…"


def find_image(folder: Path, key: str):
    """Busca robusta lidando com extensões e zeros à esquerda (id_01 vs id_1)."""
    exts = [".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG"]
    key = str(key).strip()
    possible_keys = [key]
    if "id_0" in key:
        possible_keys.append(key.replace("id_0", "id_"))
    elif "id_" in key and len(key) == 4:
        possible_keys.append(key.replace("id_", "id_0"))
    for k in possible_keys:
        for ext in exts:
            p = folder / f"{k}{ext}"
            if p.exists():
                return p
    return None



def image_to_base64(img_path: Path, max_width: int = 300) -> str:
    try:
        img = Image.open(img_path).convert("RGB")
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""



def compute_entropia_angular(theta_rad: np.ndarray, n_bins: int = 12) -> float:
    """
    Entropia de Shannon sobre bins angulares.
    Alta entropia → discurso distribuído (orgânico).
    Baixa entropia → concentrado (possível coordenação).
    Retorna valor normalizado 0–1 (H / log(n_bins)).
    """
    theta_rad = np.asarray(theta_rad)
    theta_rad = theta_rad[~np.isnan(theta_rad)] % (2 * np.pi)
    if len(theta_rad) < 2:
        return np.nan
    counts, _ = np.histogram(theta_rad, bins=n_bins, range=(0, 2 * np.pi))
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    probs = counts / counts.sum()
    h = -np.sum(probs * np.log(probs))
    h_max = np.log(n_bins)
    return float(h / h_max)


def compute_deriva(df_autor: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada autor com ≥2 posts, calcula o vetor de deriva entre posts consecutivos.
    Retorna DataFrame com colunas: autor, id_from, id_to, delta_theta, delta_r, delta_dias.
    """
    rows = []
    for autor, grp in df_autor.groupby("autor"):
        grp = grp.sort_values("_dt").reset_index(drop=True)
        if len(grp) < 2:
            continue
        for i in range(len(grp) - 1):
            a, b = grp.iloc[i], grp.iloc[i + 1]
            d_theta = float(b["_theta"] - a["_theta"])
            # normaliza para [-π, π]
            d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
            d_r = float(b["_r"] - a["_r"])
            d_dias = (b["_dt"] - a["_dt"]).days if pd.notna(b["_dt"]) and pd.notna(a["_dt"]) else np.nan
            rows.append({
                "autor": autor,
                "id_from": str(a["id"]),
                "id_to": str(b["id"]),
                "tipo_from": a["_tipo"],
                "tipo_to": b["_tipo"],
                "delta_theta_graus": round(np.degrees(d_theta), 1),
                "delta_r": round(d_r, 3),
                "delta_dias": d_dias,
                "escalada": d_r < -0.05,   # aproxima-se do centro = mais intenso
                "mudou_tipo": a["_tipo"] != b["_tipo"],
            })
    return pd.DataFrame(rows)



CLASSIFY_PROMPT = """Você é um pesquisador especialista em teoria da desumanização.
Classifique o discurso abaixo em UMA das categorias:

- infra-humanização: o alvo é comparado a animais, seres primitivos ou desprovidos de cultura/razão
- supra-humanização: o alvo é comparado a demônios, monstros sobrenaturais ou seres ameaçadores não-humanos
- quase-desumanização: o alvo é tratado como objeto moral inferior sem metáfora animal/sobrenatural
- outros: não se enquadra em nenhuma das categorias acima

Título: {titulo} | Teoria: {teoria} | Categoria: {categoria}
Texto: {texto}

Responda APENAS em JSON válido:
{{"tipo": "infra|supra|quase|outros", "justificativa": "...", "confianca": "alta|media|baixa"}}
"""

def _parse_llm(raw):
    s, e = raw.find("{"), raw.rfind("}") + 1
    if s == -1 or e == 0:
        return {"tipo": "outros", "justificativa": "Resposta inválida.", "confianca": "baixa"}
    try:
        r = json.loads(raw[s:e])
        if r.get("tipo") not in ["infra","supra","quase","outros"]: r["tipo"] = "outros"
        return r
    except Exception as ex:
        return {"tipo": "outros", "justificativa": str(ex), "confianca": "baixa"}

def classify_with_ollama(texto, titulo, teoria, categoria, url, model):
    prompt = CLASSIFY_PROMPT.format(titulo=titulo, teoria=teoria, categoria=categoria, texto=texto)
    try:
        r = requests.post(f"{url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}, timeout=90)
        r.raise_for_status()
        return _parse_llm(r.json().get("response",""))
    except requests.exceptions.ConnectionError:
        return {"tipo": None, "justificativa": "❌ Ollama não encontrado.", "confianca": None}
    except Exception as e:
        return {"tipo": None, "justificativa": f"❌ {e}", "confianca": None}

def classify_with_groq(texto, titulo, teoria, categoria, api_key, model):
    prompt = CLASSIFY_PROMPT.format(titulo=titulo, teoria=teoria, categoria=categoria, texto=texto)
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role":"user","content":prompt}],
                  "temperature": 0.1, "max_tokens": 400}, timeout=30)
        r.raise_for_status()
        return _parse_llm(r.json()["choices"][0]["message"]["content"])
    except requests.exceptions.ConnectionError:
        return {"tipo": None, "justificativa": "❌ Sem conexão com Groq.", "confianca": None}
    except requests.exceptions.HTTPError:
        code = r.status_code
        msgs = {401: "❌ API key inválida.", 400: "❌ Modelo inválido (erro 400)."}
        return {"tipo": None, "justificativa": msgs.get(code, f"❌ HTTP {code}"), "confianca": None}
    except Exception as e:
        return {"tipo": None, "justificativa": f"❌ {e}", "confianca": None}


def parse_date_range(date_range, fallback_start, fallback_end):
    if isinstance(date_range, (list, tuple)):
        if len(date_range) == 2:
            start_date, end_date = date_range
        elif len(date_range) == 1:
            start_date = end_date = date_range[0]
        else:
            start_date, end_date = fallback_start, fallback_end
    else:
        start_date = end_date = date_range

    if start_date > end_date:
        start_date, end_date = end_date, start_date
    return start_date, end_date


def circular_concentration(angles_rad: np.ndarray) -> float:
    """Comprimento do vetor médio (0..1): concentração circular."""
    if angles_rad is None:
        return np.nan
    angles_rad = np.asarray(angles_rad)
    angles_rad = angles_rad[~np.isnan(angles_rad)]
    if angles_rad.size == 0:
        return np.nan
    x = np.mean(np.cos(angles_rad))
    y = np.mean(np.sin(angles_rad))
    return float(np.sqrt(x * x + y * y))


def normalize_tipo(categoria: str, teoria: str) -> str:
    cat = "" if pd.isna(categoria) else str(categoria).strip().lower()
    teo = "" if pd.isna(teoria) else str(teoria).strip().lower()

    if "infrah" in cat:
        return "infra"
    if "suprah" in cat:
        return "supra"
    if "quase" in cat:
        return "quase"

    if "animal" in teo:
        return "infra"
    if "demon" in teo:
        return "supra"
    if "paternal" in teo or "moral" in teo:
        return "quase"

    return "outros"


def bin_radial_band(r: pd.Series):
    bins = [0.0, 0.25, 0.50, 0.75, 1.10]
    labels = [1, 2, 3, 4]
    return pd.cut(r, bins=bins, labels=labels, include_lowest=True)


def cycle_bins(cycle: str):
    if cycle == "Por hora do Dia":
        tickvals = [i * 360/24 for i in range(0, 24, 3)]
        ticktext = [f"{i:02d}h" for i in range(0, 24, 3)]
        label = "hora"
        unit_values = list(range(24))
        return tickvals, ticktext, label, unit_values
    if cycle == "Por dia da Semana":
        tickvals = [i * 360/7 for i in range(7)]
        ticktext = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
        label = "dia_semana"
        unit_values = list(range(7))
        return tickvals, ticktext, label, unit_values
    if cycle == "Por dia do Mês":
        tickvals = [i * 360/31 for i in range(0, 31, 5)]
        ticktext = [str(i+1) for i in range(0, 31, 5)]
        label = "dia_mes"
        unit_values = list(range(1, 32))
        return tickvals, ticktext, label, unit_values
    tickvals = [i * 30 for i in range(12)]
    ticktext = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
    label = "mes"
    unit_values = list(range(1, 13))
    return tickvals, ticktext, label, unit_values


# =========================
# App
# =========================
st.set_page_config(page_title="Radar crítico – desumanização", layout="wide")
st.title("Desumanização: infra/supra/quase (versão preliminar)")

if "selected_id" not in st.session_state:
    st.session_state["selected_id"] = None

with st.sidebar:
    # ── caminhos fixos no servidor ────────────────────────────────────────────
    # Edite estas constantes conforme a estrutura do seu repositório.
    CSV_PATH     = Path("dataset.csv")   # CSV na raiz do repositório
    image_folder = Path("imagens")       # pasta de imagens na raiz do repositório
    sep = ","
    enc = "utf-8"
    # ─────────────────────────────────────────────────────────────────────────

    st.divider()
    st.header("Ciclo do radar")
    cycle = st.selectbox("Ciclo", ["Por hora do Dia", "Por dia da Semana", "Por dia do Mês", "Por mês do Ano"], index=0)

    st.divider()
    st.header("Radar")
    inner = st.slider("Centro (raio mínimo)", 0.02, 0.30, 0.10, 0.01)

    st.subheader("Manchas (réplicas)")
    density = st.slider("Densidade", 0.5, 2.0, 1.0, 0.1)
    jitter_theta = st.slider("Dispersão angular", 0.01, 0.20, 0.06, 0.01)
    jitter_r = st.slider("Dispersão radial", 0.005, 0.10, 0.035, 0.005)

    st.divider()
    st.header("Métricas de engajamento")
    metrica_rep = st.selectbox("Métrica de repercussão",
        ["IV — Índice de Viralização (likes/√seguidores)",
         "IE — Engajamento Composto ((likes+coment+compart)/seguidores)",
         "IA — Alcance Ponderado (visualizações × IV_norm)",
         "IRD — Ressonância Desproporcional (likes/seg^α)"],
        index=0)
    ird_alpha = st.slider("α (expoente IRD)", 0.1, 1.0, 0.5, 0.05,
        help="0.5 = equivale ao IV. Valores menores favorecem contas pequenas.") if metrica_rep.startswith("IRD") else 0.5
    decay_lambda = st.slider("Decaimento temporal (λ)", 0.0, 0.05, 0.0, 0.005,
        help="0=sem decaimento. 0.01→meia-vida ~70 dias.")

    st.divider()
    st.header("Índice sintético")
    w_int  = st.slider("Peso: intensidade",    0.0, 1.0, 0.40, 0.05)
    w_rep  = st.slider("Peso: repercussão",    0.0, 1.0, 0.40, 0.05)
    w_conc = st.slider("Peso: concentração",   0.0, 1.0, 0.20, 0.05)

    st.divider()
    st.header("Deriva por autor")
    show_deriva_radar = st.checkbox("Mostrar deriva no radar", value=False,
        help="Traça linhas conectando posts consecutivos do mesmo autor.")
    entropia_bins = st.slider("Bins para entropia angular", 6, 24, 12, 2,
        help="Número de fatias do ciclo para calcular a entropia de Shannon.")

    st.divider()
    st.header("Heatmap polar")
    heat_metric = st.selectbox("Métrica", ["eventos", "replicas_total", "intensidade_media"], index=1)

    st.divider()
    st.header("Exibição")
    show_cloud = st.checkbox("Mostrar manchas", True)
    show_points = st.checkbox("Mostrar pontos-origem", True)

if not CSV_PATH.exists():
    st.error(f"Arquivo não encontrado: `{CSV_PATH.resolve()}`. Verifique o caminho em CSV_PATH no código.")
    st.stop()

df0 = pd.read_csv(CSV_PATH, sep=sep, encoding=enc, dtype=str, engine="python").dropna(how="all").copy()

required_cols = ["id", "data", "autor", "seguidores", "titulo", "likes", "conteudo",
                 "replicas", "categoria", "ente", "teoria", "conflito", "imagem"]
missing = [c for c in required_cols if c not in df0.columns]
if missing:
    st.error("Colunas faltando no CSV: " + ", ".join(missing) + "\n\nVerifique separador/encoding.")
    st.stop()



# ── pseudonimização de autores ────────────────────────────────────────────────
# Substitui nomes reais por pseudônimos estáveis (mesmo autor → mesmo pseudônimo
# em qualquer sessão), usando hash SHA-256 truncado como semente.
import hashlib, random as _random

_ADJETIVOS = [
    "Azul","Verde","Solar","Lunar","Névoa","Bruma","Coral","Âmbar","Pérola","Safira",
    "Cinza","Dourado","Prateado","Rubro","Índigo","Ciano","Violeta","Escarlate","Jade","Opala",
]
_SUBSTANTIVOS = [
    "Gavião","Corvo","Falcão","Lince","Lobo","Onça","Urso","Veado","Raposa","Tucano",
    "Pombo","Sabiá","Albatroz","Pelicano","Garça","Cegonha","Marta","Fênix","Grifo","Sereia",
]

def _pseudonimo(nome_real: str) -> str:
    """Gera pseudônimo estável a partir do nome real via hash SHA-256."""
    h = int(hashlib.sha256(str(nome_real).encode("utf-8")).hexdigest(), 16)
    adj = _ADJETIVOS[h % len(_ADJETIVOS)]
    sub = _SUBSTANTIVOS[(h // len(_ADJETIVOS)) % len(_SUBSTANTIVOS)]
    num = (h % 900) + 100  # número 100–999 para unicidade extra
    return f"{adj}{sub}{num}"

if "autor" in df0.columns:
    _mapa_autor = {a: _pseudonimo(a) for a in df0["autor"].dropna().unique()}
    df0["autor"] = df0["autor"].map(_mapa_autor).fillna("Anônimo")
# ─────────────────────────────────────────────────────────────────────────────

df = df0.copy()

dt_date = pd.to_datetime(df["data"].astype(str).str.strip(), dayfirst=True, errors="coerce")
hora = df["hora"].astype(str).str.strip() if "hora" in df.columns else pd.Series([""] * len(df))
dt_str = dt_date.dt.strftime("%Y-%m-%d").fillna("") + " " + hora.fillna("")
dt_full = pd.to_datetime(dt_str, errors="coerce")
df["_dt"] = dt_full.fillna(dt_date)

df = df[df["_dt"].notna()].copy()
if df.empty:
    st.error("Nenhuma data válida reconhecida. Confira o formato dd/mm/aaaa na coluna 'data'.")
    st.stop()

df["_likes"]      = df["likes"].apply(to_number)
df["_seguidores"] = df["seguidores"].apply(to_number)

def _col(name): return df[name].apply(to_number) if name in df.columns else pd.Series(0.0, index=df.index)
df["_comentarios"]       = _col("comentarios")
df["_compartilhamentos"] = _col("compartilhamentos")
df["_visualizacoes"]     = _col("visualizacoes")

# IV: Índice de Viralização = likes / √seguidores
df["_iv_raw"] = df["_likes"].clip(lower=0) / np.sqrt(df["_seguidores"].clip(lower=1))

# IE: Engajamento Composto
engaj = df["_likes"].fillna(0) + df["_comentarios"].fillna(0) + df["_compartilhamentos"].fillna(0)
df["_ie_raw"] = (engaj / df["_seguidores"].clip(lower=1)) * 100.0

# IA: Alcance Ponderado = visualizações × IV_norm
df["_ia_raw"] = df["_visualizacoes"].clip(lower=0) * robust_minmax(df["_iv_raw"]).fillna(0)

# IRD: Ressonância Desproporcional = likes / seguidores^α
df["_ird_raw"] = df["_likes"].clip(lower=0) / (df["_seguidores"].clip(lower=1) ** ird_alpha)

# decaimento temporal opcional
if decay_lambda > 0:
    dias = (pd.Timestamp.now() - df["_dt"]).dt.days.clip(lower=0)
    fator = np.exp(-decay_lambda * dias)
    for col in ["_iv_raw","_ie_raw","_ia_raw","_ird_raw"]:
        df[col] = df[col] * fator

# métrica ativa
_metric_map = {
    "IV": ("_iv_raw", "IV — likes/√seg"),
    "IE": ("_ie_raw", "IE — engaj. composto (%)"),
    "IA": ("_ia_raw", "IA — alcance ponderado"),
    "IRD": ("_ird_raw", f"IRD — ressonância (α={ird_alpha})"),
}
_key = next(k for k in _metric_map if metrica_rep.startswith(k))
_rep_col, _rep_label = _metric_map[_key]
df["_replicas_proxy"] = df[_rep_col]

df["_tipo"] = [normalize_tipo(c, t) for c, t in zip(df["categoria"], df["teoria"])]

tipos = ["infra", "supra", "quase", "outros"]
presentes = [t for t in tipos if t in df["_tipo"].unique().tolist()]
default_sel = [t for t in ["infra", "supra", "quase"] if t in presentes] or presentes[:]

left, right = st.columns([1, 2])
with left:
    selected_types = st.multiselect("Tipos a incluir", tipos, default=default_sel)

with right:
    dmin_ts = df["_dt"].min()
    dmax_ts = df["_dt"].max()
    date_range = st.date_input("Intervalo", value=(dmin_ts.date(), dmax_ts.date()))

start_date, end_date = parse_date_range(date_range, dmin_ts.date(), dmax_ts.date())
st.caption(f"Filtro ativo: {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}")

df = df[df["_tipo"].isin(selected_types)].copy()
df = df[(df["_dt"].dt.date >= start_date) & (df["_dt"].dt.date <= end_date)].copy()
if df.empty:
    st.warning("Nada para plotar com os filtros atuais.")
    st.stop()

df["_int_raw"] = np.log1p(df["_ie_raw"].clip(lower=0))
df["_int01"] = robust_minmax(df["_int_raw"])
df["_rep01"] = robust_minmax(df["_replicas_proxy"].clip(lower=0))

df["_theta"] = angle_from_cycle(df["_dt"], cycle)
df["_r"] = radial_from_intensity(df["_int01"], inner=inner, outer=1.0)

tickvals, ticktext, ciclo_label, unit_values = cycle_bins(cycle)

tab_radar, tab_painel, tab_galeria, tab_classif = st.tabs(["Radar", "Painel Analítico", "Galeria Interativa", "🤖 Classificação Assistida"])

with tab_radar:
    st.markdown("### Radar (seleção → imagem do `id` + metadados)")

    # ── Linha do tempo: lê o intervalo salvo no session_state ────────────────
    # O slider é renderizado ABAIXO do radar; o valor da rodada anterior fica
    # guardado em st.session_state["timeline_range"] para ser lido aqui.
    _all_dates   = sorted(df["_dt"].dt.date.unique())
    _all_years   = sorted({d.year for d in _all_dates})
    _slider_dates = _all_dates
    _n = len(_slider_dates)

    def _fmt_date(d):
        return d.strftime("%d/%m/%Y")

    # Inicializa com o intervalo completo na primeira execução.
    # Só escreve no session_state se a chave ainda não existe — evita o aviso
    # "widget created with default value but also set via Session State API".
    if "timeline_range" not in st.session_state:
        st.session_state["timeline_range"] = (0, _n - 1)
    else:
        _saved = st.session_state["timeline_range"]
        _lo = max(0, min(_saved[0], _n - 1))
        _hi = max(0, min(_saved[1], _n - 1))
        if _lo > _hi:
            _lo, _hi = 0, _n - 1
        st.session_state["timeline_range"] = (_lo, _hi)

    _idx_lo, _idx_hi = st.session_state["timeline_range"]

    _date_lo = _slider_dates[_idx_lo]
    _date_hi = _slider_dates[_idx_hi]
    df_radar = df[
        (df["_dt"].dt.date >= _date_lo) &
        (df["_dt"].dt.date <= _date_hi)
    ].copy()
    # ─────────────────────────────────────────────────────────────────────────

    fig = go.Figure()

    bands = [0.25, 0.50, 0.75, 1.00]
    theta_ring = np.linspace(0, 360, 361)
    for b in bands:
        fig.add_trace(go.Scatterpolar(
            theta=theta_ring,
            r=np.full_like(theta_ring, b, dtype=float),
            mode="lines",
            line=dict(width=1),
            opacity=0.25,
            hoverinfo="skip",
            showlegend=False
        ))

    TIPO_COLOR = {
        "infra":  "#e74c3c",
        "supra":  "#9b59b6",
        "quase":  "#f39c12",
        "outros": "#2ecc71",
    }

    # Lookup base64 das imagens (usa df completo para manter hover disponível)
    img_lookup = {}
    for _, row in df.iterrows():
        sid_i = str(row["id"])
        ip = find_image(image_folder, sid_i)
        if ip:
            img_lookup[sid_i] = image_to_base64(ip, max_width=300)

    st.caption(f"Exibindo **{len(df_radar)}** de {len(df)} eventos filtrados")

    np.random.seed(7)
    if show_cloud:
        for t in ["infra", "supra", "quase", "outros"]:
            dft = df_radar[df_radar["_tipo"] == t]
            if dft.empty:
                continue
            cor = TIPO_COLOR.get(t, "#aaaaaa")
            cloud_thetas, cloud_rs, cloud_sizes = [], [], []
            for _, row in dft.iterrows():
                th, rr, sz = make_cloud(
                    row["_theta"], row["_r"], row["_rep01"],
                    density=density, jitter_theta=jitter_theta, jitter_r=jitter_r
                )
                if len(th) == 0:
                    continue
                cloud_thetas.append(th)
                cloud_rs.append(rr)
                cloud_sizes.append(sz)

            if cloud_thetas:
                all_sizes = np.concatenate(cloud_sizes) * 1.8  # manchas maiores
                fig.add_trace(go.Scatterpolar(
                    theta=np.degrees(np.concatenate(cloud_thetas)),
                    r=np.concatenate(cloud_rs),
                    mode="markers",
                    marker=dict(size=all_sizes, color=cor, opacity=0.22),
                    name=f"Manchas: {t}",
                    hoverinfo="skip",
                    showlegend=False,
                ))

    if show_points:
        for t in ["infra", "supra", "quase", "outros"]:
            dft = df_radar[df_radar["_tipo"] == t]
            if dft.empty:
                continue
            cor = TIPO_COLOR.get(t, "#aaaaaa")

            # jitter nos pontos-origem para desaglomerar sobreposições
            np.random.seed(42)
            n = len(dft)
            theta_jit = np.degrees(dft["_theta"]) + np.random.uniform(-2.5, 2.5, n)
            r_jit     = dft["_r"].values          + np.random.uniform(-0.025, 0.025, n)
            r_jit     = np.clip(r_jit, 0.05, 1.05)

            customdata = np.stack([
                dft["id"].astype(str).values,
                dft["_dt"].astype(str).values,
                dft["_likes"].fillna(0).astype(str).values,
                dft["_replicas_proxy"].fillna(0).astype(str).values,
                dft["titulo"].astype(str).apply(lambda x: safe_text(x, 120)).values,
                dft["ente"].astype(str).apply(lambda x: safe_text(x, 80)).values,
                dft["teoria"].astype(str).apply(lambda x: safe_text(x, 80)).values,
                dft["conflito"].astype(str).apply(lambda x: safe_text(x, 80)).values,
            ], axis=1)

            fig.add_trace(go.Scatterpolar(
                theta=theta_jit,
                r=r_jit,
                mode="markers",
                marker=dict(
                    size=16,
                    color=cor,
                    opacity=0.92,
                    line=dict(width=1.5, color="white"),
                ),
                name=f"Eventos: {t}",
                customdata=customdata,
                hoverinfo="none",
            ))

    # Deriva no radar: linhas conectando posts consecutivos do mesmo autor
    if show_deriva_radar and "autor" in df_radar.columns:
        deriva_df = compute_deriva(df_radar)
        for autor_d, grp_d in df_radar.groupby("autor"):
            if len(grp_d) < 2:
                continue
            grp_d = grp_d.sort_values("_dt")
            cor_d = TIPO_COLOR.get(grp_d.iloc[-1]["_tipo"], "#888888")
            fig.add_trace(go.Scatterpolar(
                theta=np.degrees(grp_d["_theta"].values),
                r=grp_d["_r"].values,
                mode="lines",
                line=dict(color=cor_d, width=1.2, dash="dot"),
                opacity=0.5,
                name=f"Deriva: {autor_d}",
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        polar=dict(
            radialaxis=dict(
                range=[0, 1.05],
                tickmode="array",
                tickvals=bands,
                ticktext=["Faixa 1", "Faixa 2", "Faixa 3", "Faixa 4"],
                ticks="outside"
            ),
            angularaxis=dict(
                tickmode="array", tickvals=tickvals, ticktext=ticktext,
                direction="clockwise",
                rotation=90,
            )
        ),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # HTML com Plotly.js embutido + overlay JS para hover com imagem
    fig_html = fig.to_html(
        include_plotlyjs=True,
        full_html=False,
        div_id="radar-div",
        config={"displayModeBar": False, "responsive": True},
    )

    img_lookup_json = json.dumps(img_lookup)

    overlay_js = f"""
<style>
#hover-overlay {{
  display:none; position:fixed; z-index:9999;
  background:#fff; border:1px solid #ccc; border-radius:10px;
  box-shadow:0 4px 20px rgba(0,0,0,0.3); padding:12px;
  max-width:340px; pointer-events:none; font-family:sans-serif;
}}
#hover-overlay img {{ width:300px; border-radius:6px; display:block; margin-bottom:8px; }}
#hover-overlay .meta {{ font-size:12px; color:#333; line-height:1.6; }}
</style>
<div id="hover-overlay">
  <div id="hov-img"></div>
  <div class="meta" id="hov-meta"></div>
</div>
<script>
(function() {{
  var imgLookup = {img_lookup_json};
  var overlay = document.getElementById('hover-overlay');
  var hovImg  = document.getElementById('hov-img');
  var hovMeta = document.getElementById('hov-meta');

  function attachEvents() {{
    var gd = document.getElementById('radar-div');
    if (!gd || !gd._fullData) {{ setTimeout(attachEvents, 300); return; }}

    gd.on('plotly_hover', function(ev) {{
      var pt = ev.points[0];
      if (!pt || !pt.customdata) return;
      var cd = pt.customdata;
      var sid = cd[0];
      var b64 = imgLookup[sid];
      hovImg.innerHTML = b64
        ? '<img src="' + b64 + '">'
        : '<p style="color:#888;font-size:12px">Sem imagem</p>';
      hovMeta.innerHTML =
        '<b>ID:</b> '          + cd[0] + '<br>' +
        '<b>Data:</b> '        + cd[1] + '<br>' +
        '<b>Likes:</b> '       + cd[2] + '<br>' +
        '<b>Repercussão:</b> ' + cd[3] + '<br>' +
        '<b>Título:</b> '      + cd[4] + '<br>' +
        '<b>Ente:</b> '        + cd[5] + '<br>' +
        '<b>Teoria:</b> '      + cd[6];
      overlay.style.display = 'block';
    }});

    gd.on('plotly_unhover', function() {{
      overlay.style.display = 'none';
    }});

    gd.addEventListener('mousemove', function(e) {{
      var x = e.clientX + 18, y = e.clientY - 10;
      if (x + 360 > window.innerWidth)  x = e.clientX - 360;
      if (y + 420 > window.innerHeight) y = e.clientY - 420;
      overlay.style.left = x + 'px';
      overlay.style.top  = y + 'px';
    }});
  }}
  attachEvents();
}})();
</script>
"""

    full_html = "<div style='width:100%;height:750px'>" + fig_html + overlay_js + "</div>"
    components.html(full_html, height=760, scrolling=False)

    # ── Linha do tempo ────────────────────────────────────────────────────────
    # CSS esconde os tick-labels numéricos nativos do Streamlit (os índices
    # 0…N que aparecem abaixo da barra) e os substitui pelos marcadores de ano.
    st.markdown("""
<style>
[data-testid="stSlider"] [data-testid="stTickBar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

    # Labels de data flutuando sobre os handles (posição proporcional ao índice)
    _pct_lo = (_idx_lo / max(_n - 1, 1)) * 100
    _pct_hi = (_idx_hi / max(_n - 1, 1)) * 100
    _label_lo = _date_lo.strftime("%d/%m/%Y")
    _label_hi = _date_hi.strftime("%d/%m/%Y")

    st.markdown(f"""
<div style="position:relative; height:22px; margin-bottom:2px; margin-top:4px;">
  <span style="position:absolute; left:calc({_pct_lo:.1f}% - 36px);
    font-size:11px; color:#555; white-space:nowrap;
    background:#fff; padding:0 3px; border-radius:3px;">{_label_lo}</span>
  <span style="position:absolute; left:calc({_pct_hi:.1f}% - 36px);
    font-size:11px; color:#555; white-space:nowrap;
    background:#fff; padding:0 3px; border-radius:3px;">{_label_hi}</span>
</div>
""", unsafe_allow_html=True)

    st.slider(
        "Linha do tempo",
        min_value=0,
        max_value=_n - 1,
        key="timeline_range",
        label_visibility="collapsed",
    )

    # Relê os índices após o slider (podem ter mudado na interação)
    _idx_lo, _idx_hi = st.session_state["timeline_range"]
    _date_lo = _slider_dates[_idx_lo]
    _date_hi = _slider_dates[_idx_hi]

    # Marcadores de ano posicionados proporcionalmente abaixo da barra
    if len(_all_years) > 1:
        _yr_marks = []
        for _yr in _all_years:
            _i = next((i for i, d in enumerate(_slider_dates) if d.year == _yr), None)
            if _i is not None:
                _yr_marks.append((_yr, _i))

        _yr_html_items = ""
        for _yr, _i in _yr_marks:
            _pct = (_i / max(_n - 1, 1)) * 100
            _in_range = _idx_lo <= _i <= _idx_hi
            _color = "#1f77b4" if _in_range else "#bbb"
            _yr_html_items += f"""
  <span style="position:absolute; left:calc({_pct:.1f}% - 16px);
    font-size:11px; color:{_color}; white-space:nowrap; text-align:center;">
    {'▲' if _in_range else '▽'}<br>{_yr}
  </span>"""

        st.markdown(f"""
<div style="position:relative; height:32px; margin-top:4px;">
{_yr_html_items}
</div>
""", unsafe_allow_html=True)
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### 🖼 Evento selecionado")
    st.caption("Passe o mouse sobre um ponto para ver a imagem. Para metadados completos, selecione o ID:")

    all_ids = sorted(df["id"].astype(str).unique().tolist())
    cur = st.session_state.get("selected_id", None)
    cur_idx = all_ids.index(cur) + 1 if cur in all_ids else 0
    manual = st.selectbox("Selecione o ID:", ["—"] + all_ids, index=cur_idx, key="manual_id_select")
    if manual != "—":
        st.session_state["selected_id"] = manual

    sid = st.session_state.get("selected_id", None)
    if sid and sid != "—":
        row = df[df["id"].astype(str) == str(sid)].head(1)
        if not row.empty:
            r0 = row.iloc[0]
            img_path = find_image(image_folder, sid)
            cA, cB = st.columns([1.15, 1.0])
            with cA:
                if img_path is None:
                    st.warning(f"Imagem não encontrada: {image_folder.resolve()}/{sid}.(jpg|png|jpeg|webp)")
                else:
                    img = Image.open(img_path)
                    w_target = 600
                    w_percent = w_target / float(img.size[0])
                    h_target = int(img.size[1] * w_percent)
                    img = img.resize((w_target, h_target), Image.Resampling.LANCZOS)
                    st.image(img, caption=f"ID: {sid}", width="stretch")
            with cB:
                st.write(f"**ID:** {sid}")
                st.write(f"**Tipo:** {r0['_tipo']}")
                st.write(f"**Data/Hora:** {r0['_dt']}")
                st.write(f"**Autor:** {r0.get('autor','')}")
                st.write(f"**Likes:** {r0['_likes']}")
                st.write(f"**Seguidores:** {r0['_seguidores']}")
                st.write(f"**Repercussão (proxy):** {r0['_replicas_proxy']}")
                st.write(f"**Teoria:** {safe_text(r0.get('teoria', ''), 240)}")
                st.write(f"**Título:** {safe_text(r0.get('titulo', ''), 240)}")

with tab_painel:
    st.markdown("### Painel analítico (por tipo)")
    st.caption(
        f"Intervalo ativo: **{_date_lo.strftime('%d/%m/%Y')}** → **{_date_hi.strftime('%d/%m/%Y')}** "
        f"({len(df_radar)} registros)"
    )

    conc = (
        df_radar.groupby("_tipo")
        .apply(lambda d: circular_concentration(d["_theta"].values))
        .reset_index(name="concentracao_circular")
    )

    # Entropia angular por tipo
    entropia_por_tipo = (
        df_radar.groupby("_tipo")["_theta"]
        .apply(lambda s: compute_entropia_angular(s.values, n_bins=entropia_bins))
        .reset_index(name="entropia_angular")
    )

    resumo = (
        df_radar.groupby("_tipo")
        .agg(
            eventos=("_dt", "count"),
            iv_medio=("_iv_raw", "mean"),
            ie_medio=("_ie_raw", "mean"),
            ird_medio=("_ird_raw", "mean"),
            likes_total=("_likes", "sum"),
            repercussao_media=("_replicas_proxy", "mean"),
            int01_media=("_int01", "mean"),
            rep01_media=("_rep01", "mean"),
        )
        .reset_index()
        .merge(conc, on="_tipo", how="left")
        .merge(entropia_por_tipo, on="_tipo", how="left")
    )

    wsum = max(1e-9, (w_int + w_rep + w_conc))
    resumo["indice_sintetico_0a100"] = 100.0 * (
        (w_int / wsum) * resumo["int01_media"].fillna(0) +
        (w_rep / wsum) * resumo["rep01_media"].fillna(0) +
        (w_conc / wsum) * resumo["concentracao_circular"].fillna(0)
    )

    resumo_display = resumo.rename(columns={
        "_tipo": "tipo",
        "iv_medio": "IV médio",
        "ie_medio": "IE médio (%)",
        "ird_medio": f"IRD médio (α={ird_alpha})",
        "likes_total": "likes total",
        "repercussao_media": f"repercussão ({_rep_label})",
        "concentracao_circular": "concentração (R̄)",
        "entropia_angular": "entropia angular (H)",
        "indice_sintetico_0a100": "índice sintético (0–100)",
    })

    st.caption(f"Métrica ativa: **{_rep_label}**  |  λ={decay_lambda}  |  bins entropia={entropia_bins}")
    st.dataframe(resumo_display.sort_values("eventos", ascending=False), width="stretch")

    # gráficos comparativos
    col_b1, col_b2, col_b3 = st.columns(3)
    CMAP = {"infra":"#e74c3c","supra":"#9b59b6","quase":"#f39c12","outros":"#2ecc71"}
    with col_b1:
        fig_iv = px.bar(resumo, x="_tipo", y="iv_medio", color="_tipo",
            title="IV médio (viralização)", labels={"_tipo":"tipo","iv_medio":"IV"},
            color_discrete_map=CMAP)
        fig_iv.update_layout(showlegend=False, height=280)
        st.plotly_chart(fig_iv, key="bar_iv")
    with col_b2:
        fig_ent = px.bar(resumo, x="_tipo", y="entropia_angular", color="_tipo",
            title="Entropia angular (H)", labels={"_tipo":"tipo","entropia_angular":"H (0–1)"},
            color_discrete_map=CMAP)
        fig_ent.update_layout(showlegend=False, height=280, yaxis_range=[0,1])
        st.plotly_chart(fig_ent, key="bar_ent")
    with col_b3:
        fig_conc = px.bar(resumo, x="_tipo", y="concentracao_circular", color="_tipo",
            title="Concentração temporal (R̄)", labels={"_tipo":"tipo","concentracao_circular":"R̄"},
            color_discrete_map=CMAP)
        fig_conc.update_layout(showlegend=False, height=280, yaxis_range=[0,1])
        st.plotly_chart(fig_conc, key="bar_conc")

    st.download_button("⬇️ Baixar resumo por tipo (CSV)",
        resumo_display.to_csv(index=False).encode("utf-8"), "resumo_por_tipo.csv", "text/csv")

    # ── dicionário de métricas ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📖 Dicionário de métricas")
    st.markdown("""
#### Métricas de engajamento

**IV — Índice de Viralização**
- **Fórmula:** `likes / √seguidores`
- **Operacionalização:** divide likes pela raiz quadrada dos seguidores. A raiz quadrada corrige o viés de escala das redes sociais (lei de potência): uma conta 4× maior não produz 4× mais engajamento esperado, mas ~2× (Bakshy et al., 2012).
- **Interpretação:** valores altos = repercussão desproporcional ao tamanho da audiência. Compare relativamente entre tipos e eventos, não em termos absolutos.

---

**IE — Índice de Engajamento Composto**
- **Fórmula:** `(likes + comentários + compartilhamentos) / seguidores × 100`
- **Operacionalização:** soma todas as formas de interação e divide pelo alcance potencial. Resultado em percentual.
- **Interpretação:** taxas >3–5% são consideradas altas em redes sociais (Hootsuite, 2023). Valores muito altos (>20%) podem indicar campanhas coordenadas ou nichos muito engajados.

---

**IA — Índice de Alcance Ponderado**
- **Fórmula:** `visualizações × IV_normalizado`
- **Operacionalização:** combina alcance bruto (visualizações) com qualidade do engajamento (IV normalizado 0–1 por min-max robusto).
- **Interpretação:** alto IA = simultaneamente muito visto e muito viral. Requer coluna `visualizacoes` no CSV.

---

**IRD — Índice de Ressonância Desproporcional**
- **Fórmula:** `likes / seguidores^α`  (α ajustável no sidebar, padrão 0.5)
- **Operacionalização:** generaliza o IV com expoente α configurável. α=0.5 equivale ao IV. Valores de α menores (ex: 0.3) penalizam mais fortemente contas grandes, favorecendo a detecção de viralidade em contas pequenas — útil para identificar discursos que ressoam organicamente em nichos antes de atingir grandes influenciadores.
- **Interpretação:** IRD alto em conta pequena = o estereótipo ressoa organicamente na base; IRD alto em conta grande = amplificação institucionalizada. Compare os dois perfis para distinguir origem da viralidade de sua amplificação.

---

#### Métricas temporais

**Concentração temporal (R̄ — comprimento do vetor médio circular)**
- **Fórmula:** `√(mean(cos θ)² + mean(sin θ)²)`
- **Operacionalização:** estatística circular de Fisher (1993). θ é o ângulo do post no ciclo escolhido (dia/semana/mês/ano).
- **Interpretação:** R̄ → 0 = discursos distribuídos uniformemente no ciclo (sem padrão temporal). R̄ → 1 = forte concentração num mesmo período. Valores >0.5 merecem atenção interpretativa.

---

**Entropia Angular (H)**
- **Fórmula:** `−Σ pₖ log(pₖ) / log(n_bins)` sobre bins angulares do ciclo
- **Operacionalização:** divide o ciclo em n_bins fatias iguais, calcula a distribuição dos posts por fatia e aplica entropia de Shannon, normalizada pelo máximo teórico (log n_bins) para escala 0–1.
- **Interpretação:** H → 1 = distribuição uniforme ao longo do ciclo → padrão **orgânico** (posts em qualquer horário/dia). H → 0 = concentração extrema em poucos bins → padrão **coordenado** ou reativo (posts em horários/dias específicos). Use em conjunto com R̄: alta concentração (R̄↑) + baixa entropia (H↓) é o sinal mais forte de coordenação.

---

#### Deriva por autor

**Δθ — Deriva angular**
- **Fórmula:** diferença circular entre ângulos de posts consecutivos do mesmo autor, normalizada para [−180°, +180°]
- **Interpretação:** Δθ ≈ 0 = autor mantém o mesmo padrão horário/semanal. |Δθ| grande = mudança de período de publicação entre posts — pode indicar adaptação de estratégia ou perfil de conta com uso irregular.

**Δr — Deriva radial (escalada de intensidade)**
- **Fórmula:** diferença de raio entre posts consecutivos (r₂ − r₁)
- **Interpretação:** Δr < 0 = post mais recente está mais próximo do centro = **escalada** (maior engajamento). Δr > 0 = desescalada. Autores com Δr sistematicamente negativo ao longo do tempo apresentam **trajetória de escalada** — intensificação progressiva do discurso de desumanização.

**Mudança de tipo**
- Indica se o autor transitou entre categorias (ex: de quase-desumanização para infra-humanização). Transições para tipos mais extremos (quase → infra ou supra) são teoricamente relevantes como indicadores de radicalização discursiva.

---

#### Índice Sintético (0–100)
- **Fórmula:** `100 × (w₁·intensidade + w₂·repercussão + w₃·concentração) / (w₁+w₂+w₃)`
- **Interpretação:** instrumento de **comparação relativa** dentro do corpus — não é uma medida absoluta de desumanização. Reporte sempre os pesos utilizados e realize análise de sensibilidade variando-os.

---

#### Decaimento temporal (λ)
- **Fórmula:** `métrica × e^(−λ × dias_desde_postagem)`
- **Meia-vida:** λ=0.01 → ~70 dias | λ=0.02 → ~35 dias | λ=0.03 → ~23 dias
- **Recomendação:** ative quando o corpus abrange >3 meses ou quando há suspeita de viés de recência.
""")

with tab_galeria:
    st.markdown("### Galeria (fallback)")

    g1, g2 = st.columns([1.2, 1.2])
    with g1:
        gal_tipo = st.selectbox("Tipo", ["(todos)"] + ["infra", "supra", "quase", "outros"], index=0)
    with g2:
        n_cols = st.slider("Colunas na grade", 2, 6, 4, 1)

    dfg = df.copy()
    if gal_tipo != "(todos)":
        dfg = dfg[dfg["_tipo"] == gal_tipo]

    dfg = dfg.sort_values(["_replicas_proxy", "_likes"], ascending=False)
    st.caption(f"Itens no recorte: {len(dfg)}")

    rows = dfg.to_dict("records")
    grid_cols = st.columns(n_cols)

    for i, r in enumerate(rows[: min(len(rows), 160)]):
        col = grid_cols[i % n_cols]
        sid = str(r["id"])
        img_path = find_image(image_folder, sid)
        caption = f"{sid} | {r['_tipo']}"

        with col:
            if img_path:
                try:
                    img = Image.open(img_path)
                    st.image(img, caption=caption, width="stretch")
                except Exception:
                    st.write("🖼️ (imagem inválida)")
                    st.caption(caption)
            else:
                st.write("🖼️ (sem imagem)")
                st.caption(caption)

            if st.button("Ver", key=f"gal_{sid}"):
                st.session_state["selected_id"] = sid
                st.rerun()


with tab_classif:
    st.markdown("### 🤖 Classificação Assistida por LLM")
    st.markdown("Usa um modelo de linguagem para sugerir a classificação. Você revisa e aprova cada sugestão.")

    backend = st.radio("Backend:",
        ["🖥️  Ollama (local, sem internet)", "☁️  Groq (nuvem, modelos open-source, gratuito)"],
        horizontal=True, key="llm_backend")
    use_groq = backend.startswith("☁️")

    with st.expander("⚙️ Configuração", expanded=True):
        if use_groq:
            st.markdown("Crie sua chave em [console.groq.com](https://console.groq.com) → **API Keys → Create API Key**.")
            col1, col2 = st.columns([2, 1])
            with col1:
                groq_key = st.text_input("API Key do Groq (gsk_...)", type="password", key="groq_api_key")
            with col2:
                groq_model = st.selectbox("Modelo",
                    ["llama-3.3-70b-versatile","llama-3.1-8b-instant","gemma2-9b-it"],
                    key="groq_model", help="llama-3.3-70b-versatile é o mais recomendado.")
            if st.button("Testar conexão Groq", key="test_groq"):
                if not groq_key:
                    st.warning("Insira a API Key primeiro.")
                else:
                    test = classify_with_groq("Teste.", "teste", "", "", groq_key, groq_model)
                    st.success(f"✅ Groq conectado! `{groq_model}` respondendo.") if test["tipo"] else st.error(test["justificativa"])
        else:
            st.markdown("Instale em [ollama.com](https://ollama.com), rode `ollama pull llama3` e `ollama serve`.")
            col1, col2 = st.columns([2, 1])
            with col1:
                ollama_url = st.text_input("URL", value="http://localhost:11434", key="ollama_url")
            with col2:
                ollama_model = st.text_input("Modelo", value="llama3", key="ollama_model")
            if st.button("Testar conexão Ollama", key="test_ollama"):
                try:
                    r = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
                    models_av = [m["name"] for m in r.json().get("models",[])]
                    st.success(f"✅ Modelos: {', '.join(models_av)}") if models_av else st.warning("Sem modelos. Rode: `ollama pull llama3`")
                except Exception:
                    st.error("Não foi possível conectar.")

    st.divider()
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        apenas_outros = st.checkbox("Mostrar apenas 'outros'", value=True)
    with col_f2:
        ocultar_revisados = st.checkbox("Ocultar já revisados", value=True)

    if "revisados" not in st.session_state:
        st.session_state["revisados"] = {}

    df_cl = df.copy()
    if apenas_outros: df_cl = df_cl[df_cl["_tipo"] == "outros"]
    if ocultar_revisados: df_cl = df_cl[~df_cl["id"].astype(str).isin(st.session_state["revisados"].keys())]

    st.caption(f"{len(df_cl)} registro(s) para revisar")

    if df_cl.empty:
        st.success("Todos os registros filtrados já foram revisados nesta sessão.")
    else:
        id_sel = st.selectbox("Selecione o registro:", df_cl["id"].astype(str).tolist(), key="classif_id_sel")
        row_cl = df_cl[df_cl["id"].astype(str) == id_sel].iloc[0]

        cL, cR = st.columns([1.2, 1.0])
        with cL:
            st.markdown("#### Discurso")
            st.write(f"**ID:** {row_cl['id']}  |  **Tipo atual:** `{row_cl['_tipo']}`")
            st.write(f"**Título:** {safe_text(row_cl.get('titulo',''), 200)}")
            st.write(f"**Autor:** {row_cl.get('autor','')}  |  **Data:** {row_cl['_dt']}")
            st.write(f"**Categoria:** {row_cl.get('categoria','')}  |  **Teoria:** {row_cl.get('teoria','')}")
            st.text_area("Conteúdo", value=str(row_cl.get("conteudo","")), height=180, disabled=True, label_visibility="visible")
            img_p = find_image(image_folder, str(row_cl["id"]))
            if img_p: st.image(Image.open(img_p), width="stretch")

        with cR:
            st.markdown("#### Sugestão do modelo")
            btn_label = f"🔍 Classificar com {'Groq' if use_groq else 'Ollama'}"
            if st.button(btn_label, key="btn_classify"):
                pronto = True
                if use_groq and not st.session_state.get("groq_api_key","").strip():
                    st.error("Insira a API Key do Groq."); pronto = False
                if pronto:
                    with st.spinner("Consultando modelo..."):
                        resultado = (classify_with_groq if use_groq else classify_with_ollama)(
                            texto=str(row_cl.get("conteudo","")),
                            titulo=str(row_cl.get("titulo","")),
                            teoria=str(row_cl.get("teoria","")),
                            categoria=str(row_cl.get("categoria","")),
                            **({"api_key": st.session_state["groq_api_key"],
                                "model": st.session_state["groq_model"]} if use_groq else
                               {"url": st.session_state.get("ollama_url","http://localhost:11434"),
                                "model": st.session_state.get("ollama_model","llama3")})
                        )
                    st.session_state[f"sugestao_{id_sel}"] = resultado

            sugestao = st.session_state.get(f"sugestao_{id_sel}")
            if sugestao:
                if sugestao["tipo"] is None:
                    st.error(sugestao["justificativa"])
                else:
                    conf_icon = {"alta":"🟢","media":"🟡","baixa":"🔴"}.get(sugestao["confianca"],"⚪")
                    st.markdown(f"**Sugerido:** `{sugestao['tipo']}` {conf_icon} confiança **{sugestao['confianca']}**")
                    st.info(sugestao["justificativa"])
                    st.markdown("#### Sua decisão")
                    tipo_final = st.radio("Confirmar ou corrigir:",
                        ["infra","supra","quase","outros"],
                        index=["infra","supra","quase","outros"].index(sugestao["tipo"]),
                        horizontal=True, key=f"radio_{id_sel}")
                    nota = st.text_input("Nota do revisor (opcional):", key=f"nota_{id_sel}")
                    if st.button("✅ Salvar decisão", key=f"salvar_{id_sel}"):
                        st.session_state["revisados"][id_sel] = {
                            "backend": "groq" if use_groq else "ollama",
                            "modelo": st.session_state.get("groq_model" if use_groq else "ollama_model",""),
                            "tipo_sugerido": sugestao["tipo"], "tipo_final": tipo_final,
                            "justificativa_modelo": sugestao["justificativa"],
                            "confianca": sugestao["confianca"], "nota_revisor": nota,
                        }
                        st.success(f"Salvo: ID {id_sel} → `{tipo_final}`")
                        st.rerun()
            else:
                st.caption(f"Clique em '{btn_label}' para obter a sugestão.")

    st.divider()
    st.markdown("#### 📥 Exportar revisões")
    revisados = st.session_state.get("revisados", {})
    st.caption(f"{len(revisados)} registro(s) revisado(s) nesta sessão")
    if revisados:
        df_rev = pd.DataFrame([{"id": k, **v} for k, v in revisados.items()])
        st.dataframe(df_rev, width="stretch")
        df_export = df0.copy()
        for id_rev, dec in revisados.items():
            mask = df_export["id"].astype(str) == id_rev
            df_export.loc[mask, "categoria"] = dec["tipo_final"]
        st.download_button("⬇️ CSV com classificações revisadas",
            data=df_export.to_csv(index=False, sep=sep).encode("utf-8"),
            file_name="dataset_revisado.csv", mime="text/csv")
        st.download_button("⬇️ Log de revisões (auditoria)",
            data=df_rev.to_csv(index=False).encode("utf-8"),
            file_name="log_revisoes.csv", mime="text/csv")

st.caption(
    "Se você voltar a ver poucos registros, isso geralmente é filtro de tipologia. "
    "Use o multiselect para incluir 'outros' ou ajuste o mapeamento em normalize_tipo()."
)