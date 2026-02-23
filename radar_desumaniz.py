# radar_desumaniz.py
# VERSÃO INTEGRAL RESTAURADA - Correção de ValueError e Exibição 600px

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
    if cycle == "Dia":
        seconds = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
        frac = seconds / 86400.0
        return 2 * np.pi * frac
    if cycle == "Semana":
        frac = (dt.dt.dayofweek + (dt.dt.hour / 24.0)) / 7.0
        return 2 * np.pi * frac
    if cycle == "Mês":
        dim = dt.dt.days_in_month.astype(float)
        frac = ((dt.dt.day - 1) + (dt.dt.hour / 24.0)) / dim
        return 2 * np.pi * frac
    if cycle == "Ano":
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
    if cycle == "Dia":
        tickvals = [i * 360/24 for i in range(0, 24, 3)]
        ticktext = [f"{i:02d}h" for i in range(0, 24, 3)]
        label = "hora"
        unit_values = list(range(24))
        return tickvals, ticktext, label, unit_values
    if cycle == "Semana":
        tickvals = [i * 360/7 for i in range(7)]
        ticktext = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
        label = "dia_semana"
        unit_values = list(range(7))
        return tickvals, ticktext, label, unit_values
    if cycle == "Mês":
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
st.title("Radar crítico dos discursos: infra/supra/quase (4 faixas fixas)")

if "selected_id" not in st.session_state:
    st.session_state["selected_id"] = None

with st.sidebar:
    st.header("Dados")
    up = st.file_uploader("Envie o CSV", type=["csv"])
    sep = st.selectbox("Separador", [",", ";", "\t"], index=0)
    enc = st.selectbox("Encoding", ["utf-8", "latin-1", "utf-8-sig"], index=0)

    st.divider()
    st.header("Imagens")
    image_folder = Path(st.text_input("Pasta das imagens", value="imagens"))

    st.divider()
    st.header("Ciclo do radar")
    cycle = st.selectbox("Ciclo", ["Dia", "Semana", "Mês", "Ano"], index=0)

    st.divider()
    st.header("Radar")
    inner = st.slider("Centro (raio mínimo)", 0.02, 0.30, 0.10, 0.01)

    st.subheader("Manchas (réplicas)")
    density = st.slider("Densidade", 0.5, 2.0, 1.0, 0.1)
    jitter_theta = st.slider("Dispersão angular", 0.01, 0.20, 0.06, 0.01)
    jitter_r = st.slider("Dispersão radial", 0.005, 0.10, 0.035, 0.005)

    st.divider()
    st.header("Índice sintético")
    w_int = st.slider("Peso: intensidade", 0.0, 1.0, 0.40, 0.05)
    w_rep = st.slider("Peso: repercussão", 0.0, 1.0, 0.40, 0.05)
    w_conc = st.slider("Peso: concentração", 0.0, 1.0, 0.20, 0.05)

    st.divider()
    st.header("Heatmap polar")
    heat_metric = st.selectbox("Métrica", ["eventos", "replicas_total", "intensidade_media"], index=1)

    st.divider()
    st.header("Exibição")
    show_cloud = st.checkbox("Mostrar manchas", True)
    show_points = st.checkbox("Mostrar pontos-origem", True)

if up is None:
    st.info("Envie o CSV para iniciar.")
    st.stop()

df0 = pd.read_csv(up, sep=sep, encoding=enc, dtype=str, engine="python").dropna(how="all").copy()

required_cols = ["id", "data", "autor", "seguidores", "titulo", "likes", "conteudo",
                 "replicas", "categoria", "ente", "teoria", "conflito", "imagem"]
missing = [c for c in required_cols if c not in df0.columns]
if missing:
    st.error("Colunas faltando no CSV: " + ", ".join(missing) + "\n\nVerifique separador/encoding.")
    st.stop()

st.subheader("Prévia do CSV")
st.dataframe(df0.head(12), width="stretch")

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

df["_likes"] = df["likes"].apply(to_number)
df["_seguidores"] = df["seguidores"].apply(to_number)

df["_replicas_proxy"] = df["replicas"].fillna("").astype(str).str.len().replace(0, np.nan)

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

df["_int_raw"] = np.log1p(df["_likes"].clip(lower=0)) - np.log1p(df["_seguidores"].clip(lower=0))
df["_int01"] = robust_minmax(df["_int_raw"])
df["_rep01"] = robust_minmax(df["_replicas_proxy"].clip(lower=0))

df["_theta"] = angle_from_cycle(df["_dt"], cycle)
df["_r"] = radial_from_intensity(df["_int01"], inner=inner, outer=1.0)

tickvals, ticktext, ciclo_label, unit_values = cycle_bins(cycle)

tab_radar, tab_painel, tab_galeria = st.tabs(["Radar", "Painel Analítico", "Galeria Interativa"])

with tab_radar:
    st.markdown("### Radar (seleção → imagem do `id` + metadados)")

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

    # Lookup base64 das imagens
    img_lookup = {}
    for _, row in df.iterrows():
        sid_i = str(row["id"])
        ip = find_image(image_folder, sid_i)
        if ip:
            img_lookup[sid_i] = image_to_base64(ip, max_width=300)

    np.random.seed(7)
    if show_cloud:
        for t in ["infra", "supra", "quase", "outros"]:
            dft = df[df["_tipo"] == t]
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
            dft = df[df["_tipo"] == t]
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
            angularaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext)
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

    conc = (
        df.groupby("_tipo")
        .apply(lambda d: circular_concentration(d["_theta"].values))
        .reset_index(name="concentracao_circular")
    )

    resumo = (
        df.groupby("_tipo")
        .agg(
            eventos=("_dt", "count"),
            intensidade_media=("_int_raw", "mean"),
            replicas_proxy_total=("_replicas_proxy", "sum"),
            int01_media=("_int01", "mean"),
            rep01_media=("_rep01", "mean"),
        )
        .reset_index()
        .merge(conc, on="_tipo", how="left")
    )

    wsum = max(1e-9, (w_int + w_rep + w_conc))
    resumo["indice_sintetico_0a100"] = 100.0 * (
        (w_int / wsum) * resumo["int01_media"].fillna(0) +
        (w_rep / wsum) * resumo["rep01_media"].fillna(0) +
        (w_conc / wsum) * resumo["concentracao_circular"].fillna(0)
    )

    st.dataframe(resumo.sort_values("eventos", ascending=False), width="stretch")
    st.download_button("Baixar resumo (CSV)", resumo.to_csv(index=False).encode("utf-8"), "resumo_por_tipo.csv", "text/csv")

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

st.caption(
    "Se você voltar a ver poucos registros, isso geralmente é filtro de tipologia. "
    "Use o multiselect para incluir 'outros' ou ajuste o mapeamento em normalize_tipo()."
)