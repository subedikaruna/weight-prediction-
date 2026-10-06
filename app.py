import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import goat_ml as gm
from generate_dataset import make_dataset

st.set_page_config(page_title="Goat Weight Predictor", page_icon="🐐", layout="wide")

COL = {"Normal": "#4ade80", "Underweight": "#fb7185", "Overweight": "#fbbf24"}
ORDER = ["Underweight", "Normal", "Overweight"]

# ======================= STYLE =======================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
@property --n { syntax: '<integer>'; inherits: false; initial-value: 0; }

html, body, [class*="css"], .stApp { font-family: 'Space Grotesk', sans-serif; }
.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, #123d2e 0%, transparent 60%),
              radial-gradient(900px 500px at 100% 0%, #3b2a10 0%, transparent 55%),
              #070d0b;
  background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1300px; }

/* sidebar */
section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.03); backdrop-filter: blur(16px);
  border-right: 1px solid rgba(255,255,255,0.08);
}

/* tabs + page transition */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(255,255,255,0.04);
  padding: 6px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 18px; transition: all .25s ease; }
.stTabs [data-baseweb="tab"]:hover { background: rgba(124,255,178,0.10); transform: translateY(-2px); }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg,#16a34a,#65a30d) !important; color: #fff !important;
  box-shadow: 0 6px 24px rgba(74,222,128,0.35); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
[data-baseweb="tab-panel"] { animation: panelIn .6s cubic-bezier(.2,.8,.2,1) both; }
@keyframes panelIn { from { opacity: 0; transform: perspective(900px) translateY(24px) rotateX(6deg); }
                     to   { opacity: 1; transform: none; } }

/* buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 12px; border: 1px solid rgba(255,255,255,0.15);
  background: rgba(255,255,255,0.06); transition: all .25s ease; }
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-3px) scale(1.02); box-shadow: 0 10px 30px rgba(74,222,128,0.25); border-color: #4ade80; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg,#22c55e,#a3e635); color: #052e16; font-weight: 700;
  animation: pulse 2.6s infinite; }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(74,222,128,.45); } 50% { box-shadow: 0 0 0 14px rgba(74,222,128,0); } }

/* kpi cards (3D tilt) */
.kgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 16px; margin: 6px 0 18px; perspective: 1000px; }
.kpi { position: relative; padding: 18px 20px; border-radius: 18px; overflow: hidden;
  background: linear-gradient(145deg, rgba(255,255,255,0.09), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.12); backdrop-filter: blur(14px);
  transform-style: preserve-3d; transition: transform .35s ease, box-shadow .35s ease;
  animation: kin .7s cubic-bezier(.2,.8,.2,1) both; animation-delay: var(--d, 0s); }
.kpi::before { content: ""; position: absolute; inset: -40% -20% auto auto; width: 140px; height: 140px; border-radius: 50%;
  background: radial-gradient(circle, var(--c, #4ade80) 0%, transparent 70%); opacity: .25; }
.kpi:hover { transform: rotateX(8deg) rotateY(-10deg) translateZ(14px); box-shadow: 0 24px 50px rgba(0,0,0,.45), 0 0 30px color-mix(in srgb, var(--c, #4ade80) 35%, transparent); }
.kpi .ico { font-size: 26px; transform: translateZ(30px); display: inline-block; animation: bob 3.5s ease-in-out infinite; }
.kpi .lbl { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; opacity: .65; margin-top: 6px; }
.kpi .val { font-size: 34px; font-weight: 700; line-height: 1.15; transform: translateZ(20px); color: var(--c, #4ade80); }
.kpi .sub { font-size: 12px; opacity: .6; }
.count { --n: var(--t); counter-reset: num var(--n); animation: cnt 1.4s ease-out both; }
.count::after { content: counter(num); }
@keyframes cnt { from { --n: 0; } }
@keyframes kin { from { opacity: 0; transform: perspective(800px) rotateX(-25deg) translateY(30px); } to { opacity: 1; transform: none; } }
@keyframes bob { 0%,100% { transform: translateZ(30px) translateY(0); } 50% { transform: translateZ(30px) translateY(-5px); } }

/* status pill */
.pill { display: inline-flex; align-items: center; gap: 10px; padding: 10px 18px; margin: 4px 0 10px; border-radius: 999px;
  background: color-mix(in srgb, var(--c) 14%, transparent); border: 1px solid var(--c); animation: kin .6s both; }
.pill .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--c); animation: blink 1.4s infinite; }
@keyframes blink { 50% { box-shadow: 0 0 0 7px transparent; } 0% { box-shadow: 0 0 0 0 var(--c); } }

.sect { font-size: 20px; font-weight: 700; margin: 22px 0 8px; display: flex; align-items: center; gap: 10px; }
.sect::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(255,255,255,.25), transparent); }

[data-testid="stDataFrame"], .stPlotlyChart { border-radius: 16px; overflow: hidden; animation: kin .7s both; }
.stPlotlyChart { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ======================= 3D HERO (three.js) =======================
HERO = """
<style>
  html,body{margin:0;background:transparent;overflow:hidden;font-family:'Space Grotesk',sans-serif}
  #wrap{position:relative;height:270px;border-radius:22px;overflow:hidden;
        background:linear-gradient(120deg,rgba(22,163,74,.25),rgba(250,204,21,.10));
        border:1px solid rgba(255,255,255,.14)}
  canvas{position:absolute;inset:0}
  .t{position:absolute;left:36px;top:50%;transform:translateY(-50%);color:#fff;z-index:2;animation:in 1s both}
  h1{margin:0;font-size:44px;line-height:1.05;background:linear-gradient(90deg,#fff,#a3e635,#4ade80);
     -webkit-background-clip:text;color:transparent}
  p{margin:10px 0 0;color:#cbd5e1;font-size:15px;max-width:420px}
  @keyframes in{from{opacity:0;transform:translate(-30px,-50%)}to{opacity:1;transform:translate(0,-50%)}}
</style>
<div id="wrap"><div class="t"><h1>Goat Weight<br>Predictor</h1>
<p>Train on your own herd. Forecast weight, spot under and overweight goats early.</p></div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const wrap=document.getElementById('wrap');
const W=()=>wrap.clientWidth,H=()=>wrap.clientHeight;
const scene=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(45,W()/H(),.1,100);cam.position.set(0,1.2,9);
const r=new THREE.WebGLRenderer({antialias:true,alpha:true});r.setSize(W(),H());r.setPixelRatio(Math.min(devicePixelRatio,2));
wrap.appendChild(r.domElement);
scene.add(new THREE.AmbientLight(0xffffff,.6));
const dl=new THREE.DirectionalLight(0xffffff,.9);dl.position.set(4,6,5);scene.add(dl);
const pl=new THREE.PointLight(0x4ade80,1.4,30);pl.position.set(-4,2,3);scene.add(pl);
const mat=(c)=>new THREE.MeshStandardMaterial({color:c,flatShading:true,roughness:.7});
const goat=new THREE.Group();
const add=(g,m,x,y,z,rx=0,ry=0,rz=0)=>{const o=new THREE.Mesh(g,m);o.position.set(x,y,z);o.rotation.set(rx,ry,rz);goat.add(o);return o};
const fur=mat(0xf5f5f4),dark=mat(0x78350f),horn=mat(0xd6b370);
add(new THREE.BoxGeometry(2.2,1.1,1),fur,0,0,0);
add(new THREE.BoxGeometry(.7,1,.6),fur,1.15,.5,0,0,0,-.5);
add(new THREE.BoxGeometry(.75,.55,.5),fur,1.6,.95,0);
add(new THREE.BoxGeometry(.3,.3,.35),dark,1.95,.85,0);
add(new THREE.ConeGeometry(.08,.6,5),horn,1.45,1.45,.15,0,0,.5);
add(new THREE.ConeGeometry(.08,.6,5),horn,1.45,1.45,-.15,0,0,.5);
add(new THREE.ConeGeometry(.12,.35,4),dark,1.35,1.1,.32,1.2,0,0);
add(new THREE.ConeGeometry(.12,.35,4),dark,1.35,1.1,-.32,-1.2,0,0);
add(new THREE.ConeGeometry(.08,.3,4),fur,1.85,.55,0,0,0,3.3);
add(new THREE.ConeGeometry(.12,.4,4),fur,-1.25,.35,0,0,0,1.2);
const legs=[];
[[.75,.3],[.75,-.3],[-.75,.3],[-.75,-.3]].forEach(p=>legs.push(add(new THREE.CylinderGeometry(.11,.09,1,6),dark,p[0],-1,p[1])));
goat.position.y=.4;scene.add(goat);
const ring=new THREE.Mesh(new THREE.TorusGeometry(3.2,.025,8,80),new THREE.MeshBasicMaterial({color:0x4ade80,transparent:true,opacity:.6}));
ring.rotation.x=Math.PI/2;ring.position.y=-1.4;scene.add(ring);
const ring2=ring.clone();ring2.scale.set(1.3,1.3,1.3);ring2.material=ring.material.clone();ring2.material.opacity=.25;scene.add(ring2);
const N=260,pos=new Float32Array(N*3);for(let i=0;i<N*3;i++)pos[i]=(Math.random()-.5)*20;
const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.BufferAttribute(pos,3));
const pts=new THREE.Points(pg,new THREE.PointsMaterial({color:0xa3e635,size:.05}));scene.add(pts);
goat.position.x=2.6;ring.position.x=2.6;ring2.position.x=2.6;
let mx=0,my=0;addEventListener('mousemove',e=>{mx=(e.clientX/innerWidth-.5);my=(e.clientY/innerHeight-.5)});
const clock=new THREE.Clock();
(function loop(){requestAnimationFrame(loop);const t=clock.getElapsedTime();
 goat.rotation.y=t*.5+mx*1.2;goat.rotation.x=my*.3;goat.position.y=.4+Math.sin(t*1.6)*.15;
 legs.forEach((l,i)=>l.rotation.z=Math.sin(t*3+i*1.6)*.12);
 ring.rotation.z=t*.4;ring2.rotation.z=-t*.25;pts.rotation.y=t*.03;pts.rotation.x=my*.1;
 pl.position.x=Math.sin(t)*5;r.render(scene,cam)})();
addEventListener('resize',()=>{r.setSize(W(),H());cam.aspect=W()/H();cam.updateProjectionMatrix()});
</script>
"""
components.html(HERO, height=290)

# ======================= HELPERS =======================
TEMPLATE = pd.DataFrame({
    "goat_id": ["G001", "G001", "G002"],
    "breed": ["Sirohi", "Sirohi", "Boer"],
    "sex": ["female", "female", "male"],
    "age_months": [6, 9, 8],
    "height_cm": [52.0, 57.5, 60.0],
    "chest_girth_cm": [58.0, 64.0, 66.0],
    "body_length_cm": [50.0, 55.0, 58.0],
    "weight_kg": [16.5, 21.0, 27.0],
})


def read_any(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def section(text):
    st.markdown(f'<div class="sect">{text}</div>', unsafe_allow_html=True)


def kpi_grid(items):
    """items: (icon, label, value, sub, color, is_int)"""
    html = '<div class="kgrid">'
    for i, (ico, lbl, val, sub, color, is_int) in enumerate(items):
        v = f'<div class="val count" style="--t:{int(val)}"></div>' if is_int else f'<div class="val">{val}</div>'
        html += (f'<div class="kpi" style="--c:{color};--d:{i * 0.1:.1f}s"><span class="ico">{ico}</span>'
                 f'<div class="lbl">{lbl}</div>{v}<div class="sub">{sub}</div></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)


def pill(label, status):
    c = COL[status]
    st.markdown(f'<div class="pill" style="--c:{c}"><span class="dot"></span>{label}: <b>{status}</b></div>',
                unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Space Grotesk"),
                      legend=dict(orientation="h", y=-0.15), transition=dict(duration=600, easing="cubic-in-out"))
    return fig


def scatter3d(df):
    third = next((c for c in ["chest_girth_cm", "height_cm", "body_length_cm"]
                  if c in df.columns and df[c].notna().sum() >= 5), None)
    if third is None:
        fig = px.scatter(df, x="age_months", y="weight_kg", color="breed", opacity=0.8)
        return style(fig, 460)
    d = df.dropna(subset=[third])
    fig = px.scatter_3d(d, x="age_months", y=third, z="weight_kg", color="breed", opacity=0.85)
    fig.update_traces(marker=dict(size=3.5))
    fig.update_layout(scene=dict(xaxis_backgroundcolor="rgba(0,0,0,0)", yaxis_backgroundcolor="rgba(0,0,0,0)",
                                 zaxis_backgroundcolor="rgba(0,0,0,0)"))
    frames = []
    for a in np.linspace(0, 2 * np.pi, 60):
        frames.append(go.Frame(layout=dict(scene_camera=dict(eye=dict(x=1.9 * np.cos(a), y=1.9 * np.sin(a), z=0.9)))))
    fig.frames = frames
    fig.update_layout(updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.98, buttons=[
        dict(label="▶ Spin", method="animate",
             args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True, transition=dict(duration=0), mode="immediate")]),
        dict(label="⏸", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])])
    return style(fig, 520)


# ======================= SIDEBAR =======================
with st.sidebar:
    st.markdown("## 🐐 Control panel")
    st.header("1. Data")
    source = st.radio("Data source", ["Sample dataset", "Upload my data"])
    raw = None
    if source == "Sample dataset":
        raw = make_dataset()
    else:
        up = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
        if up is not None:
            try:
                raw = read_any(up)
            except Exception as e:
                st.error(f"Cannot read file: {e}")
    st.download_button("Download blank template (CSV)", to_csv_bytes(TEMPLATE), "goat_template.csv", "text/csv")
    st.markdown(
        "**Required columns:** `breed`, `sex`, `age_months`, `weight_kg`  \n"
        "**Optional:** `goat_id`, `height_cm`, `chest_girth_cm`, `body_length_cm`  \n"
        "Repeat `goat_id` over several dates to enable growth forecasting."
    )
    use_age = st.checkbox("Use age in weight estimator", value=True)

if raw is None:
    st.info("Upload a file in the sidebar, or switch to the sample dataset.")
    st.stop()

# reset trained model when data changes
data_key = (source, len(raw), tuple(raw.columns), float(pd.to_numeric(raw.iloc[:, -1], errors="coerce").sum()))
if st.session_state.get("data_key") != data_key:
    st.session_state["data_key"] = data_key
    st.session_state.pop("bundle", None)
    st.session_state.pop("train_df", None)

tab_data, tab_train, tab_predict, tab_batch = st.tabs(
    ["📊 Data", "🧠 Train", "🔮 Predict one goat", "🐐🐐 Predict many goats"])

# ======================= DATA =======================
with tab_data:
    try:
        clean_df = gm.clean(raw)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    dropped = len(raw) - len(clean_df)
    kpi_grid([
        ("📄", "Rows uploaded", len(raw), "total in file", "#38bdf8", True),
        ("✅", "Valid rows", len(clean_df), f"{dropped} dropped" if dropped else "all rows usable", "#4ade80", True),
        ("🧬", "Breeds", clean_df["breed"].nunique(), "in dataset", "#fbbf24", True),
    ])
    if dropped:
        st.warning(f"{dropped} rows dropped: missing values, unknown sex, or non-positive weight.")

    section("🌐 Herd in 3D")
    st.plotly_chart(scatter3d(clean_df), width="stretch")

    section("📈 Age vs weight by breed")
    fig = px.scatter(clean_df, x="age_months", y="weight_kg", color="breed", symbol="sex", opacity=0.75,
                     trendline=None)
    st.plotly_chart(style(fig, 380), width="stretch")

    section("🗂 Table")
    st.dataframe(clean_df.head(50), width="stretch")
    st.download_button("Download this dataset (CSV)", to_csv_bytes(clean_df), "goat_data.csv", "text/csv")

# ======================= TRAIN =======================
with tab_train:
    st.write("Training fits three things: a breed growth curve, a weight estimator from body measures, and a future-weight model.")
    if st.button("🚀 Train models", type="primary"):
        with st.spinner("Training..."):
            try:
                bundle, train_df = gm.train_all(raw, use_age_for_estimator=use_age)
                st.session_state["bundle"] = bundle
                st.session_state["train_df"] = train_df
                st.toast("Training done", icon="🎉")
            except Exception as e:
                st.error(str(e))
    if "bundle" in st.session_state:
        bundle = st.session_state["bundle"]
        train_df = st.session_state["train_df"]
        st.success(f"Trained on {bundle['rows']} rows.")

        section("⚖️ Weight estimator (from body measures)")
        wm = bundle["weight_model"]
        if wm is None:
            st.info("Not trained. Need at least 30 rows with height, girth, or length columns.")
        else:
            m = wm["metrics"]
            kpi_grid([
                ("🎯", "MAE (kg)", f"{m['MAE (kg)']:.2f}", "avg error, lower is better", "#4ade80", False),
                ("📐", "R²", f"{m['R2']:.3f}", "1.0 is perfect", "#38bdf8", False),
                ("🧮", "Rows used", m["Rows"], "after cleaning", "#fbbf24", True),
            ])
            st.caption("Metrics come from a holdout test split (whole goats held out when goat_id exists).")

        section("⏩ Future weight model")
        fut = bundle["future"]
        if fut["mode"] == "reference":
            st.info("Not enough repeat measurements per goat. Using the breed growth curve scaled to each goat's condition. Add goat_id with 3+ records per goat for a learned model.")
        else:
            m = fut["metrics"]
            kpi_grid([
                ("🤖", "MAE model (kg)", f"{m['MAE model (kg)']:.2f}", "learned model", "#4ade80", False),
                ("📏", "MAE baseline (kg)", f"{m['MAE reference-curve baseline (kg)']:.2f}", "reference curve", "#fb7185", False),
                ("🔗", "Training pairs", m["Pairs"], "before / after records", "#fbbf24", True),
            ])

        section("🩺 Current herd status")
        counts = train_df["status"].value_counts().reindex(ORDER).fillna(0)
        c1, c2 = st.columns([1, 1.4])
        with c1:
            pie = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.62, pull=[0.04] * 3,
                                   marker=dict(colors=[COL[s] for s in counts.index]), textinfo="percent+label"))
            pie.update_layout(showlegend=False)
            st.plotly_chart(style(pie, 340), width="stretch")
        with c2:
            st.dataframe(
                train_df.groupby(["breed", "sex"])["status"].value_counts().unstack(fill_value=0),
                width="stretch",
            )

        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button("Download trained model (.joblib)", buf.getvalue(), "goat_model.joblib")
    else:
        st.info("Click Train models.")


def need_model():
    if "bundle" not in st.session_state:
        st.info("Train a model first (Train tab).")
        st.stop()
    return st.session_state["bundle"]


# ======================= PREDICT ONE =======================
with tab_predict:
    bundle = need_model()
    wm = bundle["weight_model"]
    c1, c2, c3 = st.columns(3)
    breed = c1.selectbox("Breed", bundle["breeds"])
    sex = c2.radio("Sex", ["female", "male"], horizontal=True)
    age = c3.number_input("Age (months)", 0.0, 120.0, 6.0, 0.5)

    know_weight = st.checkbox("I know current weight", value=True)
    measures = {}
    weight = None
    if know_weight:
        weight = st.number_input("Current weight (kg)", 0.5, 300.0, 20.0, 0.5)
    else:
        if wm is None:
            st.warning("Weight estimator not available for this data. Enter weight instead.")
            st.stop()
        st.write("Enter body measures to estimate weight.")
        cols = st.columns(len([m for m in wm["num"] if m != "age_months"]))
        defaults = {"height_cm": 55.0, "chest_girth_cm": 62.0, "body_length_cm": 52.0}
        for col, m in zip(cols, [m for m in wm["num"] if m != "age_months"]):
            measures[m] = col.number_input(m, 10.0, 200.0, defaults.get(m, 50.0), 0.5)

    horizon = st.slider("Predict how many months ahead?", 1, 12, 3)

    if st.button("🔮 Predict", type="primary"):
        row = pd.DataFrame([{"breed": breed, "sex": sex, "age_months": age, **measures}])
        if not know_weight:
            weight = float(gm.predict_weight(bundle, row)[0])
        row["weight_kg"] = weight
        res = gm.predict_future(bundle, row[["breed", "sex", "age_months", "weight_kg"]], horizon).iloc[0]
        lo, hi = res.ref_weight_now_kg * gm.UNDER, res.ref_weight_now_kg * gm.OVER

        kpi_grid([
            ("⚖️", "Current weight (kg)" if know_weight else "Estimated weight (kg)", f"{weight:.1f}", f"{age:g} months old", COL[res.status_now], False),
            ("🟢", "Normal range now (kg)", f"{lo:.1f} to {hi:.1f}", breed, "#38bdf8", False),
            ("🚀", f"Weight at {res.future_age_months:.0f} months (kg)", f"{res.future_weight_kg:.1f}", f"in {horizon} months", COL[res.status_future], False),
        ])

        g, s = st.columns([1.3, 1])
        with g:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=weight, delta=dict(reference=res.ref_weight_now_kg),
                number=dict(suffix=" kg"),
                gauge=dict(axis=dict(range=[0, max(hi * 1.5, weight * 1.15)]), bar=dict(color="#e2e8f0", thickness=0.25),
                           steps=[dict(range=[0, lo], color=COL["Underweight"]),
                                  dict(range=[lo, hi], color=COL["Normal"]),
                                  dict(range=[hi, max(hi * 1.5, weight * 1.15)], color=COL["Overweight"])])))
            st.plotly_chart(style(gauge, 320), width="stretch")
        with s:
            st.write("")
            pill("Status now", res.status_now)
            pill(f"Status in {horizon} months", res.status_future)

        # feeding suggestion
        section("🌾 Daily feed suggestion (per goat)")
        plan_now = gm.feeding_plan(weight, age, res.status_now)
        plan_fut = gm.feeding_plan(res.future_weight_kg, res.future_age_months, res.status_future)
        if plan_now is None:
            st.info("Kids under 3 months: milk plus creep feed. Ask a vet for the plan.")
        else:
            table = pd.DataFrame({
                "Feed": list(plan_now),
                "Per kg body weight (g/day)": [round(v * 1000 / weight, 1) for v in plan_now.values()],
                "Now (kg/day)": [round(v, 2) for v in plan_now.values()],
            })
            if plan_fut is not None:
                table[f"At {res.future_age_months:.0f} months (kg/day)"] = [round(v, 2) for v in plan_fut.values()]
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption("Always: clean water all day, salt lick. " + gm.FEED_NOTES[res.status_now])
            st.caption("Change feed slowly over 7 to 10 days. General guide only. Confirm with a vet or animal nutritionist.")

        # trajectory chart
        section("📈 Growth trajectory")
        hs = np.arange(0, horizon + 0.01, 0.5)
        traj_in = pd.DataFrame({"breed": breed, "sex": sex, "age_months": age, "weight_kg": weight}, index=range(len(hs)))
        traj = gm.predict_future(bundle, traj_in, hs)
        x = traj["future_age_months"].round(1)
        ref = traj["ref_weight_future_kg"].values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ref * gm.OVER, line=dict(width=0), name="Overweight above", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=ref * gm.UNDER, fill="tonexty", fillcolor="rgba(74,222,128,0.15)",
                                 line=dict(width=0), name="Normal band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=ref, line=dict(color="#94a3b8", dash="dot"), name="Breed reference"))
        fig.add_trace(go.Scatter(x=x, y=traj["future_weight_kg"].values, mode="lines+markers",
                                 line=dict(color="#a3e635", width=4), marker=dict(size=9, line=dict(width=2, color="#052e16")),
                                 name="This goat (predicted)"))
        fig.update_layout(xaxis_title="age (months)", yaxis_title="weight (kg)", hovermode="x unified")
        st.plotly_chart(style(fig, 420), width="stretch")

# ======================= PREDICT MANY =======================
with tab_batch:
    bundle = need_model()
    st.write("Upload goats to forecast. Required: `breed`, `sex`, `age_months`, `weight_kg`.")
    h_batch = st.slider("Months ahead", 1, 12, 3, key="hb")
    up2 = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"], key="batch_up")
    if up2 is not None:
        try:
            herd = gm.clean(read_any(up2))
            keep = [c for c in ["goat_id", "breed", "sex", "age_months", "weight_kg"] if c in herd.columns]
            out = gm.predict_future(bundle, herd[keep], h_batch)
            plans = [gm.feeding_plan(w, a, st_) for w, a, st_ in zip(out["weight_kg"], out["age_months"], out["status_now"])]
            out["concentrate_kg_day"] = [p["Concentrate (grain/pellet)"] if p else None for p in plans]
            out["dry_roughage_kg_day"] = [p["Dry roughage (hay, straw, tree leaves)"] if p else None for p in plans]
            out["green_fodder_kg_day"] = [p["Green fodder"] if p else None for p in plans]
            out = out.round(2)

            vc = out["status_future"].value_counts().reindex(ORDER).fillna(0)
            kpi_grid([("🐐", "Goats", len(out), "forecasted", "#38bdf8", True)] +
                     [("🔴" if s == "Underweight" else "🟢" if s == "Normal" else "🟡", s, int(vc[s]),
                       f"in {h_batch} months", COL[s], True) for s in ORDER])

            c1, c2 = st.columns([1.6, 1])
            with c1:
                fig = px.scatter(out, x="weight_kg", y="future_weight_kg", color="status_future",
                                 color_discrete_map=COL, hover_data=[c for c in ["goat_id", "breed", "age_months"] if c in out.columns])
                lim = [out["weight_kg"].min(), out["future_weight_kg"].max()]
                fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dot", color="#94a3b8"), name="no change"))
                fig.update_layout(xaxis_title="weight now (kg)", yaxis_title="predicted weight (kg)")
                st.plotly_chart(style(fig, 380), width="stretch")
            with c2:
                bar = px.bar(x=vc.index, y=vc.values, color=vc.index, color_discrete_map=COL,
                             labels=dict(x="", y="goats"))
                bar.update_layout(showlegend=False)
                st.plotly_chart(style(bar, 380), width="stretch")

            st.dataframe(out, width="stretch")
            st.download_button("Download predictions (CSV)", to_csv_bytes(out), "goat_predictions.csv", "text/csv")
        except Exception as e:
            st.error(str(e))