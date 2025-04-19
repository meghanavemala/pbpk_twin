import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from dataclasses import dataclass
import yaml
import plotly.graph_objs as go
import requests
import json
import re

# --- Data Classes ---
@dataclass
class Physiology:
    weight_kg: float = 70.0
    V_blood: float = 5.0
    V_liver: float = 1.8
    Q_liver: float = 90.0
    V_muscle: float = 29.0
    Q_muscle: float = 450.0
    V_fat: float = 18.0
    Q_fat: float = 30.0
    CLint_liver: float = 20.0

@dataclass
class Compound:
    name: str = "DrugX"
    Kp_liver: float = 10.0
    Kp_muscle: float = 3.0
    Kp_fat: float = 50.0
    k_abs: float = 1.0

@dataclass
class DosingEvent:
    type: str
    amount: float
    time: float
    duration: float = 0.0

# --- Simulator Class ---
class PBPKQSPSimulator:
    def __init__(self, phys: Physiology, cmpd: Compound, qsp_params=None):
        self.phys = phys
        self.cmpd = cmpd
        self.qsp = qsp_params
        self.names = ['Gut','Blood','Liver','Muscle','Fat']
        self.vols = [1.0, phys.V_blood, phys.V_liver, phys.V_muscle, phys.V_fat]
        self.flows = [0.0, 0.0, phys.Q_liver, phys.Q_muscle, phys.Q_fat]
        self.kps   = [1.0, 1.0, cmpd.Kp_liver, cmpd.Kp_muscle, cmpd.Kp_fat]
        self.clint = [0.0, 0.0, phys.CLint_liver, 0.0, 0.0]
        self.n_pk = len(self.names)
        self.n_qsp = 3 if qsp_params else 0

    def odes(self, t, y, events):
        inj = np.zeros(self.n_pk)
        for ev in events:
            if ev.type=='iv_bolus' and abs(t-ev.time)<self.dt/2:
                inj[1] += ev.amount/self.dt
            elif ev.type=='iv_infusion' and ev.time <= t < ev.time+ev.duration:
                inj[1] += ev.amount/ev.duration
            elif ev.type=='oral' and abs(t-ev.time)<self.dt/2:
                inj[0] += ev.amount/self.dt

        Agut, Ab, Aliv, Amusc, Afat = y[:5]
        dAgut = -self.cmpd.k_abs*Agut + inj[0]
        Cb = Ab/self.vols[1]
        dAb = self.cmpd.k_abs*Agut + inj[1]
        dAliv = dAmusc = dAfat = 0.0
        for idx, (vol, flow, kp, cl) in enumerate(zip(self.vols, self.flows, self.kps, self.clint)):
            if idx<2: continue
            Ci = y[idx]/vol
            flux = flow*(Cb - Ci/kp)
            dAb   -= flux
            if idx==2: dAliv = flux - cl*Ci
            elif idx==3: dAmusc = flux
            elif idx==4: dAfat  = flux

        dPK = [dAgut, dAb, dAliv, dAmusc, dAfat]

        if self.qsp:
            kon, koff, Rtot, kprod, kdeg = self.qsp
            Rf, Rc, M = y[5:8]
            bind = kon*Cb*Rf - koff*Rc
            dRf = -bind
            dRc = bind
            dM  = kprod*Rc - kdeg*M
            return dPK + [dRf, dRc, dM]

        return dPK

    def simulate(self, events, t_end=24.0, dt=0.1):
        y0 = [0.0]*self.n_pk
        if self.qsp:
            y0 += [self.qsp[2], 0.0, 0.0]
        self.dt = dt
        t_eval = np.arange(0, t_end+dt, dt)
        sol = solve_ivp(
            fun=lambda t,y: self.odes(t,y,events),
            t_span=(0, t_end), y0=y0, t_eval=t_eval, method='RK45'
        )
        cols = self.names.copy()
        if self.qsp:
            cols += ['Free_Receptor','Drug_Receptor_Complex','Biomarker']
        df = pd.DataFrame(sol.y.T, columns=cols)
        df.insert(0, 'Time_h', sol.t)
        return df

# --- PK Metrics ---
def compute_pk_metrics(df, comp='Blood'):
    conc = df[comp]
    time = df['Time_h']
    cmax = np.max(conc)
    tmax = time[np.argmax(conc)]
    auc = np.trapz(conc, time)
    return {'Cmax': cmax, 'Tmax': tmax, 'AUC': auc}

# --- Load Config ---
def load_config(path='pbpk_config.yaml'):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

# --- LLM Insights Generator ---
def generate_llm_insights(simulation_data, pk_metrics, physiology, compound, qsp_enabled=False):
    """
    Generate insights about the simulation results using a local LLM via Ollama.
    
    Args:
        simulation_data: DataFrame with simulation results
        pk_metrics: Dictionary with PK metrics
        physiology: Physiology object with patient parameters
        compound: Compound object with drug parameters
        qsp_enabled: Boolean indicating if QSP model was used
        
    Returns:
        String with insights or error message
    """
    try:
        # Prepare the prompt with simulation data summary
        prompt = f"""
        As a pharmacokinetics expert, analyze the following PBPK simulation results and provide key insights:
        
        Drug: {compound.name}
        Patient weight: {physiology.weight_kg} kg
        
        PK metrics:
        - Cmax: {pk_metrics['Cmax']:.2f} mg
        - Tmax: {pk_metrics['Tmax']:.2f} hours
        - AUC: {pk_metrics['AUC']:.2f} mg·h
        
        Drug distribution parameters:
        - Blood volume: {physiology.V_blood} L
        - Liver Kp: {compound.Kp_liver}
        - Muscle Kp: {compound.Kp_muscle}
        - Fat Kp: {compound.Kp_fat}
        
        Intrinsic clearance: {physiology.CLint_liver}
        
        IMPORTANT: Be concise and to the point. Do NOT include any <think> or thinking sections in your response.
        
        Summarize in 4-5 bullet points, focusing on:
        1. Distribution pattern across compartments
        2. Rate of absorption and elimination
        3. Potential dosing recommendations
        4. Key considerations for this drug profile
        """
        
        # Add QSP information if enabled
        if qsp_enabled:
            max_biomarker = simulation_data['Biomarker'].max() if 'Biomarker' in simulation_data.columns else 0
            prompt += f"""
            QSP results:
            - Maximum biomarker level: {max_biomarker:.2f}
            
            Also comment on target engagement and biomarker response.
            """
        
        # Call Ollama API
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'deepseek-r1:1.5b',  # Using specific DeepSeek model
                'prompt': prompt,
                'stream': False
            }
        )
        
        if response.status_code == 200:
            response_text = response.json()['response']
            # Remove any <think>...</think> sections using regex
            cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            # Also remove any lines containing <think or </think
            cleaned_response = re.sub(r'.*<think.*\n?', '', cleaned_response)
            cleaned_response = re.sub(r'.*</think>.*\n?', '', cleaned_response)
            return cleaned_response
        else:
            return f"Error getting insights: {response.text}"
    
    except Exception as e:
        return f"Error generating insights: {str(e)}\n\nTo use this feature, please make sure Ollama is installed and running with the command: ollama serve"

# --- Streamlit UI ---
st.title("PBPK-QSP Digital Twin Simulator")
st.markdown("""
Simulate drug PK and QSP for a virtual patient.  
Parameters and logic are loaded from [pbpk_config.yaml](cci:7://file:///d:/PROJECTS/meghana-twin/pbpk_config.yaml:0:0-0:0).
""")

cfg = load_config()

# Sidebar for scenario selection
st.sidebar.header("Physiology Parameters")
phys_vals = {}
for k, v in cfg['physiology'].items():
    phys_vals[k] = st.sidebar.number_input(k, value=float(v))
phys = Physiology(**phys_vals)

st.sidebar.header("Compound Parameters")
cmpd_vals = {}
for k, v in cfg['compound'].items():
    if k == "name":
        cmpd_vals[k] = st.sidebar.text_input(k, value=str(v))
    else:
        cmpd_vals[k] = st.sidebar.number_input(k, value=float(v))
cmpd = Compound(**cmpd_vals)

st.sidebar.header("QSP Parameters")
qsp_vals = {}
for k, v in cfg['qsp'].items():
    qsp_vals[k] = st.sidebar.number_input(k, value=float(v))
qsp_tuple = tuple(qsp_vals.values())

st.sidebar.header("Dosing Regimen")
dosing_events = []
for i, d in enumerate(cfg['dosing']):
    st.sidebar.markdown(f"**Dose {i+1}**")
    dtype = st.sidebar.selectbox(f"Type {i+1}", options=['iv_bolus', 'iv_infusion', 'oral'], index=['iv_bolus', 'iv_infusion', 'oral'].index(d['type']), key=f"type_{i}")
    amount = st.sidebar.number_input(f"Amount (mg) {i+1}", value=float(d['amount']), key=f"amount_{i}")
    time = st.sidebar.number_input(f"Time (h) {i+1}", value=float(d['time']), key=f"time_{i}")
    duration = st.sidebar.number_input(f"Duration (h) {i+1}", value=float(d.get('duration', 0.0)), key=f"duration_{i}")
    dosing_events.append(DosingEvent(dtype, amount, time, duration))

t_end = st.sidebar.number_input("Simulation End Time (h)", value=24.0)
dt = st.sidebar.number_input("Time Step (h)", value=0.1)

# --- Run Simulation ---
sim = PBPKQSPSimulator(phys, cmpd, qsp_params=qsp_tuple)
df = sim.simulate(dosing_events, t_end=t_end, dt=dt)

st.subheader("Simulation Results (First 5 Rows)")
st.dataframe(df.head())

# --- Plotly Interactive Graphs ---
st.subheader("Drug Amount in Compartments Over Time")
fig = go.Figure()
for col in sim.names:
    fig.add_trace(go.Scatter(x=df['Time_h'], y=df[col], mode='lines', name=col))
fig.update_layout(
    xaxis_title="Time (h)",
    yaxis_title="Drug Amount (mg)",
    legend_title="Compartment",
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# --- PK Metrics ---
pk = compute_pk_metrics(df, comp='Blood')
st.subheader("PK Metrics (Blood Compartment)")
st.write(f"Cmax: {pk['Cmax']:.2f} mg")
st.write(f"Tmax: {pk['Tmax']:.2f} h")
st.write(f"AUC: {pk['AUC']:.2f} mg·h")

# --- Advanced: Plot QSP if present ---
if sim.qsp:
    # Separate QSP components into three different graphs
    qsp_cols = ['Free_Receptor','Drug_Receptor_Complex','Biomarker']
    qsp_titles = ['Free Receptor Over Time', 'Drug-Receptor Complex Over Time', 'Biomarker Over Time']
    for col, title in zip(qsp_cols, qsp_titles):
        if col in df.columns:
            st.subheader(title)
            fig_qsp = go.Figure()
            fig_qsp.add_trace(go.Scatter(x=df['Time_h'], y=df[col], mode='lines', name=col))
            fig_qsp.update_layout(
                xaxis_title="Time (h)",
                yaxis_title=col,
                legend_title=col,
                hovermode="x unified"
            )
            st.plotly_chart(fig_qsp, use_container_width=True)

# --- LLM Generated Insights ---
st.subheader("Insights")

with st.spinner("Generating insights..."):
    try:
        insights = generate_llm_insights(
            simulation_data=df,
            pk_metrics=pk,
            physiology=phys,
            compound=cmpd,
            qsp_enabled=bool(sim.qsp)
        )
        st.markdown(insights)
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}\n\nPlease make sure Ollama is installed and running with the DeepSeek model.")