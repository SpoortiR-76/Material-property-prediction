import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model_engine import predict, get_feature_importance, CONCRETE_COLS, STEEL_COLS, MECH_COLS
from utils import calculate_cost_metrics, calculate_sustainability_metrics, get_structural_performance_score, get_smart_suggestions, optimize_mix
import os

# Page Configuration
st.set_page_config(page_title="MaterialAI v2", layout="wide", page_icon="🏗️")

# Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-bottom: 20px;}
    .suggestion-box { background-color: #f1f3f5; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin-bottom: 10px; font-size: 0.9em;}
    </style>
    """, unsafe_allow_html=True)

# Navigation
st.sidebar.title("MaterialAI v2")
module = st.sidebar.selectbox("Select Module", ["Dashboard", "Concrete Analysis", "Steel Analysis", "Mechanical Properties", "Mix Optimizer"])

# --- DASHBOARD ---
if module == "Dashboard":
    st.title("🚀 Advanced Material Decision Support")
    st.info("Project Version 2: Self-contained and modular.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h4>State</h4><h2 style="color:green;">Independent</h2><p>Ready for Commit</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>Models</h4><h2 style="color:blue;">Active</h2><p>RF, Linear, Multi-Output</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h4>Currency</h4><h2 style="color:orange;">Rs.</h2><p>Indian Market Rates</p></div>', unsafe_allow_html=True)

# --- CONCRETE ANALYSIS ---
elif module == "Concrete Analysis":
    st.title("🏗️ Concrete Strength Predictor")
    c_in, c_res = st.columns([1, 1.2])
    
    with c_in:
        st.subheader("Mix Ingredients (kg/m³)")
        cement = st.slider("Cement", 100, 600, 250)
        slag = st.slider("Slag", 0, 400, 50)
        ash = st.slider("Fly Ash", 0, 400, 50)
        water = st.slider("Water", 100, 250, 180)
        sp = st.slider("Superplasticizer", 0, 30, 5)
        coarse = st.slider("Coarse Aggregate", 700, 1200, 1000)
        fine = st.slider("Fine Aggregate", 500, 1000, 800)
        age = st.number_input("Age (Days)", 1, 365, 28)
        
    with c_res:
        st.subheader("Analysis Results")
        inputs = [cement, slag, ash, water, sp, coarse, fine, age]
        strength = predict("concrete", inputs)
        cost = calculate_cost_metrics(cement, slag, ash, water, sp, coarse, fine)
        co2 = calculate_sustainability_metrics(cement, slag, ash, water, sp, coarse, fine)
        score = get_structural_performance_score(strength, cost, co2)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Strength", f"{strength:.1f} MPa")
        m2.metric("Cost", f"Rs. {cost:.0f}")
        m3.metric("CO2", f"{co2:.0f} kg")
        
        st.write(f"**Performance Score:** {score}")
        st.progress(min(score / 20.0, 1.0))
        
        st.subheader("Engineering Suggestions")
        for s in get_smart_suggestions(strength, cost, co2, age):
            st.markdown(f'<div class="suggestion-box">{s}</div>', unsafe_allow_html=True)

# --- STEEL ANALYSIS ---
elif module == "Steel Analysis":
    st.title("⚔️ Steel Property Predictor")
    st.info("Predict tensile strength from chemical composition.")
    
    col1, col2 = st.columns(2)
    s_inputs = []
    with col1:
        for f in STEEL_COLS[:5]:
            val = st.number_input(f"{f.upper()} (%)", 0.0, 5.0, 0.1, format="%.3f")
            s_inputs.append(val)
    with col2:
        for f in STEEL_COLS[5:]:
            val = st.number_input(f"{f.upper()} (%)", 0.0, 5.0, 0.1, format="%.3f")
            s_inputs.append(val)
            
    if st.button("Calculate Strength"):
        res = predict("steel", s_inputs)
        st.success(f"Predicted Ultimate Tensile Strength: {res:.2f} MPa")

# --- MECHANICAL PROPERTIES ---
elif module == "Mechanical Properties":
    st.title("⚙️ Mechanical Analysis & Classification")
    m_in, m_res = st.columns([1, 1.2])
    
    with m_in:
        e = st.number_input("Elastic Modulus (E)", 10000, 300000, 200000)
        g = st.number_input("Shear Modulus (G)", 5000, 150000, 77000)
        mu = st.slider("Poisson's Ratio", 0.0, 0.5, 0.3)
        ro = st.number_input("Density (Ro)", 500, 10000, 7850)
        m_inputs = [e, g, mu, ro]
        
    with m_res:
        pred_reg = predict("mechanical", m_inputs)
        is_usable = predict("mechanical_clf", m_inputs)
        
        st.metric("Ult Strength (Su)", f"{pred_reg[0]:.1f} MPa")
        st.metric("Yield Strength (Sy)", f"{pred_reg[1]:.1f} MPa")
        
        if is_usable:
            st.success("✅ CLASSIFICATION: MATERIAL SUITABLE")
        else:
            st.error("❌ CLASSIFICATION: NOT SUITSBLE")

# --- MIX OPTIMIZER ---
elif module == "Mix Optimizer":
    st.title("🎯 AI Mix Designer (Reverse Prediction)")
    target = st.number_input("Target Strength (MPa)", 10, 100, 45)
    
    if st.button("Generate Optimized Mix"):
        wrapper = lambda x: predict("concrete", x.flatten().tolist())
        optimized = optimize_mix(target, wrapper)
        
        st.subheader("Suggested Recipe")
        cols = st.columns(4)
        for i, (name, val) in enumerate(zip(CONCRETE_COLS, optimized)):
            cols[i % 4].metric(name, f"{val:.1f}")
        st.success(f"Verification: Predicted Strength is ~{predict('concrete', optimized.tolist()):.1f} MPa")
