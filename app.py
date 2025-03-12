import streamlit as st
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import plotly.express as px
import plotly.graph_objects as go
import traceback
from typing import Dict, List, Any, Tuple, Optional, Union

from models import SimpleNKCModel, AgentNKCModel
from utils.visualization import (
    parse_seeds, 
    plot_simple_model_results, 
    plot_agent_model_results
)

# ===============================
# กำหนดค่าคงที่
# ===============================
MAX_HISTORY_ITEMS = 5  # จำกัดจำนวนประวัติที่เก็บไว้

# ===============================
# ฟังก์ชันเตรียม session state
# ===============================
def initialize_session_state():
    """เตรียม session state เริ่มต้น"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'max_history_shown' not in st.session_state:
        st.session_state.max_history_shown = 3
    if 'model_type' not in st.session_state:
        st.session_state.model_type = "simple"

# ===============================
# ฟังก์ชันสร้าง UI สำหรับแบบจำลองแบบง่าย
# ===============================
def create_simple_model_ui(model: SimpleNKCModel) -> Dict[str, Any]:
    """
    สร้าง UI และรับค่าพารามิเตอร์สำหรับแบบจำลองแบบง่าย
    
    Args:
        model: แบบจำลองแบบง่าย
    
    Returns:
        Dict ที่มีพารามิเตอร์ที่ป้อนโดยผู้ใช้
    """
    param_info = model.get_parameter_info()
    advanced_params = {}

    # พารามิเตอร์หลัก
    st.sidebar.subheader("พารามิเตอร์หลัก")
    
    # จำนวนตัวแปร N
    N = st.sidebar.number_input(
        param_info["N"]["label"], 
        min_value=param_info["N"]["min"], 
        max_value=param_info["N"]["max"], 
        value=param_info["N"]["default"], 
        help=param_info["N"]["help"]
    )
    
    # คำนวณค่าเริ่มต้นที่เหมาะสมสำหรับ K
    default_k_effectuation = max(1, int(N * 0.2))  # ประมาณ 20% ของ N
    default_k_causation = max(2, int(N * 0.6))     # ประมาณ 60% ของ N
    
    # ค่า K สำหรับแต่ละกลยุทธ์
    col1, col2 = st.sidebar.columns(2)
    with col1:
        K_EFFECTUATION = st.number_input(
            param_info["K_EFFECTUATION"]["label"], 
            min_value=param_info["K_EFFECTUATION"]["min"], 
            max_value=min(param_info["K_EFFECTUATION"]["max"], N-1), 
            value=min(default_k_effectuation, N-1), 
            help=param_info["K_EFFECTUATION"]["help"]
        )
    with col2:
        K_CAUSATION = st.number_input(
            param_info["K_CAUSATION"]["label"], 
            min_value=param_info["K_CAUSATION"]["min"], 
            max_value=min(param_info["K_CAUSATION"]["max"], N-1), 
            value=min(default_k_causation, N-1), 
            help=param_info["K_CAUSATION"]["help"]
        )
    
    # เลือกกลยุทธ์
    strategies = st.sidebar.multiselect(
        param_info["strategies"]["label"],
        param_info["strategies"]["options"],
        default=param_info["strategies"]["default"],
        help=param_info["strategies"]["help"]
    )
    
    # ระดับความซับซ้อน
    C_values = st.sidebar.multiselect(
        param_info["C_values"]["label"],
        param_info["C_values"]["options"],
        default=param_info["C_values"]["default"],
        help=param_info["C_values"]["help"]
    )
    
    # สัมประสิทธิ์การเปลี่ยนแปลง
    rho_values = st.sidebar.multiselect(
        param_info["rho_values"]["label"],
        param_info["rho_values"]["options"],
        default=param_info["rho_values"]["default"],
        help=param_info["rho_values"]["help"]
    )
    
    # เกณฑ์การยอมรับ
    thresholds = st.sidebar.multiselect(
        param_info["thresholds"]["label"],
        param_info["thresholds"]["options"],
        default=param_info["thresholds"]["default"],
        help=param_info["thresholds"]["help"]
    )
    
    # Random Seeds
    seeds_input = st.sidebar.text_input(
        param_info["seeds_input"]["label"],
        param_info["seeds_input"]["default"],
        help=param_info["seeds_input"]["help"]
    )
    
    # ตัวเลือกขั้นสูง
    sim_mode = st.sidebar.radio("โหมดการตั้งค่า", ["พื้นฐาน", "ขั้นสูง"], index=0)
    
    if sim_mode == "ขั้นสูง":
        st.sidebar.subheader("ตัวเลือกขั้นสูง")
        
        advanced_params['time_steps'] = st.sidebar.number_input(
            param_info["time_steps"]["label"], 
            min_value=param_info["time_steps"]["min"], 
            max_value=param_info["time_steps"]["max"], 
            value=param_info["time_steps"]["default"], 
            help=param_info["time_steps"]["help"]
        )
        
        advanced_params['pre_chasm_steps'] = st.sidebar.number_input(
            param_info["pre_chasm_steps"]["label"], 
            min_value=param_info["pre_chasm_steps"]["min"], 
            max_value=min(param_info["pre_chasm_steps"]["max"], advanced_params.get('time_steps', 100)-5), 
            value=min(param_info["pre_chasm_steps"]["default"], advanced_params.get('time_steps', 100)-5), 
            help=param_info["pre_chasm_steps"]["help"]
        )
        
        advanced_params['max_simulations'] = st.sidebar.number_input(
            param_info["max_simulations"]["label"], 
            min_value=param_info["max_simulations"]["min"], 
            max_value=param_info["max_simulations"]["max"], 
            value=param_info["max_simulations"]["default"], 
            help=param_info["max_simulations"]["help"]
        )
        
        advanced_params['interaction_strength'] = st.sidebar.slider(
            param_info["interaction_strength"]["label"], 
            min_value=param_info["interaction_strength"]["min"], 
            max_value=param_info["interaction_strength"]["max"], 
            value=param_info["interaction_strength"]["default"], 
            step=param_info["interaction_strength"]["step"],
            help=param_info["interaction_strength"]["help"]
        )
        
        advanced_params['save_history'] = st.sidebar.checkbox(
            "บันทึกประวัติการจำลอง", 
            value=True, 
            help="บันทึกผลการจำลองเพื่อเปรียบเทียบภายหลัง"
        )
    else:
        # ค่าเริ่มต้นสำหรับโหมดพื้นฐาน
        advanced_params = {
            'time_steps': param_info["time_steps"]["default"],
            'pre_chasm_steps': param_info["pre_chasm_steps"]["default"],
            'max_simulations': param_info["max_simulations"]["default"],
            'interaction_strength': param_info["interaction_strength"]["default"],
            'save_history': True
        }
    
    # ตั้งค่ากราฟ
    st.sidebar.subheader("ตั้งค่ากราฟ")
    graph_params = {}
    
    graph_params['graph_type'] = st.sidebar.selectbox(
        "เลือกประเภทกราฟ", 
        ["2D แยกตามเกณฑ์", "3D รวม Time Steps", "2D Time Series", "แสดงทั้งหมด"], 
        index=0
    )
    
    # ตั้งค่าเพิ่มเติมสำหรับแต่ละประเภทกราฟ
    if graph_params['graph_type'] == "2D แยกตามเกณฑ์" or graph_params['graph_type'] == "แสดงทั้งหมด":
        graph_params['y_axis'] = st.sidebar.selectbox(
            "เลือกข้อมูลในแกน Y", 
            ["อัตราการยอมรับก่อน Chasm (%)", "อัตราการยอมรับหลัง Chasm (%)"], 
            index=1
        )
        graph_params['x_axis'] = st.sidebar.selectbox(
            "เลือกข้อมูลในแกน X", 
            ["rho", "C"], 
            index=0
        )
        
    if graph_params['graph_type'] == "3D รวม Time Steps" or graph_params['graph_type'] == "2D Time Series" or graph_params['graph_type'] == "แสดงทั้งหมด":
        graph_params['color_by'] = st.sidebar.selectbox(
            "เลือกสีตาม", 
            ["C", "กลยุทธ์", "rho"], 
            index=0, 
            help="เลือกว่าสีจะแสดง C หรือ กลยุทธ์"
        )
        graph_params['time_step_range'] = st.sidebar.slider(
            "เลือกช่วง Time Step", 
            0, 
            advanced_params.get('time_steps', 100), 
            (16, advanced_params.get('time_steps', 100)), 
            help="กรองข้อมูลตามช่วง Time Step (หน่วย: รอบ)"
        )
    
    return {
        'N': N,
        'K_EFFECTUATION': K_EFFECTUATION,
        'K_CAUSATION': K_CAUSATION,
        'strategies': strategies,
        'C_values': C_values,
        'rho_values': rho_values,
        'thresholds': thresholds,
        'seeds_input': seeds_input,
        'advanced_params': advanced_params,
        'graph_params': graph_params
    }

# ===============================
# ฟังก์ชันสร้าง UI สำหรับแบบจำลองหลายเอเจนต์
# ===============================
def create_agent_model_ui(model: AgentNKCModel) -> Dict[str, Any]:
    """
    สร้าง UI และรับค่าพารามิเตอร์สำหรับแบบจำลองหลายเอเจนต์
    
    Args:
        model: แบบจำลองหลายเอเจนต์
    
    Returns:
        Dict ที่มีพารามิเตอร์ที่ป้อนโดยผู้ใช้
    """
    param_info = model.get_parameter_info()
    advanced_params = {}
    
    # พารามิเตอร์หลัก
    st.sidebar.subheader("พารามิเตอร์หลัก")
    
    # จำนวนบิตในสตริง N
    N = st.sidebar.number_input(
        param_info["N"]["label"], 
        min_value=param_info["N"]["min"], 
        max_value=param_info["N"]["max"], 
        value=param_info["N"]["default"], 
        help=param_info["N"]["help"]
    )
    
    # ระดับการเชื่อมโยงระหว่างเอเจนต์ C
    c_values = st.sidebar.multiselect(
        param_info["c_values"]["label"],
        param_info["c_values"]["options"],
        default=param_info["c_values"]["default"],
        help=param_info["c_values"]["help"]
    )
    
    # จำนวนขั้นตอนการค้นหา
    steps = st.sidebar.number_input(
        param_info["steps"]["label"], 
        min_value=param_info["steps"]["min"], 
        max_value=param_info["steps"]["max"], 
        value=param_info["steps"]["default"], 
        help=param_info["steps"]["help"]
    )
    
    # จำนวนรอบการจำลอง
    runs = st.sidebar.number_input(
        param_info["runs"]["label"], 
        min_value=param_info["runs"]["min"], 
        max_value=param_info["runs"]["max"], 
        value=param_info["runs"]["default"], 
        help=param_info["runs"]["help"]
    )
    
    # แสดงความคืบหน้า
    show_progress = st.sidebar.checkbox(
        param_info["show_progress"]["label"],
        value=param_info["show_progress"]["default"],
        help=param_info["show_progress"]["help"]
    )
    
    # บันทึกประวัติ
    save_history = st.sidebar.checkbox(
        "บันทึกประวัติการจำลอง", 
        value=True, 
        help="บันทึกผลลัพธ์การจำลองเพื่อเปรียบเทียบภายหลัง"
    )
    
    # เพิ่มส่วนการปรับแต่งค่า Fitness
    st.sidebar.subheader("ตั้งค่าการคำนวณ Fitness")
    
    # โหมดตั้งค่า Fitness
    fitness_mode = st.sidebar.radio(
        "โหมดการตั้งค่า Fitness",
        ["พื้นฐาน", "ขั้นสูง"],
        index=0
    )
    
    fitness_params = {}
    
    if fitness_mode == "ขั้นสูง":
        # ค่า K สำหรับ Effectuation
        k_effectuation = st.sidebar.number_input(
            "K สำหรับ Effectuation",
            min_value=1,
            max_value=5,
            value=2,
            help="จำนวนบิตที่มีอิทธิพลต่อแต่ละตำแหน่งสำหรับกลยุทธ์ Effectuation"
        )
        
        # ค่า K สำหรับ Causation
        k_causation = st.sidebar.number_input(
            "K สำหรับ Causation",
            min_value=3,
            max_value=10,
            value=7,
            help="จำนวนบิตที่มีอิทธิพลต่อแต่ละตำแหน่งสำหรับกลยุทธ์ Causation"
        )
        
        # น้ำหนักสำหรับบิตของตนเอง (own_part)
        own_weight = st.sidebar.slider(
            "น้ำหนักสำหรับบิตของตนเอง",
            min_value=0.1,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="น้ำหนักที่ใช้คูณกับอัตราส่วนของบิตของเอเจนต์เอง (ค่ามากขึ้น = ตนเองมีอิทธิพลมากขึ้น)"
        )
        
        # น้ำหนักสำหรับบิตจากเอเจนต์อื่น (cross_part)
        cross_weight = st.sidebar.slider(
            "น้ำหนักสำหรับบิตจากเอเจนต์อื่น",
            min_value=-0.5,
            max_value=0.5,
            value=-0.4,
            step=0.05,
            help="น้ำหนักที่ใช้คูณกับอัตราส่วนของบิตจากเอเจนต์อื่น (ค่าติดลบ = ปฏิสัมพันธ์มีผลเชิงลบ)"
        )
        
        # ค่าคงที่ในการคำนวณ Fitness
        base_fitness = st.sidebar.slider(
            "ค่าคงที่ใน Fitness",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="ค่าคงที่ที่เพิ่มเข้าไปในสมการคำนวณ Fitness"
        )
        
        # ขอบเขตของค่าสุ่มในการคำนวณ Fitness
        random_range = st.sidebar.slider(
            "ขอบเขตของค่าสุ่ม",
            min_value=0.0,
            max_value=0.5,
            value=0.15,
            step=0.05,
            help="ขอบเขตบนของค่าสุ่มที่เพิ่มเข้าไปในสมการคำนวณ Fitness"
        )
        
        # ความเอนเอียงเริ่มต้น (เปอร์เซ็นต์บิตเป็น 1)
        p_ones = st.sidebar.slider(
            "ความเอนเอียงเริ่มต้น (% บิตเป็น 1)",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="ความน่าจะเป็นที่บิตเริ่มต้นจะเป็น 1 (0.1 = 10%)"
        )
        
        fitness_params = {
            'k_effectuation': k_effectuation,
            'k_causation': k_causation,
            'own_weight': own_weight,
            'cross_weight': cross_weight,
            'base_fitness': base_fitness,
            'random_range': random_range,
            'p_ones': p_ones
        }
    else:
        # ค่าเริ่มต้นสำหรับโหมดพื้นฐาน - ค่าที่แนะนำสำหรับให้ได้ผลลัพธ์ตามทฤษฎี
        fitness_params = {
            'k_effectuation': 2,
            'k_causation': 7,
            'own_weight': 0.6,
            'cross_weight': -0.4,
            'base_fitness': 0.1,
            'random_range': 0.15,
            'p_ones': 0.1
        }
    
    return {
        'N': N,
        'c_values': c_values,
        'steps': steps,
        'runs': runs,
        'show_progress': show_progress,
        'save_history': save_history,
        'fitness_params': fitness_params
    }

# ===============================
# ฟังก์ชันหลัก
# ===============================
def main():
    st.title("🚀 NKC Model Simulation (ผสมผสาน)")
    
    # เตรียม session state
    initialize_session_state()
    
    # เลือกประเภทแบบจำลอง
    model_type = st.sidebar.radio(
        "เลือกประเภทแบบจำลอง",
        ["แบบง่าย (ตัวแปรต่อเนื่อง)", "หลายเอเจนต์ (บิตสตริง)"],
        index=0 if st.session_state.model_type == "simple" else 1
    )
    
    # กำหนดประเภทแบบจำลองใน session state
    st.session_state.model_type = "simple" if model_type == "แบบง่าย (ตัวแปรต่อเนื่อง)" else "agent"
    
    # สร้างแบบจำลองตามที่เลือก
    if st.session_state.model_type == "simple":
        model = SimpleNKCModel()
        params = create_simple_model_ui(model)
    else:
        model = AgentNKCModel()
        params = create_agent_model_ui(model)
    
    # คำอธิบายแบบจำลอง
    with st.expander("📝 คำอธิบายแบบจำลอง", expanded=False):
        st.markdown(model.get_description())
    
    # ปุ่มเริ่มการจำลอง
    run_button = st.sidebar.button("▶️ เริ่มการจำลอง", use_container_width=True)
    
    if run_button:
        try:
            if st.session_state.model_type == "simple":
                # ดึงพารามิเตอร์สำหรับแบบจำลองแบบง่าย
                N = params['N']
                K_EFFECTUATION = params['K_EFFECTUATION']
                K_CAUSATION = params['K_CAUSATION']
                strategies = params['strategies']
                C_values = params['C_values']
                rho_values = params['rho_values']
                thresholds = params['thresholds']
                seeds_input = params['seeds_input']
                advanced_params = params['advanced_params']
                graph_params = params['graph_params']
                
                # ตรวจสอบข้อมูลนำเข้า
                if not strategies:
                    st.error("กรุณาเลือกอย่างน้อยหนึ่งกลยุทธ์")
                    st.stop()
                if not C_values:
                    st.error("กรุณาเลือกอย่างน้อยหนึ่งระดับความซับซ้อน (C)")
                    st.stop()
                if not rho_values:
                    st.error("กรุณาเลือกอย่างน้อยหนึ่งสัมประสิทธิ์การเปลี่ยนแปลง (rho)")
                    st.stop()
                if not thresholds:
                    st.error("กรุณาเลือกอย่างน้อยหนึ่งเกณฑ์การยอมรับ")
                    st.stop()
                
                # แปลง seeds
                try:
                    seeds = parse_seeds(seeds_input)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                
                # สร้างรายการ k ตามกลยุทธ์ที่เลือก
                k_list = []
                if "Effectuation" in strategies:
                    k_list.append(K_EFFECTUATION)
                if "Causation" in strategies:
                    k_list.append(K_CAUSATION)
                
                # สร้างพารามิเตอร์สำหรับการจำลอง
                params_list = []
                for k in k_list:
                    for C in C_values:
                        for rho in rho_values:
                            for thresh in thresholds:
                                for seed in seeds:
                                    params_list.append((
                                        N, k, C, rho, thresh, seed, 
                                        advanced_params['time_steps'],
                                        advanced_params['pre_chasm_steps'],
                                        advanced_params['interaction_strength']
                                    ))
                
                # ตรวจสอบจำนวนการจำลอง
                if len(params_list) > advanced_params['max_simulations']:
                    st.warning(f"จำนวนการจำลองทั้งหมด ({len(params_list)}) เกินขีดจำกัด ({advanced_params['max_simulations']}) จะใช้เพียง {advanced_params['max_simulations']} ชุดแรก")
                    params_list = params_list[:advanced_params['max_simulations']]
                
                # แสดงระหว่างประมวลผล
                with st.status("กำลังรันการจำลอง...") as status:
                    st.write(f"รันการจำลองจำนวน {len(params_list)} ชุด...")
                    
                    # ประมวลผลแบบขนาน
                    with st.spinner():
                        results = Parallel(n_jobs=-1, max_nbytes=None)(delayed(model.run_simulation)(*p) for p in params_list)
                    
                    # สร้าง DataFrame
                    df = pd.DataFrame(results)
                    
                    # ตรวจสอบว่ามีข้อมูลหรือไม่
                    if df.empty:
                        status.update(label="ไม่มีข้อมูล", state="error")
                        st.error("ไม่มีข้อมูลสำหรับการจำลอง กรุณาตรวจสอบพารามิเตอร์")
                        st.stop()
                    
                    # กรองข้อมูลที่มี error (ถ้ามี)
                    if 'error' in df.columns:
                        error_rows = df[df['error'].notnull()]
                        if not error_rows.empty:
                            st.warning(f"พบข้อผิดพลาดในการจำลอง {len(error_rows)} ชุด จากทั้งหมด {len(df)} ชุด")
                        df = df[df['error'].isnull()].drop('error', axis=1)
                    
                    # คำนวณสถิติสรุป
                    summary = df.groupby(['กลยุทธ์', 'C', 'rho', 'เกณฑ์การยอมรับ']).agg({
                        'อัตราการยอมรับก่อน Chasm (%)': ['mean', 'std', 'min', 'max'],
                        'อัตราการยอมรับหลัง Chasm (%)': ['mean', 'std', 'min', 'max'],
                        'step_ข้าม_threshold': ['mean', 'min', 'max']
                    }).reset_index()
                    
                    # ปรับชื่อคอลัมน์ให้อ่านง่าย
                    summary.columns = ['กลยุทธ์', 'C', 'rho', 'เกณฑ์การยอมรับ', 
                                     'ก่อน_Mean', 'ก่อน_Std', 'ก่อน_Min', 'ก่อน_Max', 
                                     'หลัง_Mean', 'หลัง_Std', 'หลัง_Min', 'หลัง_Max',
                                     'Step_ข้าม_Mean', 'Step_ข้าม_Min', 'Step_ข้าม_Max']
                    
                    # บันทึกประวัติ (ถ้าเปิดใช้งาน)
                    if advanced_params['save_history']:
                        # จำกัดจำนวนประวัติที่เก็บไว้
                        if len(st.session_state.history) >= MAX_HISTORY_ITEMS:
                            st.session_state.history.pop(0)  # ลบรายการเก่าสุด
                        
                        # บันทึกข้อมูลใหม่
                        st.session_state.history.append({
                            'model_type': 'simple',
                            'timestamp': pd.Timestamp.now().strftime("%H:%M:%S"),
                            'data': df.copy(),
                            'summary': summary.copy(),
                            'params': {
                                'N': N,
                                'K_EFFECTUATION': K_EFFECTUATION if "Effectuation" in strategies else None,
                                'K_CAUSATION': K_CAUSATION if "Causation" in strategies else None,
                                'strategies': strategies,
                                'C_values': C_values,
                                'rho_values': rho_values,
                                'thresholds': thresholds,
                                'seeds': seeds,
                                'time_steps': advanced_params['time_steps'],
                                'pre_chasm_steps': advanced_params['pre_chasm_steps']
                            }
                        })
                        
                    status.update(label="การจำลองเสร็จสมบูรณ์", state="complete")
                
                # แสดงผลลัพธ์
                st.header("📊 ผลลัพธ์การจำลอง")
                
                # แสดงสรุปพารามิเตอร์
                with st.expander("แสดงพารามิเตอร์ที่ใช้", expanded=False):
                    params_df = pd.DataFrame({
                        'พารามิเตอร์': [
                            'จำนวนบิตในสตริง (N)',
                            'ระดับการเชื่อมโยงระหว่างเอเจนต์ (C)' if st.session_state.model_type == "agent" else '',
                            'จำนวนขั้นตอนการค้นหา' if st.session_state.model_type == "agent" else '',
                            'จำนวนรอบการจำลอง' if st.session_state.model_type == "agent" else '',
                            'K สำหรับ Effectuation' if st.session_state.model_type == "agent" else '',
                            'K สำหรับ Causation' if st.session_state.model_type == "agent" else '',
                            'น้ำหนักสำหรับบิตของตนเอง' if st.session_state.model_type == "agent" else '',
                            'น้ำหนักสำหรับบิตจากเอเจนต์อื่น' if st.session_state.model_type == "agent" else '',
                            'ค่าคงที่ใน Fitness' if st.session_state.model_type == "agent" else '',
                            'ขอบเขตของค่าสุ่ม' if st.session_state.model_type == "agent" else '',
                            'ความเอนเอียงเริ่มต้น (% บิตเป็น 1)' if st.session_state.model_type == "agent" else ''
                        ],
                        'ค่า': [
                            str(N),
                            ', '.join(map(str, c_values)) if st.session_state.model_type == "agent" else '',
                            str(steps) if st.session_state.model_type == "agent" else '',
                            str(runs) if st.session_state.model_type == "agent" else '',
                            str(fitness_params['k_effectuation']) if st.session_state.model_type == "agent" else '',
                            str(fitness_params['k_causation']) if st.session_state.model_type == "agent" else '',
                            str(fitness_params['own_weight']) if st.session_state.model_type == "agent" else '',
                            str(fitness_params['cross_weight']) if st.session_state.model_type == "agent" else '',
                            str(fitness_params['base_fitness']) if st.session_state.model_type == "agent" else '',
                            str(fitness_params['random_range']) if st.session_state.model_type == "agent" else '',
                            str(fitness_params['p_ones']) if st.session_state.model_type == "agent" else ''
                        ]
                    })
                    st.dataframe(params_df, use_container_width=True)
                
                # แสดงตารางสรุป
                st.subheader("ตารางสรุป")
                st.dataframe(summary, use_container_width=True)
                
                # แสดงกราฟ
                st.header("📈 กราฟแสดงผล")
                if isinstance(df, pd.DataFrame):
                    # Pass the correct K values to the plotting function
                    plot_simple_model_results(df, summary, graph_params, K_EFFECTUATION, K_CAUSATION)
                else:
                    st.error("เกิดข้อผิดพลาด: ข้อมูลไม่ถูกต้อง")
                
                # ดาวน์โหลดข้อมูล
                st.header("💾 ดาวน์โหลดข้อมูล")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📊 ดาวน์โหลดข้อมูลละเอียดเป็น CSV",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name="simulation_detailed_results.csv",
                        mime="text/csv"
                    )
                with col2:
                    st.download_button(
                        label="📑 ดาวน์โหลดข้อมูลสรุปเป็น CSV",
                        data=summary.to_csv(index=False).encode('utf-8'),
                        file_name="simulation_summary.csv",
                        mime="text/csv"
                    )
            
            else:  # แบบจำลองหลายเอเจนต์
                # ดึงพารามิเตอร์สำหรับแบบจำลองหลายเอเจนต์
                N = params['N']
                c_values = params['c_values']
                steps = params['steps']
                runs = params['runs']
                show_progress = params['show_progress']
                save_history = params['save_history']
                fitness_params = params['fitness_params']
                
                # ตรวจสอบข้อมูลนำเข้า
                if not c_values:
                    st.error("กรุณาเลือกอย่างน้อยหนึ่งค่า C")
                    st.stop()
                
                # แสดงระหว่างประมวลผล
                with st.status("กำลังรันการจำลอง...") as status:
                    st.write(f"รันการจำลองสำหรับ C={c_values}, จำนวน {runs} รอบ...")
                    
                    # สร้างแบบจำลองใหม่พร้อมพารามิเตอร์ fitness ที่ปรับแต่ง
                    adjusted_model = AgentNKCModel(fitness_params=fitness_params)
                    
                    # รันการจำลอง
                    with st.spinner():
                        results = adjusted_model.run_simulation(
                            N=N,
                            c_values=c_values,
                            steps=steps,
                            runs=runs,
                            show_progress=show_progress,
                            fitness_params=fitness_params
                        )
                    
                    # ตรวจสอบความผิดพลาด
                    if 'error' in results:
                        status.update(label=f"เกิดข้อผิดพลาด: {results['error']}", state="error")
                        st.error(f"เกิดข้อผิดพลาดในการจำลอง: {results['error']}")
                        st.stop()
                    
                    # บันทึกประวัติ (ถ้าเปิดใช้งาน)
                    if save_history:
                        # จำกัดจำนวนประวัติที่เก็บไว้
                        if len(st.session_state.history) >= MAX_HISTORY_ITEMS:
                            st.session_state.history.pop(0)  # ลบรายการเก่าสุด
                        
                        # บันทึกข้อมูลใหม่
                        st.session_state.history.append({
                            'model_type': 'agent',
                            'timestamp': pd.Timestamp.now().strftime("%H:%M:%S"),
                            'results': results,
                            'params': {
                                'N': N,
                                'c_values': c_values,
                                'steps': steps,
                                'runs': runs,
                                'fitness_params': fitness_params
                            }
                        })
                    
                    status.update(label="การจำลองเสร็จสมบูรณ์", state="complete")
                
                # แสดงผลลัพธ์
                st.header("📊 ผลลัพธ์การจำลอง")
                
                # แสดงสรุปพารามิเตอร์
                with st.expander("แสดงพารามิเตอร์ที่ใช้", expanded=False):
                    params_df = pd.DataFrame({
                        'พารามิเตอร์': [
                            'จำนวนบิตในสตริง (N)',
                            'ระดับการเชื่อมโยงระหว่างเอเจนต์ (C)',
                            'จำนวนขั้นตอนการค้นหา',
                            'จำนวนรอบการจำลอง',
                            'K สำหรับ Effectuation',
                            'K สำหรับ Causation',
                            'น้ำหนักสำหรับบิตของตนเอง',
                            'น้ำหนักสำหรับบิตจากเอเจนต์อื่น',
                            'ค่าคงที่ใน Fitness',
                            'ขอบเขตของค่าสุ่ม',
                            'ความเอนเอียงเริ่มต้น (% บิตเป็น 1)'
                        ],
                        'ค่า': [
                            N,
                            ', '.join([str(c) for c in c_values]) if c_values else None,
                            steps,
                            runs,
                            fitness_params['k_effectuation'],
                            fitness_params['k_causation'],
                            fitness_params['own_weight'],
                            fitness_params['cross_weight'],
                            fitness_params['base_fitness'],
                            fitness_params['random_range'],
                            fitness_params['p_ones']
                        ]
                    })
                    st.dataframe(params_df, use_container_width=True)
                
                # แสดงกราฟ
                st.header("📈 กราฟแสดงผล")
                plot_agent_model_results(results, unique_id="current")
                
                # ดาวน์โหลดข้อมูล
                st.header("💾 ดาวน์โหลดข้อมูล")
                
                # แปลงผลลัพธ์เป็น DataFrame
                if 'results' in results:
                    results_df = pd.DataFrame(results['results'])
                    
                    st.download_button(
                        label="📊 ดาวน์โหลดผลลัพธ์เป็น CSV",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name="agent_simulation_results.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {str(e)}")
            st.error(traceback.format_exc())
    
    # แสดงประวัติการจำลอง (ถ้ามี)
    if 'history' in st.session_state and st.session_state.history:
        st.header("📜 ประวัติการจำลอง")
        st.write(f"มีประวัติการจำลองทั้งหมด {len(st.session_state.history)} รายการ")
        
        # เลือกจำนวนประวัติที่จะแสดง
        max_to_show = min(len(st.session_state.history), st.session_state.max_history_shown)
        
        # ใช้ slider เฉพาะเมื่อมีประวัติมากกว่า 1 รายการ
        if len(st.session_state.history) > 1:
            entries_to_show = st.slider(
                "จำนวนประวัติที่แสดง", 
                min_value=1, 
                max_value=min(len(st.session_state.history), MAX_HISTORY_ITEMS), 
                value=max_to_show
            )
            st.session_state.max_history_shown = entries_to_show
        else:
            # ถ้ามีแค่ 1 รายการ ไม่ต้องใช้ slider
            entries_to_show = 1
            st.session_state.max_history_shown = 1
        
        # แสดงประวัติย้อนหลัง
        for i in range(1, entries_to_show + 1):
            idx = len(st.session_state.history) - i
            if idx < 0:
                break
                
            hist = st.session_state.history[idx]
            with st.expander(f"การจำลองที่ {idx+1} - เวลา {hist['timestamp']} ({hist['model_type']})", expanded=False):
                # แสดงพารามิเตอร์
                st.subheader("พารามิเตอร์")
                params = hist['params']
                
                if hist['model_type'] == 'simple':
                    params_text = f"""
                    - N = {params['N']}
                    - K Effectuation = {params['K_EFFECTUATION'] if params['K_EFFECTUATION'] else 'ไม่ได้ใช้'}
                    - K Causation = {params['K_CAUSATION'] if params['K_CAUSATION'] else 'ไม่ได้ใช้'}
                    - กลยุทธ์: {', '.join(params['strategies'])}
                    - C: {', '.join([str(c) for c in params['C_values']])}
                    - rho: {', '.join([str(r) for r in params['rho_values']])}
                    - เกณฑ์การยอมรับ: {', '.join([str(t) for t in params['thresholds']])}
                    - จำนวน time steps: {params['time_steps']}
                    - ขั้นตอนก่อน Chasm: {params['pre_chasm_steps']}
                    """
                    st.markdown(params_text)
                    
                    # แสดงตารางสรุป
                    st.subheader("ตารางสรุป")
                    st.dataframe(hist['summary'], use_container_width=True)
                    
                    # ปุ่มดาวน์โหลด
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label=f"📊 ดาวน์โหลดข้อมูลละเอียดชุดที่ {idx+1}",
                            data=hist['data'].to_csv(index=False).encode('utf-8'),
                            file_name=f"history_{idx+1}_detailed.csv",
                            mime="text/csv"
                        )
                    with col2:
                        st.download_button(
                            label=f"📑 ดาวน์โหลดข้อมูลสรุปชุดที่ {idx+1}",
                            data=hist['summary'].to_csv(index=False).encode('utf-8'),
                            file_name=f"history_{idx+1}_summary.csv",
                            mime="text/csv"
                        )
                else:  # agent model
                    fitness_params = params.get('fitness_params', {})
                    params_text = f"""
                    - N = {params['N']}
                    - C = {', '.join([str(c) for c in params['c_values']])}
                    - จำนวนขั้นตอนการค้นหา: {params['steps']}
                    - จำนวนรอบการจำลอง: {params['runs']}
                    """
                    
                    if fitness_params:
                        params_text += f"""
                        - K สำหรับ Effectuation: {fitness_params.get('k_effectuation', 2)}
                        - K สำหรับ Causation: {fitness_params.get('k_causation', 5)}
                        - น้ำหนักสำหรับบิตของตนเอง: {fitness_params.get('own_weight', 0.4)}
                        - น้ำหนักสำหรับบิตจากเอเจนต์อื่น: {fitness_params.get('cross_weight', -0.2)}
                        """
                    
                    st.markdown(params_text)
                    
                    # แสดงกราฟ
                    st.subheader("กราฟผลลัพธ์")
                    plot_agent_model_results(hist['results'], unique_id=f"history_{idx}")
        
        # ปุ่มล้างประวัติ
        if st.button("🗑️ ล้างประวัติการจำลองทั้งหมด"):
            st.session_state.history = []
            st.success("ล้างประวัติการจำลองเรียบร้อยแล้ว")
            st.rerun()

# รันแอปพลิเคชัน
if __name__ == "__main__":
    main()