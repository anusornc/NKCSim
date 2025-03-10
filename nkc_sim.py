import streamlit as st
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple
import traceback

# ===============================
# กำหนดค่าคงที่
# ===============================
MAX_HISTORY_ITEMS = 5  # จำกัดจำนวนประวัติที่เก็บไว้
MAX_VARIABLES = 30     # จำกัดจำนวนตัวแปรสูงสุด

# ===============================
# ฟังก์ชันการจำลอง (ปรับปรุงแบบจำลองให้สมจริงมากขึ้น)
# ===============================
def run_simulation(
    N: int, 
    strategy_k: int, 
    C: float, 
    rho: float, 
    threshold: float, 
    seed: int, 
    time_steps: int, 
    pre_chasm_steps: int,
    interaction_strength: float = 0.2  # เพิ่มปฏิสัมพันธ์ระหว่างตัวแปร
) -> Dict[str, Any]:
    """
    จำลองการยอมรับนวัตกรรมด้วยแบบจำลอง NKC
    
    Args:
        N: จำนวนตัวแปรทั้งหมด
        strategy_k: จำนวนตัวแปรที่เลือกใช้
        C: ระดับความซับซ้อน (ความผันผวน)
        rho: สัมประสิทธิ์การเปลี่ยนแปลง
        threshold: เกณฑ์การยอมรับ
        seed: random seed
        time_steps: จำนวนขั้นตอนการจำลอง
        pre_chasm_steps: จุดแบ่งก่อน/หลัง Chasm
        interaction_strength: ความแรงของปฏิสัมพันธ์ระหว่างตัวแปร

    Returns:
        Dictionary ผลลัพธ์การจำลอง
    """
    try:
        np.random.seed(seed)
        
        # สร้างตัวแปรเริ่มต้น
        variables = np.random.normal(loc=5, scale=2, size=N)
        variables = np.clip(variables, 1, 12)
        
        # สร้าง interaction matrix (เพิ่มเติม)
        interaction_matrix = np.random.normal(0, interaction_strength, size=(N, N))
        np.fill_diagonal(interaction_matrix, 0)  # ไม่มีปฏิสัมพันธ์กับตัวเอง
        
        # เลือกตัวแปรและคำนวณค่าเริ่มต้น
        selected_vars = np.random.choice(N, strategy_k, replace=False)
        initial_mean = np.mean(variables[selected_vars])
        
        pre_adoption = 0
        post_adoption = 0
        averages = []
        threshold_crossed = False
        threshold_crossed_step = -1
        
        # วนลูปตามจำนวน time steps
        for step in range(time_steps):
            if step > 0:
                # คำนวณปฏิสัมพันธ์ระหว่างตัวแปร
                interaction_effects = np.zeros(N)
                for i in range(N):
                    interaction_effects[i] = np.sum(interaction_matrix[i, :] * variables) / N
                
                # สร้าง noise
                noise = np.random.normal(0, scale=C, size=strategy_k)
                
                # ปรับ trend_factor เป็นฟังก์ชันลอการิทึม (ปรับปรุง)
                # ใช้ S-curve แทนเพื่อจำลองการแพร่กระจายนวัตกรรมได้สมจริงขึ้น
                s_curve_factor = 2.0 / (1.0 + np.exp(-0.1 * (step - time_steps/2))) - 1.0
                trend_factor = 0.5 * np.log1p(step) + s_curve_factor * 0.3
                
                # อัปเดตตัวแปร
                variables[selected_vars] = (
                    variables[selected_vars] * rho + 
                    (1 - rho) * initial_mean + 
                    noise * (1 - rho) + 
                    trend_factor + 
                    interaction_effects[selected_vars] * (1 - rho)  # เพิ่มผลของปฏิสัมพันธ์
                )
                
                # จำกัดค่าตัวแปร
                variables = np.clip(variables, 1, 12)
        
            # คำนวณค่าเฉลี่ย
            avg = np.mean(variables[selected_vars])
            averages.append(avg)
            
            # ตรวจสอบเกณฑ์การยอมรับ
            if avg >= threshold:
                if not threshold_crossed:
                    threshold_crossed = True
                    threshold_crossed_step = step
                    
                if step < pre_chasm_steps:
                    pre_adoption += 1
                else:
                    post_adoption += 1
        
        # สร้าง dictionary ผลลัพธ์
        result = {
            'กลยุทธ์': 'Effectuation' if strategy_k < N/2 else 'Causation',
            'N': N,
            'K': strategy_k,
            'C': C,
            'rho': rho,
            'เกณฑ์การยอมรับ': threshold,
            'seed': seed,
            'อัตราการยอมรับก่อน Chasm (%)': round(pre_adoption / pre_chasm_steps * 100, 2) if pre_chasm_steps > 0 else 0,
            'อัตราการยอมรับหลัง Chasm (%)': round(post_adoption / (time_steps - pre_chasm_steps) * 100, 2) if (time_steps - pre_chasm_steps) > 0 else 0,
            'step_ข้าม_threshold': threshold_crossed_step,
            'averages': averages
        }
        return result
    
    except Exception as e:
        # จัดการข้อผิดพลาด
        error_msg = f"Error in simulation with N={N}, K={strategy_k}, C={C}, rho={rho}, threshold={threshold}, seed={seed}: {str(e)}"
        st.error(error_msg)
        return {
            'กลยุทธ์': 'Error',
            'N': N,
            'K': strategy_k,
            'C': C,
            'rho': rho,
            'เกณฑ์การยอมรับ': threshold,
            'seed': seed,
            'error': str(e),
            'อัตราการยอมรับก่อน Chasm (%)': 0,
            'อัตราการยอมรับหลัง Chasm (%)': 0,
            'averages': []
        }

# ===============================
# ฟังก์ชันจัดกลุ่มข้อมูลสำหรับกราฟ 3D
# ===============================
def sample_representative_data(df: pd.DataFrame, max_points: int = 1000) -> pd.DataFrame:
    """
    เลือกข้อมูลตัวแทนที่ดีกว่าการสุ่ม
    
    Args:
        df: DataFrame ที่ต้องการสุ่มตัวอย่าง
        max_points: จำนวนจุดข้อมูลสูงสุดที่ต้องการ
    
    Returns:
        DataFrame ของตัวแทนข้อมูล
    """
    if len(df) <= max_points:
        return df
    
    # จัดกลุ่มตามมิติสำคัญ
    grouped = df.groupby(['กลยุทธ์', 'C', 'rho', 'เกณฑ์การยอมรับ'])
    
    result = []
    for _, group in grouped:
        # คำนวณจำนวนจุดที่ต้องการจากแต่ละกลุ่ม
        n_samples = max(1, int(max_points * len(group) / len(df)))
        
        # เลือกจุดข้อมูลอย่างสม่ำเสมอ
        step = max(1, len(group) // n_samples)
        samples = group.iloc[::step, :]
        
        # เลือกเพิ่มถ้าได้น้อยเกินไป
        if len(samples) < n_samples and len(group) > n_samples:
            additional = group.sample(n_samples - len(samples))
            samples = pd.concat([samples, additional])
        
        result.append(samples)
    
    return pd.concat(result)

# ===============================
# ฟังก์ชันช่วยตรวจสอบ seeds
# ===============================
def parse_seeds(seed_input: str) -> List[int]:
    """
    แปลงข้อความ seeds เป็นรายการตัวเลข
    
    Args:
        seed_input: ข้อความที่มี seeds คั่นด้วยเครื่องหมายจุลภาค
    
    Returns:
        รายการ seeds ที่เป็นตัวเลข
    
    Raises:
        ValueError: หากรูปแบบไม่ถูกต้อง
    """
    if not seed_input or seed_input.strip() == "":
        raise ValueError("กรุณากรอก Random Seeds")
    
    try:
        # แยกด้วยเครื่องหมายจุลภาคและแปลงเป็นตัวเลข
        seeds = []
        for s in seed_input.split(","):
            cleaned = s.strip()
            if cleaned:  # ตรวจสอบว่าไม่ว่างเปล่า
                seeds.append(int(cleaned))
        
        if not seeds:
            raise ValueError("ไม่พบ seeds ที่ถูกต้อง")
        
        return seeds
    except Exception as e:
        raise ValueError(f"กรุณากรอก Random Seeds เป็นตัวเลขคั่นด้วยเครื่องหมายจุลภาค (เช่น 42, 123): {str(e)}")

# ===============================
# ฟังก์ชันเตรียม session state
# ===============================
def initialize_session_state():
    """เตรียม session state เริ่มต้น"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'max_history_shown' not in st.session_state:
        st.session_state.max_history_shown = 3

# ===============================
# UI ด้วย Streamlit
# ===============================
def main():
    st.title("🚀 NKC Model Simulation (ปรับปรุงแนวโน้มและปฏิสัมพันธ์)")
    
    # เตรียม session state
    initialize_session_state()
    
    # คำอธิบายแบบจำลอง
    with st.expander("📝 ภาพรวมแบบจำลอง", expanded=False):
        st.markdown("""
        ### แบบจำลอง NKC
        จำลองการยอมรับนวัตกรรมระหว่างสองกลยุทธ์:
        - **Effectuation**: กลยุทธ์แบบยืดหยุ่น ใช้ตัวแปรน้อย (K น้อย)
        - **Causation**: กลยุทธ์แบบมีโครงสร้าง ใช้ตัวแปรมาก (K มาก)

        **พารามิเตอร์หลัก:**
        - **N**: จำนวนตัวแปรทั้งหมดในระบบ
        - **K**: จำนวนตัวแปรที่ใช้ในกลยุทธ์ (K < N/2 คือ Effectuation, K ≥ N/2 คือ Causation)
        - **C**: ระดับความซับซ้อนหรือความผันผวน (สูง = ผันผวนมาก)
        - **rho**: สัมประสิทธิ์การเปลี่ยนแปลง (สูง = เปลี่ยนช้า, ต่ำ = เปลี่ยนเร็ว)
        - **เกณฑ์การยอมรับ**: ค่าเฉลี่ยขั้นต่ำที่ถือว่ายอมรับนวัตกรรม
        - **Chasm**: จุดแบ่งระหว่างผู้ใช้กลุ่มแรก (Early Adopters) และตลาดส่วนใหญ่ (Mainstream Market)
        """)
    
    # แถบด้านข้าง
    with st.sidebar:
        st.header("⚙️ ตั้งค่าการจำลอง")
        
        # เลือกโหมดการตั้งค่า
        sim_mode = st.radio("โหมดการตั้งค่า", ["พื้นฐาน", "ขั้นสูง"], index=0)
        
        # พารามิเตอร์หลัก
        st.subheader("พารามิเตอร์หลัก")
        
        # จำนวนตัวแปร N
        N = st.number_input(
            "จำนวนตัวแปรทั้งหมด (N)", 
            min_value=5, 
            max_value=MAX_VARIABLES, 
            value=10, 
            help="จำนวนตัวแปรในระบบ (5-30)"
        )
        
        # คำนวณค่าเริ่มต้นที่เหมาะสมสำหรับ K
        default_k_effectuation = max(1, int(N * 0.2))  # ประมาณ 20% ของ N
        default_k_causation = max(2, int(N * 0.6))     # ประมาณ 60% ของ N
        
        # ค่า K สำหรับแต่ละกลยุทธ์
        col1, col2 = st.columns(2)
        with col1:
            K_EFFECTUATION = st.number_input(
                "K (Effectuation)", 
                min_value=1, 
                max_value=N-1, 
                value=default_k_effectuation, 
                help="จำนวนตัวแปรที่ใช้ในกลยุทธ์ยืดหยุ่น"
            )
        with col2:
            K_CAUSATION = st.number_input(
                "K (Causation)", 
                min_value=1, 
                max_value=N-1, 
                value=default_k_causation, 
                help="จำนวนตัวแปรที่ใช้ในกลยุทธ์มีโครงสร้าง"
            )
        
        # เลือกกลยุทธ์
        strategies = st.multiselect(
            "เลือกกลยุทธ์",
            ["Effectuation", "Causation"],
            default=["Effectuation", "Causation"],
            help="Effectuation: ใช้ข้อมูลน้อย | Causation: ใช้ข้อมูลมาก"
        )
        
        # ระดับความซับซ้อน
        C_values = st.multiselect(
            "ระดับความซับซ้อน (C)",
            [0.5, 1.0, 1.5, 2.0, 3.0],
            default=[1.0, 3.0],
            help="C ต่ำ = เสถียร, C สูง = ผันผวน (หน่วย: ค่าเบี่ยงเบนมาตรฐาน)"
        )
        
        # สัมประสิทธิ์การเปลี่ยนแปลง
        rho_values = st.multiselect(
            "สัมประสิทธิ์การเปลี่ยนแปลง (rho)",
            [0.1, 0.3, 0.5, 0.7, 0.9],
            default=[0.1, 0.9],
            help="0.1 = เปลี่ยนเร็ว, 0.9 = เปลี่ยนช้า (หน่วย: สัดส่วน)"
        )
        
        # เกณฑ์การยอมรับ
        thresholds = st.multiselect(
            "เกณฑ์การยอมรับ",
            [5.0, 6.0, 7.0, 8.0],
            default=[5.0, 7.0],
            help="ค่าเฉลี่ยขั้นต่ำที่ถือว่ายอมรับนวัตกรรม (หน่วย: คะแนน)"
        )
        
        # Random Seeds
        seeds_input = st.text_input(
            "Random Seeds (คั่นด้วยเครื่องหมายจุลภาค)",
            "42, 123",
            help="ตัวเลขสำหรับการสุ่ม (เช่น 42, 123)"
        )
        
        # ตัวเลือกขั้นสูง (แสดงเฉพาะเมื่อเลือกโหมดขั้นสูง)
        advanced_params = {}
        if sim_mode == "ขั้นสูง":
            st.subheader("ตัวเลือกขั้นสูง")
            
            advanced_params['time_steps'] = st.number_input(
                "จำนวนขั้นตอนการจำลอง", 
                min_value=20, 
                max_value=500, 
                value=100, 
                help="จำนวนรอบการจำลอง (หน่วย: รอบ)"
            )
            
            advanced_params['pre_chasm_steps'] = st.number_input(
                "ขั้นตอนก่อน Chasm", 
                min_value=5, 
                max_value=advanced_params.get('time_steps', 100)-5, 
                value=16, 
                help="จุดแบ่งก่อน/หลัง Chasm (หน่วย: รอบ)"
            )
            
            advanced_params['max_simulations'] = st.number_input(
                "จำนวนการจำลองสูงสุด", 
                min_value=1, 
                max_value=200, 
                value=50, 
                help="จำกัดการรันเพื่อประสิทธิภาพ"
            )
            
            advanced_params['interaction_strength'] = st.slider(
                "ความแรงของปฏิสัมพันธ์ระหว่างตัวแปร", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.2, 
                step=0.05,
                help="ระดับปฏิสัมพันธ์ระหว่างตัวแปรต่างๆ (0 = ไม่มี, 0.5 = สูง)"
            )
            
            advanced_params['save_history'] = st.checkbox(
                "บันทึกประวัติการจำลอง", 
                value=True, 
                help="บันทึกผลการจำลองเพื่อเปรียบเทียบภายหลัง"
            )
        else:
            # ค่าเริ่มต้นสำหรับโหมดพื้นฐาน
            advanced_params = {
                'time_steps': 100,
                'pre_chasm_steps': 16,
                'max_simulations': 50,
                'interaction_strength': 0.2,
                'save_history': True
            }
        
        # ตั้งค่ากราฟ
        st.subheader("ตั้งค่ากราฟ")
        graph_type = st.selectbox(
            "เลือกประเภทกราฟ", 
            ["2D แยกตามเกณฑ์", "3D รวม Time Steps", "2D Time Series", "แสดงทั้งหมด"], 
            index=0
        )
        
        # ตั้งค่าเพิ่มเติมสำหรับแต่ละประเภทกราฟ
        graph_params = {}
        if graph_type == "2D แยกตามเกณฑ์" or graph_type == "แสดงทั้งหมด":
            graph_params['y_axis'] = st.selectbox(
                "เลือกข้อมูลในแกน Y", 
                ["อัตราการยอมรับก่อน Chasm (%)", "อัตราการยอมรับหลัง Chasm (%)"], 
                index=1
            )
            graph_params['x_axis'] = st.selectbox(
                "เลือกข้อมูลในแกน X", 
                ["rho", "C"], 
                index=0
            )
            
        if graph_type == "3D รวม Time Steps" or graph_type == "2D Time Series" or graph_type == "แสดงทั้งหมด":
            graph_params['color_by'] = st.selectbox(
                "เลือกสีตาม", 
                ["C", "กลยุทธ์", "rho"], 
                index=0, 
                help="เลือกว่าสีจะแสดง C หรือ กลยุทธ์"
            )
            graph_params['time_step_range'] = st.slider(
                "เลือกช่วง Time Step", 
                0, 
                advanced_params.get('time_steps', 100), 
                (16, advanced_params.get('time_steps', 100)), 
                help="กรองข้อมูลตามช่วง Time Step (หน่วย: รอบ)"
            )
        
        # ปุ่มเริ่มการจำลอง
        st.markdown("---")
        run_button = st.button("▶️ เริ่มการจำลอง", use_container_width=True)
    
    # ส่วนหลัก - ประมวลผลและแสดงผล
    if run_button:
        try:
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
            params = []
            for k in k_list:
                for C in C_values:
                    for rho in rho_values:
                        for thresh in thresholds:
                            for seed in seeds:
                                params.append((
                                    N, k, C, rho, thresh, seed, 
                                    advanced_params['time_steps'],
                                    advanced_params['pre_chasm_steps'],
                                    advanced_params['interaction_strength']
                                ))
            
            # ตรวจสอบจำนวนการจำลอง
            if len(params) > advanced_params['max_simulations']:
                st.warning(f"จำนวนการจำลองทั้งหมด ({len(params)}) เกินขีดจำกัด ({advanced_params['max_simulations']}) จะใช้เพียง {advanced_params['max_simulations']} ชุดแรก")
                params = params[:advanced_params['max_simulations']]
            
            # แสดงระหว่างประมวลผล
            with st.status("กำลังรันการจำลอง...") as status:
                st.write(f"รันการจำลองจำนวน {len(params)} ชุด...")
                
                # ประมวลผลแบบขนาน
                with st.spinner():
                    results = Parallel(n_jobs=-1, max_nbytes=None)(delayed(run_simulation)(*p) for p in params)
                
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
                        'จำนวนตัวแปรทั้งหมด (N)', 
                        'K สำหรับ Effectuation', 
                        'K สำหรับ Causation',
                        'กลยุทธ์ที่เลือก',
                        'ระดับความซับซ้อน (C)',
                        'สัมประสิทธิ์การเปลี่ยนแปลง (rho)',
                        'เกณฑ์การยอมรับ',
                        'จำนวนขั้นตอนการจำลอง',
                        'ขั้นตอนก่อน Chasm',
                        'ความแรงของปฏิสัมพันธ์'
                    ],
                    'ค่า': [
                        N,
                        K_EFFECTUATION,
                        K_CAUSATION,
                        ', '.join(strategies),
                        ', '.join([str(c) for c in C_values]),
                        ', '.join([str(r) for r in rho_values]),
                        ', '.join([str(t) for t in thresholds]),
                        advanced_params['time_steps'],
                        advanced_params['pre_chasm_steps'],
                        advanced_params['interaction_strength']
                    ]
                })
                st.dataframe(params_df, use_container_width=True)
            
            # แสดงตารางสรุป
            st.subheader("ตารางสรุป")
            st.dataframe(summary, use_container_width=True)
            
            # แสดงกราฟตามประเภทที่เลือก
            def plot_2d_by_threshold():
                """สร้างกราฟ 2D แยกตามเกณฑ์"""
                for thresh in thresholds:
                    with st.expander(f"กราฟสำหรับเกณฑ์การยอมรับ = {thresh}", expanded=True):
                        filtered_df = df[df['เกณฑ์การยอมรับ'] == thresh]
                        if filtered_df.empty:
                            st.warning(f"ไม่มีข้อมูลสำหรับเกณฑ์การยอมรับ = {thresh}")
                            continue
                        
                        fig = px.scatter(
                            filtered_df,
                            x=graph_params['x_axis'],
                            y=graph_params['y_axis'],
                            color="กลยุทธ์",
                            facet_col="C" if graph_params['x_axis'] == 'rho' else 'rho',
                            title=f"{graph_params['y_axis']} (เกณฑ์ = {thresh})",
                            hover_data=['N', 'K', 'seed'],
                            labels={
                                'rho': 'สัมประสิทธิ์การเปลี่ยนแปลง (rho)',
                                'C': 'ระดับความซับซ้อน (C)',
                                graph_params['y_axis']: graph_params['y_axis']
                            },
                            height=400
                        )
                        # ปรับแต่งกราฟให้อ่านง่าย
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=50, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # แสดงกราฟเวลาที่ข้ามเกณฑ์
                        fig_cross = px.scatter(
                            filtered_df[filtered_df['step_ข้าม_threshold'] >= 0],
                            x=graph_params['x_axis'],
                            y="step_ข้าม_threshold",
                            color="กลยุทธ์",
                            facet_col="C" if graph_params['x_axis'] == 'rho' else 'rho',
                            title=f"จำนวน Time Step ที่ข้ามเกณฑ์ = {thresh}",
                            hover_data=['N', 'K', 'seed'],
                            labels={
                                'rho': 'สัมประสิทธิ์การเปลี่ยนแปลง (rho)',
                                'C': 'ระดับความซับซ้อน (C)',
                                'step_ข้าม_threshold': 'Time Step ที่ข้ามเกณฑ์'
                            },
                            height=400
                        )
                        fig_cross.update_layout(
                            margin=dict(l=20, r=20, t=50, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_cross, use_container_width=True)
            
            def plot_3d():
                """สร้างกราฟ 3D"""
                st.subheader("กราฟ 3D")
                
                # เตรียมข้อมูลสำหรับกราฟ 3D
                time_data = []
                for result in results:
                    if 'error' in result:
                        continue
                    
                    for t, avg in enumerate(result['averages']):
                        if graph_params['time_step_range'][0] <= t <= graph_params['time_step_range'][1]:
                            time_data.append({
                                'กลยุทธ์': result['กลยุทธ์'],
                                'C': result['C'],
                                'rho': result['rho'],
                                'เกณฑ์การยอมรับ': result['เกณฑ์การยอมรับ'],
                                'Time Step': t,
                                'ค่าเฉลี่ย': avg,
                                'ข้ามเกณฑ์': avg >= result['เกณฑ์การยอมรับ']
                            })
                
                time_df = pd.DataFrame(time_data)
                
                if time_df.empty:
                    st.error("ไม่มีข้อมูลสำหรับกราฟ 3D")
                    return
                
                # เลือกตัวแทนข้อมูลแทนการสุ่ม
                sample_df = sample_representative_data(time_df, max_points=2000)
                
                color_col = graph_params['color_by']
                
                # กำหนดสีที่เหมาะสม
                if color_col == 'กลยุทธ์':
                    colorscale = 'Bluered'  # สีแยกชัดเจนระหว่างกลยุทธ์
                    colorbar_title = 'กลยุทธ์'
                    marker_colormap = sample_df[color_col].map({'Effectuation': 0, 'Causation': 1})
                elif color_col == 'C':
                    colorscale = 'Viridis'  # ไล่ระดับตาม C
                    colorbar_title = 'ระดับความซับซ้อน (C)'
                    marker_colormap = sample_df[color_col]
                else:  # rho
                    colorscale = 'Plasma'  # ไล่ระดับตาม rho
                    colorbar_title = 'สัมประสิทธิ์การเปลี่ยนแปลง (rho)'
                    marker_colormap = sample_df[color_col]
                
                # สร้างกราฟ 3D
                fig = go.Figure(data=[go.Scatter3d(
                    x=sample_df['rho'],
                    y=sample_df['Time Step'],
                    z=sample_df['ค่าเฉลี่ย'],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=marker_colormap,
                        colorscale=colorscale,
                        showscale=True,
                        colorbar_title=colorbar_title,
                        opacity=0.7
                    ),
                    text=[f"กลยุทธ์: {row['กลยุทธ์']}<br>C: {row['C']}<br>rho: {row['rho']}<br>ค่าเฉลี่ย: {row['ค่าเฉลี่ย']:.2f}<br>เกณฑ์: {row['เกณฑ์การยอมรับ']}" 
                          for _, row in sample_df.iterrows()],
                    hoverinfo='text'
                )])
                
                # ปรับแต่งกราฟ
                fig.update_layout(
                    title=f"กราฟ 3D: rho vs Time Step vs ค่าเฉลี่ย (สีตาม {color_col})",
                    scene=dict(
                        xaxis_title="สัมประสิทธิ์การเปลี่ยนแปลง (rho)",
                        yaxis_title="Time Step",
                        zaxis_title="ค่าเฉลี่ย"
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    height=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            def plot_time_series():
                """สร้างกราฟ Time Series"""
                st.subheader("กราฟ Time Series")
                
                # เตรียมข้อมูลสำหรับกราฟ time series
                time_data = []
                for result in results:
                    if 'error' in result:
                        continue
                    
                    for t, avg in enumerate(result['averages']):
                        if graph_params['time_step_range'][0] <= t <= graph_params['time_step_range'][1]:
                            time_data.append({
                                'กลยุทธ์': result['กลยุทธ์'],
                                'C': result['C'],
                                'rho': result['rho'],
                                'เกณฑ์การยอมรับ': result['เกณฑ์การยอมรับ'],
                                'Time Step': t,
                                'ค่าเฉลี่ย': avg,
                                'ข้ามเกณฑ์': avg >= result['เกณฑ์การยอมรับ'],
                                'id': f"{result['กลยุทธ์']}_{result['C']}_{result['rho']}_{result['เกณฑ์การยอมรับ']}_{result['seed']}"
                            })
                
                time_df = pd.DataFrame(time_data)
                
                if time_df.empty:
                    st.error("ไม่มีข้อมูลสำหรับกราฟ Time Series")
                    return
                
                # กรองข้อมูลตามเกณฑ์การยอมรับ
                for thresh in thresholds:
                    with st.expander(f"Time Series สำหรับเกณฑ์การยอมรับ = {thresh}", expanded=True):
                        thresh_df = time_df[time_df['เกณฑ์การยอมรับ'] == thresh]
                        
                        color_col = graph_params['color_by']
                        
                        # สร้างคอลัมน์รวมสำหรับการจัดกลุ่ม
                        if color_col == 'กลยุทธ์':
                            thresh_df['group'] = thresh_df['กลยุทธ์']
                            facet_col = 'C'
                            facet_row = 'rho'
                        elif color_col == 'C':
                            thresh_df['group'] = thresh_df['C'].astype(str)
                            facet_col = 'กลยุทธ์'
                            facet_row = 'rho'
                        else:  # rho
                            thresh_df['group'] = thresh_df['rho'].astype(str)
                            facet_col = 'กลยุทธ์'
                            facet_row = 'C'
                        
                        # สร้างกราฟเส้น
                        fig = px.line(
                            thresh_df,
                            x="Time Step",
                            y="ค่าเฉลี่ย",
                            color="group",
                            facet_col=facet_col,
                            facet_row=facet_row,
                            title=f"กราฟ Time Series: ค่าเฉลี่ยในแต่ละ Time Step (เกณฑ์ = {thresh})",
                            labels={
                                'Time Step': 'Time Step',
                                'ค่าเฉลี่ย': 'ค่าเฉลี่ย',
                                'group': color_col
                            },
                            line_group='id',  # แยกเส้นตาม simulation
                            height=600
                        )
                        
                        # เพิ่มเส้นเกณฑ์การยอมรับ โดยใช้วิธีที่เรียบง่ายกว่า
                        # แทนที่จะพยายามเพิ่มเส้นในแต่ละ facet ให้ใช้ add_hline ซึ่งจะเพิ่มเส้นทั้งหมดโดยอัตโนมัติ
                        fig.add_hline(
                            y=thresh,
                            line=dict(color='red', width=2, dash='dash'),
                            annotation_text=f"เกณฑ์: {thresh}",
                            annotation_position="top right"
                        )
                        
                        # ปรับความโปร่งใส
                        fig.update_traces(opacity=0.5)
                        
                        # ปรับแต่งกราฟ
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=50, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            # แสดงกราฟตามประเภทที่เลือก
            st.header("📈 กราฟแสดงผล")
            if graph_type == "2D แยกตามเกณฑ์":
                plot_2d_by_threshold()
            elif graph_type == "3D รวม Time Steps":
                plot_3d()
            elif graph_type == "2D Time Series":
                plot_time_series()
            else:  # แสดงทั้งหมด
                plot_2d_by_threshold()
                plot_3d()
                plot_time_series()
            
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
            with st.expander(f"การจำลองที่ {idx+1} - เวลา {hist['timestamp']}", expanded=False):
                # แสดงพารามิเตอร์
                st.subheader("พารามิเตอร์")
                params = hist['params']
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
        
        # ปุ่มล้างประวัติ
        if st.button("🗑️ ล้างประวัติการจำลองทั้งหมด"):
            st.session_state.history = []
            st.success("ล้างประวัติการจำลองเรียบร้อยแล้ว")
            st.rerun()

# รันแอปพลิเคชัน
if __name__ == "__main__":
    main()