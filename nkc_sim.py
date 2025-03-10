import streamlit as st
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import plotly.express as px
import plotly.graph_objects as go

# ===============================
# ฟังก์ชันการจำลอง
# ===============================
def run_simulation(N, strategy_k, C, rho, threshold, seed, time_steps, pre_chasm_steps):
    np.random.seed(seed)
    
    variables = np.random.normal(loc=5, scale=2, size=N)
    variables = np.clip(variables, 1, 9)
    
    selected_vars = np.random.choice(N, strategy_k, replace=False)
    initial_mean = np.mean(variables[selected_vars])
    
    pre_adoption = 0
    post_adoption = 0
    averages = []
    
    for step in range(time_steps):
        if step > 0:
            noise = np.random.normal(0, scale=C, size=strategy_k)
            variables[selected_vars] = variables[selected_vars] * rho + \
                                       (1 - rho) * initial_mean + \
                                       noise * (1 - rho)
            variables = np.clip(variables, 1, 9)
        
        avg = np.mean(variables[selected_vars])
        averages.append(avg)
        
        if avg >= threshold:
            if step < pre_chasm_steps:
                pre_adoption += 1
            else:
                post_adoption += 1
    
    return {
        'กลยุทธ์': 'Effectuation' if strategy_k < N/2 else 'Causation',
        'N': N,
        'K': strategy_k,
        'C': C,
        'rho': rho,
        'เกณฑ์การยอมรับ': threshold,
        'seed': seed,
        'อัตราการยอมรับก่อน Chasm (%)': round(pre_adoption / pre_chasm_steps * 100, 2),
        'อัตราการยอมรับหลัง Chasm (%)': round(post_adoption / (time_steps - pre_chasm_steps) * 100, 2),
        'averages': averages
    }

# ===============================
# UI ด้วย Streamlit
# ===============================
st.title("🚀 NKC Model Simulation (ฉบับปรับแกน Time Step)")

st.markdown("""
### 📝 ภาพรวม
จำลองการยอมรับนวัตกรรมระหว่าง **Effectuation** (ยืดหยุ่น) และ **Causation** (มีโครงสร้าง)  
ตั้งค่าการจำลองและกราฟทั้งหมดในแถบข้าง (Sidebar)
""")

# Sidebar
st.sidebar.header("⚙️ ตั้งค่าการจำลอง")

# พารามิเตอร์หลัก
st.sidebar.subheader("พารามิเตอร์หลัก")
N = st.sidebar.number_input("จำนวนตัวแปรทั้งหมด (N)", min_value=5, max_value=20, value=10, help="จำนวนตัวแปรในระบบ")
K_EFFECTUATION = st.sidebar.number_input("K สำหรับ Effectuation", min_value=1, max_value=N-1, value=2, help="จำนวนตัวแปรที่ใช้ในกลยุทธ์ยืดหยุ่น")
K_CAUSATION = st.sidebar.number_input("K สำหรับ Causation", min_value=1, max_value=N-1, value=5, help="จำนวนตัวแปรที่ใช้ในกลยุทธ์มีโครงสร้าง")

strategies = st.sidebar.multiselect(
    "เลือกกลยุทธ์",
    ["Effectuation", "Causation"],
    default=["Effectuation", "Causation"],
    help="Effectuation: ใช้ข้อมูลน้อย | Causation: ใช้ข้อมูลมาก"
)

C_values = st.sidebar.multiselect(
    "ระดับความซับซ้อน (C)",
    [1, 7, 9],
    default=[1, 9],
    help="1=ต่ำ (เสถียร), 7=ปานกลาง, 9=สูง (ผันผวน)"
)

rho_values = st.sidebar.multiselect(
    "สัมประสิทธิ์การเปลี่ยนแปลง (rho)",
    [0.1, 0.7, 0.9],
    default=[0.1, 0.9],
    help="0.1=เปลี่ยนเร็ว, 0.7=ปานกลาง, 0.9=เปลี่ยนช้า"
)

thresholds = st.sidebar.multiselect(
    "เกณฑ์การยอมรับ",
    [5, 6, 7],
    default=[5, 7],
    help="ค่าเฉลี่ยขั้นต่ำที่ถือว่ายอมรับนวัตกรรม"
)

seeds_input = st.sidebar.text_input(
    "Random Seeds (คั่นด้วยเครื่องหมายจุลภาค)",
    "42, 123",
    help="ตัวเลขสำหรับการสุ่ม (เช่น 42, 123)"
)

# ตัวเลือกขั้นสูง
with st.sidebar.expander("ตัวเลือกขั้นสูง"):
    time_steps = st.number_input("จำนวนขั้นตอนการจำลอง", min_value=50, value=100, help="จำนวนรอบการจำลอง")
    pre_chasm_steps = st.number_input("ขั้นตอนก่อน Chasm", min_value=5, max_value=time_steps-1, value=16, help="จุดแบ่งก่อน/หลัง Chasm")
    max_simulations = st.number_input("จำนวนการจำลองสูงสุด", min_value=1, value=50, help="จำกัดการรันเพื่อประสิทธิภาพ")

# ตัวเลือกกราฟ
st.sidebar.subheader("ตั้งค่ากราฟ")
graph_type = st.sidebar.selectbox("เลือกประเภทกราฟ", ["2D แยกตามเกณฑ์", "3D รวม Time Steps"], index=0)
if graph_type == "2D แยกตามเกณฑ์":
    y_axis = st.sidebar.selectbox("เลือกข้อมูลในแกน Y", 
                                  ["อัตราการยอมรับก่อน Chasm (%)", "อัตราการยอมรับหลัง Chasm (%)"], index=1)
    x_axis = st.sidebar.selectbox("เลือกข้อมูลในแกน X", ["rho", "C"], index=0)

# ปุ่มเริ่มการจำลอง
st.sidebar.markdown("---")
if st.sidebar.button("เริ่มการจำลอง"):
    try:
        seeds = [int(s.strip()) for s in seeds_input.split(",")]
    except ValueError:
        st.error("กรุณากรอก Random Seeds เป็นตัวเลขคั่นด้วยเครื่องหมายจุลภาค (เช่น 42, 123)")
        st.stop()

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

    k_list = []
    if "Effectuation" in strategies:
        k_list.append(K_EFFECTUATION)
    if "Causation" in strategies:
        k_list.append(K_CAUSATION)
    
    params = []
    for k in k_list:
        for C in C_values:
            for rho in rho_values:
                for thresh in thresholds:
                    for seed in seeds:
                        params.append((N, k, C, rho, thresh, seed, time_steps, pre_chasm_steps))
    
    if len(params) > max_simulations:
        st.warning(f"จำนวนการจำลองทั้งหมด ({len(params)}) เกินขีดจำกัด ({max_simulations}) จะใช้เพียง {max_simulations} ชุดแรก")
        params = params[:max_simulations]

    with st.spinner("กำลังรันการจำลอง..."):
        results = Parallel(n_jobs=-1, max_nbytes=None)(delayed(run_simulation)(*p) for p in params)
    
    df = pd.DataFrame(results)
    
    if df.empty:
        st.error("ไม่มีข้อมูลสำหรับการจำลอง กรุณาตรวจสอบพารามิเตอร์")
        st.stop()

    summary = df.groupby(['กลยุทธ์', 'C', 'rho', 'เกณฑ์การยอมรับ']).agg({
        'อัตราการยอมรับก่อน Chasm (%)': ['mean', 'std'],
        'อัตราการยอมรับหลัง Chasm (%)': ['mean', 'std']
    }).reset_index()
    summary.columns = ['กลยุทธ์', 'C', 'rho', 'เกณฑ์การยอมรับ', 
                      'ก่อน Chasm Mean', 'ก่อน Chasm Std', 
                      'หลัง Chasm Mean', 'หลัง Chasm Std']

    # บันทึกประวัติ
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(df)
    st.write(f"บันทึกการจำลองครั้งที่: {len(st.session_state.history)}")

    st.header("📊 ผลลัพธ์การจำลอง")
    st.dataframe(summary)

    # แสดงกราฟ
    if graph_type == "2D แยกตามเกณฑ์":
        for thresh in thresholds:
            st.subheader(f"กราฟสำหรับเกณฑ์การยอมรับ = {thresh}")
            filtered_df = df[df['เกณฑ์การยอมรับ'] == thresh]
            if filtered_df.empty:
                st.warning(f"ไม่มีข้อมูลสำหรับเกณฑ์การยอมรับ = {thresh}")
                continue
            fig = px.scatter(
                filtered_df,
                x=x_axis,
                y=y_axis,
                color="กลยุทธ์",
                title=f"{y_axis} (เกณฑ์ = {thresh})",
                hover_data=['N', 'K', 'seed', 'C' if x_axis != 'C' else 'rho']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # 3D รวม Time Steps (ปรับให้ Time Step เป็นแกน Y)
        time_data = []
        for result in results:
            for t, avg in enumerate(result['averages']):
                time_data.append({
                    'กลยุทธ์': result['กลยุทธ์'],
                    'C': result['C'],
                    'rho': result['rho'],
                    'เกณฑ์การยอมรับ': result['เกณฑ์การยอมรับ'],
                    'Time Step': t,
                    'ค่าเฉลี่ย': avg
                })
        time_df = pd.DataFrame(time_data)
        
        if time_df.empty:
            st.error("ไม่มีข้อมูลสำหรับกราฟ 3D")
            st.stop()
            
        sample_df = time_df.sample(min(1000, len(time_df)))
        
        # ปรับให้ Time Step เป็นแกน Y และ C เป็นสี
        fig = go.Figure(data=[go.Scatter3d(
            x=sample_df['rho'],
            y=sample_df['Time Step'],  # เปลี่ยนจาก C เป็น Time Step
            z=sample_df['ค่าเฉลี่ย'],
            mode='markers',
            marker=dict(
                size=5,
                color=sample_df['C'],  # เปลี่ยนสีให้แสดง C
                colorscale='Viridis',
                showscale=True,
                colorbar_title="ระดับความซับซ้อน (C)"
            ),
            text=sample_df['กลยุทธ์']
        )])
        fig.update_layout(
            title="กราฟ 3D: rho vs Time Step vs ค่าเฉลี่ย (สีตาม C)",
            scene=dict(
                xaxis_title="rho",
                yaxis_title="Time Step",  # ปรับชื่อแกน
                zaxis_title="ค่าเฉลี่ย"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="ดาวน์โหลดข้อมูลเป็น CSV",
        data=df.to_csv(index=False),
        file_name="simulation_results.csv",
        mime="text/csv"
    )

# แสดงประวัติการจำลอง
if 'history' in st.session_state and len(st.session_state.history) > 1:
    st.header("📜 ประวัติการจำลอง")
    for i, hist_df in enumerate(st.session_state.history):
        st.subheader(f"การจำลองครั้งที่ {i+1}")
        st.dataframe(hist_df.describe())