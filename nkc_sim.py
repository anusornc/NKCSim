import streamlit as st
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import plotly.express as px
from stqdm import stqdm  # สำหรับ progress bar

# ===============================
# 1. ตั้งค่าพารามิเตอร์จาก UI
# ===============================
st.sidebar.header("พารามิเตอร์หลัก")
st.sidebar.caption("ค่าเริ่มต้นแสดงในกล่อง สามารถแก้ไขได้")

# 1.1 กลยุทธ์
strategies = st.sidebar.multiselect(
    "เลือกกลยุทธ์",
    ["Effectuation", "Causation"],
    default=["Effectuation", "Causation"]
)

# 1.2 ตัวแปรสภาพแวดล้อม
C_values = st.sidebar.multiselect(
    "ระดับความซับซ้อน (C)",
    options=[1, 7, 9],
    default=[1, 7, 9],
    max_selections=2
)
rho_values = st.sidebar.multiselect(
    "สัมประสิทธิ์การเปลี่ยนแปลง (rho)",
    options=[0.1, 0.7, 0.9],
    default=[0.1, 0.7, 0.9],
    max_selections=2
)
thresholds = st.sidebar.multiselect(
    "เกณฑ์การยอมรับ",
    options=[5, 6, 7],
    default=[6],
    max_selections=2
)

# 1.3 พารามิเตอร์การสุ่ม
seeds_input = st.sidebar.text_input(
    "Random Seeds (คั่นด้วยจุลภาค)",
    "42,123,789"
)

# 1.4 ตัวเลือกขั้นสูง
with st.sidebar.expander("ตัวเลือกขั้นสูง"):
    N = st.number_input("จำนวนตัวแปรทั้งหมด (N)", 10, 1, 100)
    K_EFFECTUATION = st.number_input("K สำหรับ Effectuation", 2, 1, N)
    K_CAUSATION = st.number_input("K สำหรับ Causation", 5, 1, N)
    TIME_STEPS = st.number_input("จำนวนรอบจำลอง (TIME_STEPS)", 100, 10, 1000)
    PRE_CHASM_STEPS = st.number_input("รอบก่อน Chasm (PRE_CHASM_STEPS)", 16, 1, 100)

# ===============================
# 2. ฟังก์ชันการจำลอง (ปรับปรุงแล้ว)
# ===============================
def run_simulation(strategy_k, C, rho, threshold, seed):
    np.random.seed(seed)
    
    # 1. เริ่มต้นตัวแปร
    variables = np.random.normal(loc=5, scale=2, size=N)
    variables = np.clip(variables, 1, 9)
    
    # 2. เลือกตัวแปร K ตัว (สุ่มใหม่ทุกรอบหากต้องการ)
    selected_vars = np.random.choice(N, strategy_k, replace=False)
    initial_mean = np.mean(variables[selected_vars])
    
    pre_adoption = 0
    post_adoption = 0
    
    for step in range(TIME_STEPS):
        if step > 0:
            # 3. อัปเดตด้วย Mean Reversion (สูตรปรับปรุงแล้ว)
            noise = np.random.normal(0, scale=C, size=strategy_k)
            variables[selected_vars] = (
                rho * variables[selected_vars] + 
                (1 - rho) * initial_mean + 
                noise  # ไม่คูณด้วย (1 - rho)
            )
            variables = np.clip(variables, 1, 9)
        
        # 4. ตรวจสอบเกณฑ์
        avg = np.mean(variables[selected_vars])
        if avg >= threshold:
            if step < PRE_CHASM_STEPS:
                pre_adoption += 1
            else:
                post_adoption += 1
    
    return {
        'กลยุทธ์': 'Effectuation' if strategy_k == K_EFFECTUATION else 'Causation',
        'C': C,
        'rho': rho,
        'เกณฑ์การยอมรับ': threshold,
        'seed': seed,
        'pre_adoption_rate': round(pre_adoption / PRE_CHASM_STEPS * 100, 2),
        'post_adoption_rate': round(post_adoption / (TIME_STEPS - PRE_CHASM_STEPS) * 100, 2)
    }

# ===============================
# 3. การรันการจำลอง
# ===============================
if st.sidebar.button("เริ่มการจำลอง"):
    # 3.1 ตรวจสอบ Input
    if not strategies:
        st.error("โปรดเลือกอย่างน้อย 1 กลยุทธ์")
        st.stop()
        
    try:
        seeds = [int(s.strip()) for s in seeds_input.split(",")]
    except ValueError:
        st.error("โปรดตรวจสอบรูปแบบ Random Seeds (ตัวอย่าง: 42,123,789)")
        st.stop()
        
    # 3.2 คำนวณจำนวนสถานการณ์
    k_list = []
    if "Effectuation" in strategies:
        k_list.append(K_EFFECTUATION)
    if "Causation" in strategies:
        k_list.append(K_CAUSATION)
        
    total_scenarios = len(k_list) * len(C_values) * len(rho_values) * len(thresholds) * len(seeds)
    if total_scenarios > 100:
        st.warning(f"คำเตือน: กำลังรัน {total_scenarios} สถานการณ์ (อาจใช้เวลานาน)")

    # 3.3 เตรียมพารามิเตอร์
    params = []
    for k in k_list:
        for C in C_values:
            for rho in rho_values:
                for thresh in thresholds:
                    for seed in seeds:
                        params.append((k, C, rho, thresh, seed))

    # 3.4 รันการจำลองแบบขนาน
    with st.spinner("กำลังรันการจำลอง..."):
        results = Parallel(n_jobs=-1)(delayed(run_simulation)(*p) for p in stqdm(params))

    # 3.5 สร้าง DataFrame
    df = pd.DataFrame(results)
    summary = df.groupby(['กลยุทธ์', 'C', 'rho', 'เกณฑ์การยอมรับ']).agg({
        'pre_adoption_rate': ['mean', 'std'],
        'post_adoption_rate': ['mean', 'std']
    }).reset_index()
    summary.columns = ['กลยุทธ์', 'C', 'rho', 'เกณฑ์การยอมรับ', 
                      'Pre Mean', 'Pre Std', 'Post Mean', 'Post Std']

    # ===============================
    # 4. แสดงผลลัพธ์
    # ===============================
    st.header("📊 ผลลัพธ์การจำลอง")
    
    # 4.1 ตารางสรุป
    st.subheader("สรุปสถิติ")
    st.dataframe(summary.style.format({
        'Pre Mean': '{:.2f}%',
        'Pre Std': '{:.2f}%',
        'Post Mean': '{:.2f}%',
        'Post Std': '{:.2f}%'
    }))
    
    # 4.2 กราฟเปรียบเทียบ
    st.subheader("กราฟเปรียบเทียบ")
    fig = px.line(
        summary,
        x="rho",
        y="Post Mean",
        color="กลยุทธ์",
        facet_col="C",
        facet_row="เกณฑ์การยอมรับ",
        title="ประสิทธิภาพหลัง Chasm",
        labels={"rho": "สัมประสิทธิ์การเปลี่ยนแปลง", "Post Mean": "อัตราการยอมรับ (%)"},
        category_orders={
            "C": [1, 7, 9],
            "rho": [0.1, 0.7, 0.9],
            "เกณฑ์การยอมรับ": [5, 6, 7]
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4.3 ดาวน์โหลดข้อมูล
    st.subheader("ดาวน์โหลดข้อมูล")
    st.download_button(
        "ดาวน์โหลดข้อมูลเต็ม",
        df.to_csv(index=False),
        "simulation_data.csv"
    )
    st.download_button(
        "ดาวน์โหลดสรุปสถิติ",
        summary.to_csv(index=False),
        "summary.csv"
    )