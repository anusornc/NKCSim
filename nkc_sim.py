import streamlit as st
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import plotly.express as px

# ===============================
# 1. ตั้งค่าพารามิเตอร์พื้นฐาน
# ===============================
N = 10
K_EFFECTUATION = 2
K_CAUSATION = 5
TIME_STEPS = 100
PRE_CHASM_STEPS = 16

# ===============================
# 2. ฟังก์ชันการจำลอง (ปรับปรุงแล้ว)
# ===============================
def run_simulation(strategy_k, C, rho, threshold, seed):
    np.random.seed(seed)
    
    # 1. เริ่มต้นตัวแปรด้วย Normal Distribution (ค่าเฉลี่ย 5)
    variables = np.random.normal(loc=5, scale=2, size=N)
    variables = np.clip(variables, 1, 9)
    
    # 2. เลือกตัวแปร K ตัวแบบ Fix
    selected_vars = np.random.choice(N, strategy_k, replace=False)
    initial_mean = np.mean(variables[selected_vars])  # ค่าเฉลี่ยเริ่มต้น
    
    pre_adoption = 0
    post_adoption = 0
    
    for step in range(TIME_STEPS):
        if step > 0:
            # 3. อัปเดตตัวแปรด้วย Mean-Reverting Process
            noise = np.random.normal(0, scale=C, size=strategy_k)
            variables[selected_vars] = variables[selected_vars] * rho + \
                                       (1 - rho) * initial_mean + \
                                       noise * (1 - rho)
            variables = np.clip(variables, 1, 9)
        
        # 4. คำนวณค่าเฉลี่ย
        avg = np.mean(variables[selected_vars])
        
        # 5. ตรวจสอบเกณฑ์
        if avg >= threshold:
            if step < PRE_CHASM_STEPS:
                pre_adoption += 1
            else:
                post_adoption += 1
    
    return {
        'กลยุทธ์': 'Effectuation' if strategy_k == K_EFFECTUATION else 'Causation',
        'K': strategy_k,
        'C': C,
        'rho': rho,
        'เกณฑ์การยอมรับ': threshold,
        'seed': seed,
        'อัตราการยอมรับก่อน Chasm (%)': round(pre_adoption / PRE_CHASM_STEPS * 100, 2),
        'อัตราการยอมรับหลัง Chasm (%)': round(post_adoption / (TIME_STEPS - PRE_CHASM_STEPS) * 100, 2)
    }

# ===============================
# 3. สร้างหน้า UI ด้วย Streamlit
# ===============================
st.title("🚀 NKC Model Simulation")

# ===============================
# คำอธิบายการทำงาน
# ===============================
st.markdown("""
### 📝 คำอธิบายการทำงาน
- **วัตถุประสงค์**: จำลองการยอมรับนวัตกรรมระหว่างกลยุทธ์ **Effectuation (ปรับตัวตามสถานการณ์)** และ **Causation (วางแผนตามเป้าหมาย)** 
- **ปัจจัยสำคัญ**:
  - **สภาพแวดล้อม**: ควบคุมด้วยระดับความซับซ้อน (C) และอัตราการเปลี่ยนแปลง (ρ)
  - **เกณฑ์การยอมรับ**: กำหนดค่าเฉลี่ยขั้นต่ำที่ถือว่า "ยอมรับนวัตกรรม"
  - **การสุ่ม**: ใช้ Random Seeds เพื่อให้ผลลัพธ์ซ้ำได้
""")

st.markdown("""
### 🎚️ รายละเอียดพารามิเตอร์
1. **กลยุทธ์**:
   - **Effectuation (K=2)**: ใช้ข้อมูลน้อย ตัดสินใจแบบยืดหยุ่น เหมาะกับสภาพแวดล้อมไม่แน่นอน
   - **Causation (K=5)**: ใช้ข้อมูลมาก ตัดสินใจแบบมีโครงสร้าง เหมาะกับสภาพแวดล้อมเสถียร

2. **ระดับความซับซ้อน (C)**:
   - **1 (ต่ำ)**: สภาพแวดล้อมมีความเสถียร
   - **7 (ปานกลาง)**: สภาพแวดล้อมเปลี่ยนแปลงปานกลาง
   - **9 (สูง)**: สภาพแวดล้อมซับซ้อนและผันผวนสูง

3. **สัมประสิทธิ์การเปลี่ยนแปลง (ρ)**:
   - **0.1**: เปลี่ยนแปลงเร็ว (สภาพแวดล้อมไม่เสถียร)
   - **0.7**: เปลี่ยนแปลงปานกลาง
   - **0.9**: เปลี่ยนแปลงช้า (สภาพแวดล้อมเสถียร)

4. **เกณฑ์การยอมรับ**:
   - **5 (ต่ำ)**: ยอมรับนวัตกรรมเมื่อค่าเฉลี่ย ≥5
   - **6 (ปานกลาง)**: ยอมรับเมื่อค่าเฉลี่ย ≥6
   - **7 (สูง)**: ยอมรับเมื่อค่าเฉลี่ย ≥7

5. **Random Seeds**:  
   ค่าเริ่มต้นสำหรับการสุ่ม (เช่น 42, 123, 789) เพื่อให้ผลลัพธ์ซ้ำได้
""")


st.sidebar.header("ส่วนพารามิเตอร์")

# 3.1 ตัวเลือกกลยุทธ์
st.sidebar.subheader("กลยุทธ์")
strategies = st.sidebar.multiselect(
    "เลือกกลยุทธ์",
    ["Effectuation", "Causation"],
    default=["Effectuation", "Causation"]
)
st.sidebar.caption("""
- **Effectuation (ปรับตัวตามสถานการณ์)**: ใช้ข้อมูลน้อย (K=2) ตัดสินใจแบบยืดหยุ่น
- **Causation (วางแผนตามเป้าหมาย)**: ใช้ข้อมูลมาก (K=5) ตัดสินใจแบบมีโครงสร้าง
""")

# 3.2 ตัวเลือกพารามิเตอร์
st.sidebar.subheader("พารามิเตอร์หลัก")
C_values = st.sidebar.multiselect(
    "ระดับความซับซ้อน (C)",
    [1,7,9],
    default=[1,7,9]
)
st.sidebar.caption("1=ต่ำ, 7=ปานกลาง, 9=สูง (สภาพแวดล้อมภายนอก)")

rho_values = st.sidebar.multiselect(
    "สัมประสิทธิ์การเปลี่ยนแปลง (rho)",
    [0.1,0.7,0.9],
    default=[0.1,0.7,0.9]
)
st.sidebar.caption("0.1=เปลี่ยนแปลงเร็ว, 0.9=เปลี่ยนแปลงช้า")

thresholds = st.sidebar.multiselect(
    "เกณฑ์การยอมรับ",
    [5,6,7],
    default=[5,6,7]
)
st.sidebar.caption("ค่าเฉลี่ยของตัวแปรที่ต้องการให้ถือว่า 'ยอมรับ'")

seeds = st.sidebar.text_input(
    "Random Seeds (คั่นด้วยเครื่องหมายจุลภาค)",
    "42,123,789"
)
st.sidebar.caption("ค่าเริ่มต้นสำหรับการสุ่ม (เพื่อให้ผลลัพธ์ซ้ำได้)")

# 3.3 ตัวเลือกขั้นสูง
with st.sidebar.expander("ตัวเลือกขั้นสูง"):
    st.caption("ปรับค่าเหล่านี้เฉพาะเมื่อจำเป็น")
    TIME_STEPS = st.number_input(
        "จำนวนขั้นตอนการจำลอง",
        100
    )
    st.caption("จำนวนรอบการอัปเดตสภาพแวดล้อม (เริ่มต้น=100)")
    
    PRE_CHASM_STEPS = st.number_input(
        "ขั้นตอนก่อน Chasm",
        16
    )
    st.caption("ขั้นตอนก่อนการยอมรับนวัตกรรมแบบก้าวกระโดด (เริ่มต้น=16)")

# 3.4 ปุ่มเริ่มการจำลอง
st.sidebar.markdown("---")
if st.sidebar.button("เริ่มการจำลอง"):
    # ... (โค้ดเดิม)
    # แปลงข้อมูลจาก UI
    seeds = [int(s) for s in seeds.split(",")]
    k_list = []
    if "Effectuation" in strategies:
        k_list.append(K_EFFECTUATION)
    if "Causation" in strategies:
        k_list.append(K_CAUSATION)
    
    # สร้างพารามิเตอร์ทั้งหมด
    params = []
    for k in k_list:
        for C in C_values:
            for rho in rho_values:
                for thresh in thresholds:
                    for seed in seeds:
                        params.append((k, C, rho, thresh, seed))
    
    # รันการจำลองแบบขนาน
    with st.spinner("กำลังรันการจำลอง..."):
        results = Parallel(n_jobs=-1)(delayed(run_simulation)(*p) for p in params)
    
    # สร้าง DataFrame
    df = pd.DataFrame(results)
    summary = df.groupby(['กลยุทธ์', 'C', 'rho', 'เกณฑ์การยอมรับ']).agg({
        'อัตราการยอมรับก่อน Chasm (%)': 'mean',
        'อัตราการยอมรับหลัง Chasm (%)': 'mean'
    }).reset_index()
    
    # แสดงผลลัพธ์
    st.header("📊 ผลลัพธ์การจำลอง")
    st.dataframe(summary)
    
    # สร้างกราฟด้วย Plotly
    fig = px.line(
        summary,
        x="rho",
        y="อัตราการยอมรับหลัง Chasm (%)",
        color="กลยุทธ์",
        facet_col="C",
        facet_row="เกณฑ์การยอมรับ",
        title="เปรียบเทียบกลยุทธ์ภายใต้สภาพแวดล้อมต่างๆ",
        labels={"rho": "สัมประสิทธิ์การเปลี่ยนแปลง (rho)"},
        hover_data=["C", "เกณฑ์การยอมรับ"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # อนุญาตให้ดาวน์โหลดข้อมูล
    st.download_button(
        label="ดาวน์โหลดข้อมูลเป็น CSV",
        data=df.to_csv(index=False),
        file_name="simulation_results.csv",
        mime="text/csv"
    )