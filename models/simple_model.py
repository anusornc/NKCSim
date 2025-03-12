import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from .base_model import NKCModel

class SimpleNKCModel(NKCModel):
    """
    แบบจำลองแบบง่ายจากโค้ดแรก ใช้ตัวแปรแบบต่อเนื่อง
    และแนวโน้มแบบลอการิทึมหรือ S-curve
    """
    
    def __init__(self):
        """
        กำหนดค่าเริ่มต้นสำหรับแบบจำลองแบบง่าย
        """
        super().__init__(
            name="SimpleNKCModel",
            description="""
            ## แบบจำลองแบบง่าย (ตัวแปรต่อเนื่อง)
            
            แบบจำลองนี้ใช้ตัวแปรแบบต่อเนื่องที่มีค่าอยู่ระหว่าง 1-12 และมีการเปลี่ยนแปลงตามเวลา
            โดยมีปัจจัยที่มีผลต่อการเปลี่ยนแปลงดังนี้:
            
            1. **แนวโน้มตามเวลา** - ใช้แนวโน้มแบบลอการิทึมและ S-curve เพื่อจำลองการแพร่กระจายนวัตกรรม
            2. **ความผันผวน (C)** - ค่า C สูงหมายถึงความผันผวนสูง ทำให้ผลลัพธ์ไม่แน่นอน
            3. **ความเฉื่อย (rho)** - ค่า rho สูงหมายถึงการเปลี่ยนแปลงช้า มีความเฉื่อยสูง
            4. **ปฏิสัมพันธ์ระหว่างตัวแปร** - ตัวแปรมีผลกระทบซึ่งกันและกัน
            
            แบบจำลองนี้เปรียบเทียบระหว่างกลยุทธ์ Effectuation (ยืดหยุ่น, K น้อย) และ 
            Causation (มีโครงสร้าง, K มาก) เพื่อดูว่ากลยุทธ์ใดทำให้เกิดการยอมรับนวัตกรรมได้ดีกว่า
            """
        )
        self.MAX_VARIABLES = 30
    
    def run_simulation(self, 
                       N: int, 
                       strategy_k: int, 
                       C: float, 
                       rho: float, 
                       threshold: float, 
                       seed: int, 
                       time_steps: int, 
                       pre_chasm_steps: int,
                       interaction_strength: float = 0.2) -> Dict[str, Any]:
        """
        จำลองการยอมรับนวัตกรรมด้วยแบบจำลอง NKC แบบง่าย
        
        Parameters:
        - N: จำนวนตัวแปรทั้งหมด
        - strategy_k: จำนวนตัวแปรที่เลือกใช้
        - C: ระดับความซับซ้อน (ความผันผวน)
        - rho: สัมประสิทธิ์การเปลี่ยนแปลง
        - threshold: เกณฑ์การยอมรับ
        - seed: random seed
        - time_steps: จำนวนขั้นตอนการจำลอง
        - pre_chasm_steps: จุดแบ่งก่อน/หลัง Chasm
        - interaction_strength: ความแรงของปฏิสัมพันธ์ระหว่างตัวแปร

        Returns:
        - Dict ที่มีผลลัพธ์การจำลอง
        """
        try:
            np.random.seed(seed)
            
            # สร้างตัวแปรเริ่มต้น
            variables = np.random.normal(loc=5, scale=2, size=N)
            variables = np.clip(variables, 1, 12)
            
            # สร้าง interaction matrix
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
            
            # Initialize c_values
            c_values = []
            
            # วนลูปตามจำนวน time steps
            for step in range(time_steps):
                if step > 0:
                    # คำนวณปฏิสัมพันธ์ระหว่างตัวแปร
                    interaction_effects = np.zeros(N)
                    for i in range(N):
                        interaction_effects[i] = np.sum(interaction_matrix[i, :] * variables) / N
                    
                    # สร้าง noise
                    noise = np.random.normal(0, scale=C, size=strategy_k)
                    
                    # ปรับ trend_factor เป็นฟังก์ชันลอการิทึมและ S-curve
                    s_curve_factor = 2.0 / (1.0 + np.exp(-0.1 * (step - time_steps/2))) - 1.0
                    trend_factor = 0.5 * np.log1p(step) + s_curve_factor * 0.3
                    
                    # อัปเดตตัวแปร
                    variables[selected_vars] = (
                        variables[selected_vars] * rho + 
                        (1 - rho) * initial_mean + 
                        noise * (1 - rho) + 
                        trend_factor + 
                        interaction_effects[selected_vars] * (1 - rho)
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
            print(error_msg)
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
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        ส่งคืนข้อมูลพารามิเตอร์สำหรับการแสดงผลใน UI
        """
        return {
            "N": {
                "type": "number",
                "label": "จำนวนตัวแปรทั้งหมด (N)",
                "min": 5,
                "max": self.MAX_VARIABLES,
                "default": 10,
                "help": "จำนวนตัวแปรในระบบ (5-30)"
            },
            "K_EFFECTUATION": {
                "type": "number",
                "label": "K สำหรับ Effectuation",
                "min": 1,
                "max": self.MAX_VARIABLES - 1,
                "default": 2,
                "help": "จำนวนตัวแปรที่ใช้ในกลยุทธ์ยืดหยุ่น"
            },
            "K_CAUSATION": {
                "type": "number",
                "label": "K สำหรับ Causation",
                "min": 1,
                "max": self.MAX_VARIABLES - 1,
                "default": 5,
                "help": "จำนวนตัวแปรที่ใช้ในกลยุทธ์มีโครงสร้าง"
            },
            "strategies": {
                "type": "multiselect",
                "label": "เลือกกลยุทธ์",
                "options": ["Effectuation", "Causation"],
                "default": ["Effectuation", "Causation"],
                "help": "Effectuation: ใช้ข้อมูลน้อย | Causation: ใช้ข้อมูลมาก"
            },
            "C_values": {
                "type": "multiselect",
                "label": "ระดับความซับซ้อน (C)",
                "options": [0.5, 1.0, 1.5, 2.0, 3.0],
                "default": [1.0, 3.0],
                "help": "C ต่ำ = เสถียร, C สูง = ผันผวน (หน่วย: ค่าเบี่ยงเบนมาตรฐาน)"
            },
            "rho_values": {
                "type": "multiselect",
                "label": "สัมประสิทธิ์การเปลี่ยนแปลง (rho)",
                "options": [0.1, 0.3, 0.5, 0.7, 0.9],
                "default": [0.1, 0.9],
                "help": "0.1 = เปลี่ยนเร็ว, 0.9 = เปลี่ยนช้า (หน่วย: สัดส่วน)"
            },
            "thresholds": {
                "type": "multiselect",
                "label": "เกณฑ์การยอมรับ",
                "options": [5.0, 6.0, 7.0, 8.0],
                "default": [5.0, 7.0],
                "help": "ค่าเฉลี่ยขั้นต่ำที่ถือว่ายอมรับนวัตกรรม (หน่วย: คะแนน)"
            },
            "seeds_input": {
                "type": "text",
                "label": "Random Seeds (คั่นด้วยเครื่องหมายจุลภาค)",
                "default": "42, 123",
                "help": "ตัวเลขสำหรับการสุ่ม (เช่น 42, 123)"
            },
            "time_steps": {
                "type": "number",
                "label": "จำนวนขั้นตอนการจำลอง",
                "min": 20,
                "max": 500,
                "default": 100,
                "help": "จำนวนรอบการจำลอง (หน่วย: รอบ)"
            },
            "pre_chasm_steps": {
                "type": "number",
                "label": "ขั้นตอนก่อน Chasm",
                "min": 5,
                "max": 495,  # time_steps - 5
                "default": 16,
                "help": "จุดแบ่งก่อน/หลัง Chasm (หน่วย: รอบ)"
            },
            "max_simulations": {
                "type": "number",
                "label": "จำนวนการจำลองสูงสุด",
                "min": 1,
                "max": 200,
                "default": 50,
                "help": "จำกัดการรันเพื่อประสิทธิภาพ"
            },
            "interaction_strength": {
                "type": "slider",
                "label": "ความแรงของปฏิสัมพันธ์ระหว่างตัวแปร",
                "min": 0.0,
                "max": 0.5,
                "default": 0.2,
                "step": 0.05,
                "help": "ระดับปฏิสัมพันธ์ระหว่างตัวแปรต่างๆ (0 = ไม่มี, 0.5 = สูง)"
            }
        }