import numpy as np
import pandas as pd
import streamlit as st  # เพิ่มบรรทัดนี้
from typing import Dict, List, Any, Tuple, Optional, Union
from tqdm import tqdm
from .base_model import NKCModel

class AgentNKCModel(NKCModel):
    """
    แบบจำลองที่มีหลายเอเจนต์ ใช้บิตสตริงและปฏิสัมพันธ์ระหว่างเอเจนต์
    ดัดแปลงจากโค้ดที่สอง
    """
    
    def __init__(self, fitness_params=None):
        """
        กำหนดค่าเริ่มต้นสำหรับแบบจำลองหลายเอเจนต์
        
        Args:
            fitness_params: พารามิเตอร์สำหรับการคำนวณ fitness (ถ้าไม่ระบุจะใช้ค่าเริ่มต้น)
        """
        super().__init__(
            name="AgentNKCModel",
            description="""
            ## แบบจำลองหลายเอเจนต์ (บิตสตริง)
            
            แบบจำลองนี้จำลองการทำงานของเอเจนต์หลายตัวที่มีสถานะเป็นบิตสตริง
            (สตริงของเลข 0 และ 1) และมีปฏิสัมพันธ์ระหว่างกัน
            
            แบบจำลองประกอบด้วยเอเจนต์ 4 ตัวที่แบ่งเป็น 2 กลุ่ม:
            - **Effectuation (K=2)**: เอเจนต์ 0 และ 1 ใช้กลยุทธ์แบบยืดหยุ่น K น้อย
            - **Causation (K=5)**: เอเจนต์ 2 และ 3 ใช้กลยุทธ์แบบมีโครงสร้าง K มาก
            
            ค่า K ในที่นี้กำหนดจำนวนบิตที่มีอิทธิพลต่อแต่ละตำแหน่งในบิตสตริง
            ส่วนค่า C กำหนดระดับการเชื่อมโยงระหว่างเอเจนต์
            
            แบบจำลองนี้ใช้การค้นหาเฉพาะที่ (local search) เพื่อให้เอเจนต์ปรับตัว
            ไปสู่สถานะที่มีความเหมาะสม (fitness) สูงขึ้น
            """
        )
        self.payoff_cache = {}
        self.S = 4  # จำนวนเอเจนต์คงที่ 4 ตัว
        self.effectuation_agents = [0, 1]  # เอเจนต์ที่ใช้กลยุทธ์ Effectuation (K=2)
        self.causation_agents = [2, 3]  # เอเจนต์ที่ใช้กลยุทธ์ Causation (K=5)
        
        # ตั้งค่าพารามิเตอร์สำหรับการคำนวณ fitness
        self.fitness_params = {
            'k_effectuation': 2,
            'k_causation': 5,
            'own_weight': 0.4,
            'cross_weight': -0.2,
            'base_fitness': 0.1,
            'random_range': 0.3,
            'p_ones': 0.1
        }
        
        # อัปเดตพารามิเตอร์จากที่ส่งเข้ามา (ถ้ามี)
        if fitness_params is not None:
            self.fitness_params.update(fitness_params)
    
    def generate_nk_matrix(self, N: int, K: int, seed: Optional[int] = None) -> np.ndarray:
        """
        สร้างเมทริกซ์ NK ที่กำหนดความสัมพันธ์ภายในเอเจนต์
        
        Parameters:
        - N: จำนวนบิตในสตริง
        - K: จำนวนบิตที่มีอิทธิพลต่อแต่ละตำแหน่ง
        - seed: random seed
        
        Returns:
        - เมทริกซ์ NxN ที่กำหนดความสัมพันธ์
        """
        if seed is not None:
            np.random.seed(seed)
        M = np.zeros((N, N), dtype=int)
        for i in range(N):
            M[i, i] = 1  # ตนเองมีอิทธิพลต่อตนเองเสมอ
            for k_offset in range(1, K+1):
                M[i, (i + k_offset) % N] = 1  # บิตต่อเนื่องมีอิทธิพลแบบวงกลม
        return M
    
    def generate_c_matrix_kappa(self, N: int, c: int, kappa: int = 2, seed: Optional[int] = None) -> np.ndarray:
        """
        สร้างเมทริกซ์ C ที่กำหนดความสัมพันธ์ระหว่างเอเจนต์
        
        Parameters:
        - N: จำนวนบิตในสตริง
        - c: จำนวนบิตที่มีอิทธิพลจากเอเจนต์อื่น
        - kappa: ระยะห่างสูงสุดของบิตที่มีอิทธิพล
        - seed: random seed
        
        Returns:
        - เมทริกซ์ NxN ที่กำหนดความสัมพันธ์
        """
        if seed is not None:
            np.random.seed(seed)
        M = np.zeros((N, N), dtype=int)
        max_struct = 1 + 2*kappa
        if c <= max_struct:
            for i in range(N):
                M[i, i] = 1  # ตนเองมีอิทธิพลต่อตนเองเสมอ
                extra_needed = c - 1
                offset = 1
                while extra_needed > 0 and offset <= kappa:
                    if extra_needed > 0:
                        M[i, (i - offset) % N] = 1
                        extra_needed -= 1
                    if extra_needed > 0:
                        M[i, (i + offset) % N] = 1
                        extra_needed -= 1
                    offset += 1
        else:
            # เติมตนเอง + เพื่อนบ้านระยะ kappa => สุ่มเติมส่วนที่เหลือ
            for i in range(N):
                M[i, i] = 1
                for offset in range(1, kappa+1):
                    M[i, (i - offset) % N] = 1
                    M[i, (i + offset) % N] = 1
                have = M[i].sum()
                need = c - have
                if need > 0:
                    zero_positions = np.where(M[i] == 0)[0]
                    if need > len(zero_positions):
                        need = len(zero_positions)
                    chosen = np.random.choice(zero_positions, need, replace=False)
                    M[i, chosen] = 1
        return M
    
    def build_im_matrix_for_4agents(self, c: int) -> List[List[np.ndarray]]:
        """
        สร้างเมทริกซ์ความสัมพันธ์สำหรับเอเจนต์ 4 ตัว
        
        Parameters:
        - c: ระดับการเชื่อมโยงระหว่างเอเจนต์
        
        Returns:
        - เมทริกซ์ความสัมพันธ์ขนาด SxSxNxN
        """
        # Seeds สำหรับความสามารถในการทำซ้ำ
        seed_nk = [101, 202, 303, 404]
        seed_c = [
            [None, 111, 112, 113],
            [121, None, 122, 123],
            [131, 132, None, 133],
            [141, 142, 143, None]
        ]
        
        # ใช้ค่า K ที่กำหนดไว้ในพารามิเตอร์
        K_values = [
            self.fitness_params['k_effectuation'],
            self.fitness_params['k_effectuation'],
            self.fitness_params['k_causation'],
            self.fitness_params['k_causation']
        ]
        N = self.N

        # สร้างเมทริกซ์ NK สำหรับแต่ละเอเจนต์
        nk_matrices = []
        for i in range(self.S):
            nk = self.generate_nk_matrix(N, K_values[i], seed=seed_nk[i])
            nk_matrices.append(nk)

        # สร้างเมทริกซ์ C สำหรับปฏิสัมพันธ์ระหว่างเอเจนต์
        c_matrices = [[None]*self.S for _ in range(self.S)]
        for i in range(self.S):
            for j in range(self.S):
                if i == j:
                    c_matrices[i][j] = None
                else:
                    c_matrices[i][j] = self.generate_c_matrix_kappa(N, c, kappa=2, seed=seed_c[i][j])

        # สร้างเมทริกซ์ IM ที่รวมความสัมพันธ์ทั้งหมด
        IM = []
        for i in range(self.S):
            row_blocks = []
            for j in range(self.S):
                if i == j:
                    row_blocks.append(nk_matrices[i])
                else:
                    row_blocks.append(c_matrices[i][j])
            IM.append(row_blocks)
        return IM
    
    def build_global_payoff_cache(self) -> None:
        """
        เตรียม cache สำหรับเก็บค่า payoff
        """
        self.payoff_cache = {}
    
    def get_payoff(self, agent_idx: int, locus: int, pattern_bits: Tuple[int, ...]) -> float:
        """
        คำนวณ payoff สำหรับบิตแต่ละตำแหน่ง
        
        Parameters:
        - agent_idx: ดัชนีของเอเจนต์
        - locus: ตำแหน่งบิตที่กำลังพิจารณา
        - pattern_bits: รูปแบบบิตที่มีอิทธิพล
        
        Returns:
        - payoff ที่คำนวณได้
        """
        key = (agent_idx, locus, pattern_bits)
        if key in self.payoff_cache:
            return self.payoff_cache[key]

        # แบ่งบิตเป็นส่วนของตนเองและส่วนที่มาจากเอเจนต์อื่น
        total_bits = len(pattern_bits)
        half = total_bits // 2
        own_part = pattern_bits[:half]
        cross_part = pattern_bits[half:]

        own_ratio = sum(own_part) / (len(own_part)+1e-9)
        cross_ratio = sum(cross_part) / (len(cross_part)+1e-9)

        # ใช้พารามิเตอร์จาก fitness_params
        base_fitness = self.fitness_params['base_fitness']
        own_weight = self.fitness_params['own_weight']
        cross_weight = self.fitness_params['cross_weight']
        random_range = self.fitness_params['random_range']

        # สุ่มค่าในช่วง [0..random_range]
        rand_val = random_range * np.random.random()

        # คำนวณ payoff
        payoff = base_fitness + own_weight*own_ratio + cross_weight*cross_ratio + rand_val

        # จำกัด payoff ให้อยู่ในช่วง [0,1]
        payoff = max(0.0, min(1.0, payoff))

        self.payoff_cache[key] = payoff
        return payoff
    
    def compute_agent_fitness(self, agent_idx: int, bitstrings: List[np.ndarray], IM: List[List[np.ndarray]]) -> float:
        """
        คำนวณความเหมาะสม (fitness) ของเอเจนต์
        
        Parameters:
        - agent_idx: ดัชนีของเอเจนต์
        - bitstrings: สถานะปัจจุบันของเอเจนต์ทั้งหมด
        - IM: เมทริกซ์ความสัมพันธ์
        
        Returns:
        - ค่าความเหมาะสมเฉลี่ย
        """
        nk_i = IM[agent_idx][agent_idx]  # เมทริกซ์ NK ของเอเจนต์
        total_contribution = 0.0
        
        for L in range(self.N):
            # บิตของตนเอง
            own_positions = np.where(nk_i[L] == 1)[0]
            pattern_bits = []
            for pos in own_positions:
                pattern_bits.append(bitstrings[agent_idx][pos])

            # บิตจากเอเจนต์อื่น
            cross_positions = []
            for j in range(self.S):
                if j == agent_idx:
                    continue
                c_ij = IM[agent_idx][j]
                if c_ij is None:
                    continue
                cross_pos_j = np.where(c_ij[L] == 1)[0]
                if len(cross_pos_j) > 0:
                    cross_positions.append((j, cross_pos_j))

            # เรียงลำดับเพื่อให้ผลลัพธ์คงที่
            cross_positions.sort(key=lambda x: x[0])
            for (j, pos_array) in cross_positions:
                for posj in pos_array:
                    pattern_bits.append(bitstrings[j][posj])

            pattern_bits = tuple(pattern_bits)
            contribution = self.get_payoff(agent_idx, L, pattern_bits)
            total_contribution += contribution

        return total_contribution / self.N
    
    def run_local_search(self, IM: List[List[np.ndarray]], steps: int = 50, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        รันการค้นหาเฉพาะที่เพื่อปรับปรุงความเหมาะสม
        
        Parameters:
        - IM: เมทริกซ์ความสัมพันธ์
        - steps: จำนวนขั้นตอนการค้นหา
        - seed: random seed
        
        Returns:
        - (effectuation_traj, causation_traj): เส้นทางความเหมาะสมของแต่ละกลุ่ม
        """
        if seed is not None:
            np.random.seed(seed)

        # เริ่มต้นด้วยบิตสตริงแบบสุ่ม (ส่วนใหญ่เป็น 0)
        p_ones = self.fitness_params['p_ones']
        bitstrings = []
        for _ in range(self.S):
            bs = (np.random.rand(self.N) < p_ones).astype(int)
            bitstrings.append(bs)

        effectuation_traj = []
        causation_traj = []

        for _ in range(steps):
            # ปรับปรุงเฉพาะที่ - แต่ละเอเจนต์หาการเปลี่ยนแปลงที่ดีที่สุด
            for agent_idx in range(self.S):
                current_f = self.compute_agent_fitness(agent_idx, bitstrings, IM)
                best_improvement = 0.0
                best_flip = None
                
                # ทดลองพลิกแต่ละบิต
                for b in range(self.N):
                    bitstrings[agent_idx][b] ^= 1  # พลิกบิต
                    new_f = self.compute_agent_fitness(agent_idx, bitstrings, IM)
                    improvement = new_f - current_f
                    bitstrings[agent_idx][b] ^= 1  # คืนค่าเดิม
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_flip = b
                
                # ทำการเปลี่ยนแปลงที่ดีที่สุด (ถ้ามี)
                if best_flip is not None:
                    bitstrings[agent_idx][best_flip] ^= 1

            # วัดค่าเฉลี่ยแยกตามกลุ่ม
            ef_fits = [self.compute_agent_fitness(a, bitstrings, IM) for a in self.effectuation_agents]
            ca_fits = [self.compute_agent_fitness(a, bitstrings, IM) for a in self.causation_agents]
            effectuation_traj.append(np.mean(ef_fits))
            causation_traj.append(np.mean(ca_fits))

        return np.array(effectuation_traj), np.array(causation_traj)
    
    def simulate_fixed_landscape(self, c: int, runs: int = 50, steps: int = 50, show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """ 
        รันการจำลองการค้นหาบนภูมิทัศน์ความเหมาะสมคงที่
        
        Parameters:
        - c: ระดับการเชื่อมโยงระหว่างเอเจนต์
        - runs: จำนวนรอบการจำลอง
        - steps: จำนวนขั้นตอนในแต่ละรอบ
        - show_progress: แสดง progress bar หรือไม่
        
        Returns:
        - (ef_all, ca_all): ผลลัพธ์ของกลุ่ม Effectuation และ Causation
        """
        IM = self.build_im_matrix_for_4agents(c)
        self.build_global_payoff_cache()  # เตรียม cache เดียวสำหรับทุกรอบ

        ef_all = []
        ca_all = []
        
        # แสดง progress bar ด้วย streamlit progress แทน tqdm
        if show_progress:
            progress_text = f"กำลังจำลองสำหรับ C={c}"
            my_bar = st.progress(0, text=progress_text)
            iterator = range(runs)
            for r_idx, r in enumerate(iterator):
                ef_traj, ca_traj = self.run_local_search(IM, steps=steps, seed=1000+r)
                ef_all.append(ef_traj)
                ca_all.append(ca_traj)
                # อัปเดต progress bar
                my_bar.progress((r_idx + 1) / runs, text=f"{progress_text} - {r_idx+1}/{runs}")
        else:
            # ไม่แสดง progress
            for r in range(runs):
                ef_traj, ca_traj = self.run_local_search(IM, steps=steps, seed=1000+r)
                ef_all.append(ef_traj)
                ca_all.append(ca_traj)

        ef_all = np.array(ef_all)  # รูปร่าง (runs, steps)
        ca_all = np.array(ca_all)
        return ef_all, ca_all
    
    def run_simulation(self, 
                       N: int, 
                       c_values: List[int], 
                       steps: int = 50, 
                       runs: int = 50, 
                       show_progress: bool = True,
                       fitness_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        รันการจำลองสำหรับค่า C ที่แตกต่างกัน
        
        Parameters:
        - N: จำนวนบิตในสตริง
        - c_values: รายการของค่า C ที่ต้องการทดสอบ
        - steps: จำนวนขั้นตอนในแต่ละรอบ
        - runs: จำนวนรอบการจำลอง
        - show_progress: แสดง progress bar หรือไม่
        - fitness_params: พารามิเตอร์สำหรับการคำนวณ fitness
        
        Returns:
        - Dict ที่มีผลลัพธ์การจำลอง
        """
        try:
            self.N = N
            
            # อัปเดตพารามิเตอร์ fitness ถ้ามี
            if fitness_params is not None:
                self.fitness_params.update(fitness_params)
                
            # เก็บข้อมูล K ที่ใช้จริงสำหรับรายงานผล
            k_effectuation = self.fitness_params['k_effectuation']
            k_causation = self.fitness_params['k_causation']
            
            scenario_data = {}
            
            for c in c_values:
                ef_all, ca_all = self.simulate_fixed_landscape(c, runs=runs, steps=steps, show_progress=show_progress)
                ef_mean = ef_all.mean(axis=0)
                ca_mean = ca_all.mean(axis=0)
                ef_var = ef_all.var(axis=0)
                ca_var = ca_all.var(axis=0)
                
                scenario_data[c] = {
                    "ef_mean": ef_mean.tolist(),
                    "ca_mean": ca_mean.tolist(),
                    "ef_var": ef_var.tolist(),
                    "ca_var": ca_var.tolist(),
                    "steps": list(range(1, steps + 1))
                }
            
            # แปลงผลลัพธ์ให้เข้ากับรูปแบบของแบบจำลองแบบง่าย
            effectuation_results = []
            causation_results = []
            
            for c in c_values:
                # ผลลัพธ์สำหรับ Effectuation
                for seed in range(runs):
                    effectuation_results.append({
                        'กลยุทธ์': f'Effectuation (K={k_effectuation})',
                        'N': N,
                        'K': k_effectuation,
                        'C': c,
                        'rho': None,  # ไม่มีความหมายในแบบจำลองนี้
                        'เกณฑ์การยอมรับ': None,  # ไม่มีความหมายในแบบจำลองนี้
                        'seed': seed,
                        'fitness_mean': scenario_data[c]["ef_mean"][-1],  # ใช้ค่าสุดท้าย
                        'fitness_var': scenario_data[c]["ef_var"][-1],
                        'averages': scenario_data[c]["ef_mean"]
                    })
                
                # ผลลัพธ์สำหรับ Causation
                for seed in range(runs):
                    causation_results.append({
                        'กลยุทธ์': f'Causation (K={k_causation})',
                        'N': N,
                        'K': k_causation,
                        'C': c,
                        'rho': None,
                        'เกณฑ์การยอมรับ': None,
                        'seed': seed,
                        'fitness_mean': scenario_data[c]["ca_mean"][-1],
                        'fitness_var': scenario_data[c]["ca_var"][-1],
                        'averages': scenario_data[c]["ca_mean"]
                    })
            
            all_results = effectuation_results + causation_results
            return {
                'results': all_results,
                'scenario_data': scenario_data,
                'N': N,
                'steps': steps,
                'runs': runs,
                'c_values': c_values,
                'fitness_params': self.fitness_params
            }
            
        except Exception as e:
            # จัดการข้อผิดพลาด
            error_msg = f"Error in agent simulation with N={N}, c_values={c_values}: {str(e)}"
            print(error_msg)
            return {
                'error': str(e),
                'N': N,
                'steps': steps,
                'runs': runs,
                'c_values': c_values,
                'fitness_params': self.fitness_params
            }
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        ส่งคืนข้อมูลพารามิเตอร์สำหรับการแสดงผลใน UI
        """
        return {
            "N": {
                "type": "number",
                "label": "จำนวนบิตในสตริง (N)",
                "min": 5,
                "max": 20,
                "default": 10,
                "help": "จำนวนบิตในสตริงของแต่ละเอเจนต์ (5-20)"
            },
            "c_values": {
                "type": "multiselect",
                "label": "ระดับการเชื่อมโยงระหว่างเอเจนต์ (C)",
                "options": [1, 3, 5, 7, 9],
                "default": [1, 7, 9],
                "help": "ระดับการเชื่อมโยงระหว่างเอเจนต์ (จำนวนบิตที่มีอิทธิพลข้ามเอเจนต์)"
            },
            "steps": {
                "type": "number",
                "label": "จำนวนขั้นตอนการค้นหา",
                "min": 10,
                "max": 200,
                "default": 50,
                "help": "จำนวนขั้นตอนในแต่ละรอบการค้นหา"
            },
            "runs": {
                "type": "number",
                "label": "จำนวนรอบการจำลอง",
                "min": 1,
                "max": 200,
                "default": 50,
                "help": "จำนวนรอบการจำลองซ้ำ (จำนวนมากให้ผลที่แม่นยำขึ้น)"
            },
            "show_progress": {
                "type": "checkbox",
                "label": "แสดงความคืบหน้าระหว่างการจำลอง",
                "default": True,
                "help": "แสดง progress bar ระหว่างการจำลอง"
            }
        }