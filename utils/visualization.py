import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple, Optional, Union

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
    if 'กลยุทธ์' in df.columns and 'C' in df.columns:
        grouped = df.groupby(['กลยุทธ์', 'C'])
    elif 'กลยุทธ์' in df.columns:
        grouped = df.groupby(['กลยุทธ์'])
    elif 'C' in df.columns:
        grouped = df.groupby(['C'])
    else:
        # ถ้าไม่มีคอลัมน์สำหรับจัดกลุ่ม ให้สุ่มตัวอย่าง
        return df.sample(min(max_points, len(df)))
    
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

def plot_simple_model_results(df: pd.DataFrame, summary: pd.DataFrame, graph_params: Dict[str, Any], K_EFFECTUATION: int, K_CAUSATION: int) -> None:
    """
    สร้างกราฟสำหรับผลลัพธ์ของแบบจำลองแบบง่าย
    
    Args:
        df: DataFrame ที่มีผลลัพธ์การจำลอง
        summary: DataFrame ที่มีสรุปผลลัพธ์
        graph_params: พารามิเตอร์สำหรับกราฟ
        K_EFFECTUATION: The K value for Effectuation.
        K_CAUSATION: The K value for Causation.
    """
    if 'graph_type' not in graph_params:
        st.error("ไม่พบประเภทกราฟในพารามิเตอร์")
        return
    
    graph_type = graph_params['graph_type']
    
    if graph_type == "2D แยกตามเกณฑ์" or graph_type == "แสดงทั้งหมด":
        plot_2d_by_threshold(df, graph_params, K_EFFECTUATION, K_CAUSATION)
    
    if graph_type == "3D รวม Time Steps" or graph_type == "แสดงทั้งหมด":
        plot_3d(df, graph_params)
    
    if graph_type == "2D Time Series" or graph_type == "แสดงทั้งหมด":
        plot_time_series(df, graph_params)

def plot_2d_by_threshold(df: pd.DataFrame, graph_params: Dict[str, Any], K_EFFECTUATION: int, K_CAUSATION: int) -> None:
    """
    สร้างกราฟ 2D แยกตามเกณฑ์
    
    Args:
        df: DataFrame ที่มีผลลัพธ์การจำลอง
        graph_params: พารามิเตอร์สำหรับกราฟ
        K_EFFECTUATION: The K value for Effectuation.
        K_CAUSATION: The K value for Causation.
    """
    thresholds = df['เกณฑ์การยอมรับ'].unique()
    
    for i, thresh in enumerate(thresholds):
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
                title=f"{graph_params['y_axis']} (เกณฑ์ = {thresh}, K_EFFECTUATION={K_EFFECTUATION}, K_CAUSATION={K_CAUSATION})",
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
            st.plotly_chart(fig, use_container_width=True, key=f"scatter_threshold_{i}")
            
            # แสดงกราฟเวลาที่ข้ามเกณฑ์
            if 'step_ข้าม_threshold' in filtered_df.columns:
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
                st.plotly_chart(fig_cross, use_container_width=True, key=f"cross_threshold_{i}")

def plot_3d(results: Union[List[Dict[str, Any]], pd.DataFrame], graph_params: Dict[str, Any]) -> None:
    """
    สร้างกราฟ 3D
    
    Args:
        results: รายการของผลลัพธ์การจำลองหรือ DataFrame
        graph_params: พารามิเตอร์สำหรับกราฟ
    """
    st.subheader("กราฟ 3D")
    
    # เตรียมข้อมูลสำหรับกราฟ 3D
    time_data = []
    
    # ตรวจสอบว่า results เป็น DataFrame หรือรายการของ dictionary
    if isinstance(results, pd.DataFrame):
        # กรณีที่ results เป็น DataFrame
        for _, result in results.iterrows():
            # ข้ามรายการที่มีข้อผิดพลาด
            if 'error' in result and pd.notna(result['error']):
                continue
            
            # แปลง averages จาก string เป็น list ถ้าจำเป็น
            averages = result['averages']
            if isinstance(averages, str):
                try:
                    import ast
                    averages = ast.literal_eval(averages)
                except (ValueError, SyntaxError):
                    continue
            
            for t, avg in enumerate(averages):
                if graph_params['time_step_range'][0] <= t <= graph_params['time_step_range'][1]:
                    entry = {
                        'กลยุทธ์': result['กลยุทธ์'],
                        'Time Step': t,
                        'ค่าเฉลี่ย': avg
                    }
                    
                    # เพิ่มข้อมูลเฉพาะที่มีในผลลัพธ์
                    for key in ['C', 'rho', 'เกณฑ์การยอมรับ']:
                        if key in result and pd.notna(result[key]):
                            entry[key] = result[key]
                    
                    time_data.append(entry)
    else:
        # กรณีที่ results เป็นรายการของ dictionary
        for result in results:
            if isinstance(result, dict) and 'error' in result:
                continue
            
            # ข้อมูลอาจเป็น dict หรือเป็นแถวของ DataFrame
            try:
                if isinstance(result, dict):
                    strategy = result.get('กลยุทธ์')
                    averages_data = result.get('averages')
                else:
                    strategy = result['กลยุทธ์']
                    averages_data = result['averages']
                
                # แปลง averages จาก string เป็น list ถ้าจำเป็น
                if isinstance(averages_data, str):
                    import ast
                    averages_data = ast.literal_eval(averages_data)
                
                for t, avg in enumerate(averages_data):
                    if graph_params['time_step_range'][0] <= t <= graph_params['time_step_range'][1]:
                        entry = {
                            'กลยุทธ์': strategy,
                            'Time Step': t,
                            'ค่าเฉลี่ย': avg
                        }
                        
                        # เพิ่มข้อมูลเฉพาะที่มีในผลลัพธ์
                        for key in ['C', 'rho', 'เกณฑ์การยอมรับ']:
                            if isinstance(result, dict) and key in result:
                                entry[key] = result[key]
                            elif hasattr(result, key) and getattr(result, key) is not None:
                                entry[key] = getattr(result, key)
                        
                        time_data.append(entry)
            except (KeyError, TypeError, ValueError) as e:
                # ข้ามข้อมูลที่ไม่ถูกต้อง
                continue
    
    time_df = pd.DataFrame(time_data)
    
    if time_df.empty:
        st.error("ไม่มีข้อมูลสำหรับกราฟ 3D")
        return
    
    # เลือกตัวแทนข้อมูลแทนการสุ่ม
    sample_df = sample_representative_data(time_df, max_points=2000)
    
    color_col = graph_params.get('color_by', 'กลยุทธ์')
    if color_col not in sample_df.columns:
        color_col = 'กลยุทธ์'  # ใช้กลยุทธ์เป็นค่าเริ่มต้น
    
    # กำหนดแกน x
    if 'rho' in sample_df.columns:
        x_col = 'rho'
        x_title = "สัมประสิทธิ์การเปลี่ยนแปลง (rho)"
    elif 'C' in sample_df.columns:
        x_col = 'C'
        x_title = "ระดับความซับซ้อน (C)"
    else:
        x_col = 'Time Step'
        x_title = "Time Step"
        # ถ้าใช้ Time Step เป็นแกน x ให้ใช้กลยุทธ์เป็นสี
        color_col = 'กลยุทธ์'
    
    # กำหนดสีที่เหมาะสม
    if color_col == 'กลยุทธ์':
        colorscale = 'Bluered'  # สีแยกชัดเจนระหว่างกลยุทธ์
        colorbar_title = 'กลยุทธ์'
        marker_colormap = sample_df[color_col].astype('category').cat.codes
    elif color_col == 'C':
        colorscale = 'Viridis'  # ไล่ระดับตาม C
        colorbar_title = 'ระดับความซับซ้อน (C)'
        marker_colormap = sample_df[color_col]
    else:  # rho หรืออื่นๆ
        colorscale = 'Plasma'  # ไล่ระดับตาม rho
        colorbar_title = 'สัมประสิทธิ์การเปลี่ยนแปลง (rho)'
        marker_colormap = sample_df[color_col]
    
    # สร้างข้อความสำหรับ hover
    hover_text = []
    for _, row in sample_df.iterrows():
        text = f"กลยุทธ์: {row['กลยุทธ์']}<br>Time Step: {row['Time Step']}<br>ค่าเฉลี่ย: {row['ค่าเฉลี่ย']:.2f}"
        for key in ['C', 'rho', 'เกณฑ์การยอมรับ']:
            if key in row and row[key] is not None:
                text += f"<br>{key}: {row[key]}"
        hover_text.append(text)
    
    # สร้างกราฟ 3D
    fig = go.Figure(data=[go.Scatter3d(
        x=sample_df[x_col],
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
        text=hover_text,
        hoverinfo='text'
    )])
    
    # ปรับแต่งกราฟ
    fig.update_layout(
        title=f"กราฟ 3D: {x_title} vs Time Step vs ค่าเฉลี่ย (สีตาม {color_col})",
        scene=dict(
            xaxis_title=x_title,
            yaxis_title="Time Step",
            zaxis_title="ค่าเฉลี่ย"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True, key="scatter3d_plot")

def plot_time_series(results, graph_params: Dict[str, Any]) -> None:
    """
    สร้างกราฟ Time Series
    
    Args:
        results: รายการของผลลัพธ์การจำลองหรือ DataFrame
        graph_params: พารามิเตอร์สำหรับกราฟ
    """
    st.subheader("กราฟ Time Series")
    
    # เตรียมข้อมูลสำหรับกราฟ time series
    time_data = []
    
    # ตรวจสอบว่า results เป็น DataFrame หรือรายการของ dictionary
    if isinstance(results, pd.DataFrame):
        # กรณีที่ results เป็น DataFrame
        if 'averages' not in results.columns:
            st.error("ไม่พบข้อมูล 'averages' ในผลลัพธ์")
            return
            
        for _, result in results.iterrows():
            # ข้ามรายการที่มีข้อผิดพลาด
            if 'error' in result and pd.notna(result['error']):
                continue
                
            # แปลง averages จาก string เป็น list ถ้าจำเป็น
            averages = result['averages']
            if isinstance(averages, str):
                # พยายามแปลง string เป็น list
                try:
                    import ast
                    averages = ast.literal_eval(averages)
                except (ValueError, SyntaxError):
                    # ถ้าแปลงไม่ได้ ข้ามไป
                    continue
            
            # สร้างรหัสเฉพาะสำหรับแต่ละชุดข้อมูล
            result_cols = ['กลยุทธ์', 'seed', 'C', 'rho', 'เกณฑ์การยอมรับ']
            id_parts = []
            for col in result_cols:
                if col in result and pd.notna(result[col]):
                    id_parts.append(f"{col}_{result[col]}")
            
            result_id = "_".join(id_parts) if id_parts else f"row_{_}"
            
            # สร้างข้อมูลสำหรับแต่ละ time step
            time_step_range = graph_params.get('time_step_range', [0, len(averages)])
            for t, avg in enumerate(averages):
                if time_step_range[0] <= t <= time_step_range[1]:
                    entry = {
                        'กลยุทธ์': result['กลยุทธ์'],
                        'Time Step': t,
                        'ค่าเฉลี่ย': avg,
                        'id': result_id
                    }
                    
                    # เพิ่มข้อมูลเฉพาะที่มีในผลลัพธ์
                    threshold_field = 'เกณฑ์การยอมรับ'
                    for key in ['C', 'rho', threshold_field]:
                        if key in result and pd.notna(result[key]):
                            entry[key] = result[key]
                    
                    # เพิ่มฟิลด์ว่าข้ามเกณฑ์หรือไม่ (ถ้ามีข้อมูล)
                    if threshold_field in entry:
                        entry['ข้ามเกณฑ์'] = avg >= entry[threshold_field]
                    
                    time_data.append(entry)
    else:
        # กรณีที่ results เป็นรายการของ dictionary
        for result in results:
            if 'error' in result:
                continue
            
            # สร้างรหัสเฉพาะสำหรับแต่ละชุดข้อมูล
            id_parts = [result['กลยุทธ์'], str(result.get('seed', 0))]
            for key in ['C', 'rho', 'เกณฑ์การยอมรับ']:
                if key in result and result[key] is not None:
                    id_parts.append(f"{key}_{result[key]}")
            
            result_id = "_".join(id_parts)
            
            for t, avg in enumerate(result['averages']):
                if graph_params.get('time_step_range', [0, len(result['averages'])])[0] <= t <= graph_params.get('time_step_range', [0, len(result['averages'])])[1]:
                    entry = {
                        'กลยุทธ์': result['กลยุทธ์'],
                        'Time Step': t,
                        'ค่าเฉลี่ย': avg,
                        'id': result_id
                    }
                    
                    # เพิ่มข้อมูลเฉพาะที่มีในผลลัพธ์
                    threshold_field = 'เกณฑ์การยอมรับ'
                    for key in ['C', 'rho', threshold_field]:
                        if key in result and result[key] is not None:
                            entry[key] = result[key]
                    
                    # เพิ่มฟิลด์ว่าข้ามเกณฑ์หรือไม่ (ถ้ามีข้อมูล)
                    if threshold_field in entry:
                        entry['ข้ามเกณฑ์'] = avg >= entry[threshold_field]
                    
                    time_data.append(entry)
    
    time_df = pd.DataFrame(time_data)
    
    if time_df.empty:
        st.error("ไม่มีข้อมูลสำหรับกราฟ Time Series")
        return
    
    # แบ่งตามเกณฑ์การยอมรับ (ถ้ามี)
    thresholds = time_df['เกณฑ์การยอมรับ'].dropna().unique() if 'เกณฑ์การยอมรับ' in time_df.columns else [None]
    
    for i, thresh in enumerate(thresholds):
        expander_title = f"Time Series สำหรับเกณฑ์การยอมรับ = {thresh}" if thresh is not None else "Time Series"
        with st.expander(expander_title, expanded=True):
            # กรองข้อมูลตามเกณฑ์ (ถ้ามี)
            if thresh is not None:
                thresh_df = time_df[time_df['เกณฑ์การยอมรับ'] == thresh]
            else:
                thresh_df = time_df
            
            color_col = graph_params.get('color_by', 'กลยุทธ์')
            if color_col not in thresh_df.columns:
                color_col = 'กลยุทธ์'  # ใช้กลยุทธ์เป็นค่าเริ่มต้น
            
            # สร้างคอลัมน์รวมสำหรับการจัดกลุ่ม
            if color_col == 'กลยุทธ์':
                thresh_df['group'] = thresh_df['กลยุทธ์']
                facet_col = 'C' if 'C' in thresh_df.columns else None
                facet_row = 'rho' if 'rho' in thresh_df.columns else None
            elif color_col == 'C' and 'C' in thresh_df.columns:
                thresh_df['group'] = thresh_df['C'].astype(str)
                facet_col = 'กลยุทธ์'
                facet_row = 'rho' if 'rho' in thresh_df.columns else None
            elif color_col == 'rho' and 'rho' in thresh_df.columns:
                thresh_df['group'] = thresh_df['rho'].astype(str)
                facet_col = 'กลยุทธ์'
                facet_row = 'C' if 'C' in thresh_df.columns else None
            else:
                thresh_df['group'] = 'ทั้งหมด'
                facet_col = 'กลยุทธ์'
                facet_row = None
            
            # สร้างกราฟเส้น
            fig = px.line(
                thresh_df,
                x="Time Step",
                y="ค่าเฉลี่ย",
                color="group",
                facet_col=facet_col,
                facet_row=facet_row,
                title=f"กราฟ Time Series: ค่าเฉลี่ยในแต่ละ Time Step" + (f" (เกณฑ์ = {thresh})" if thresh is not None else ""),
                labels={
                    'Time Step': 'Time Step',
                    'ค่าเฉลี่ย': 'ค่าเฉลี่ย',
                    'group': color_col
                },
                line_group='id',  # แยกเส้นตาม simulation
                height=600
            )
            
            # เพิ่มเส้นเกณฑ์การยอมรับ (ถ้ามี)
            if thresh is not None:
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
            
            st.plotly_chart(fig, use_container_width=True, key=f"timeseries_{i}")

def plot_agent_model_results(results: Dict[str, Any], unique_id: Optional[str] = None) -> None:
    """
    สร้างกราฟสำหรับผลลัพธ์ของแบบจำลองหลายเอเจนต์
    
    Args:
        results: ผลลัพธ์จากการจำลอง
        unique_id: ID ที่ไม่ซ้ำกันสำหรับการแสดงผลหลายรอบ
    """
    # ใช้ unique_id หรือ id ของ results เพื่อให้ key ไม่ซ้ำกัน
    key_prefix = f"{unique_id}_" if unique_id else f"agent_{id(results)}_"
    
    if 'error' in results:
        st.error(f"เกิดข้อผิดพลาดในการจำลอง: {results['error']}")
        return
    
    scenario_data = results.get('scenario_data', {})
    if not scenario_data:
        st.warning("ไม่มีข้อมูลผลลัพธ์สำหรับการแสดงกราฟ")
        return
    
    c_values = results.get('c_values', [])
    steps = results.get('steps', 50)
    k_effectuation = results.get('k_effectuation', 2)
    k_causation = results.get('k_causation', 5)
    
    # สร้างกราฟค่าเฉลี่ยของประสิทธิภาพ (Performance Mean)
    st.subheader("กราฟค่าเฉลี่ยของประสิทธิภาพ")
    
    fig_mean = go.Figure()
    for c in c_values:
        if c in scenario_data:
            fig_mean.add_trace(go.Scatter(
                x=list(range(1, steps + 1)),
                y=scenario_data[c]["ef_mean"],
                name=f"Effectuation, K={k_effectuation}, C={c}",
                line=dict(width=2)
            ))
            fig_mean.add_trace(go.Scatter(
                x=list(range(1, steps + 1)),
                y=scenario_data[c]["ca_mean"],
                name=f"Causation, K={k_causation}, C={c}",
                line=dict(dash='dash', width=2)
            ))
    
    fig_mean.update_layout(
        title="ประสิทธิภาพเฉลี่ย vs เวลา",
        xaxis_title="รอบเวลา",
        yaxis_title="ประสิทธิภาพเฉลี่ย",
        legend_title="กลยุทธ์",
        height=500,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_mean, use_container_width=True, key=f"{key_prefix}mean_chart")
    
    # สร้างกราฟความแปรปรวนของประสิทธิภาพ (Performance Variance)
    st.subheader("กราฟความแปรปรวนของประสิทธิภาพ")
    
    fig_var = go.Figure()
    for c in c_values:
        if c in scenario_data:
            fig_var.add_trace(go.Scatter(
                x=list(range(1, steps + 1)),
                y=scenario_data[c]["ef_var"],
                name=f"Effectuation, K={k_effectuation}, C={c}",
                line=dict(width=2)
            ))
            fig_var.add_trace(go.Scatter(
                x=list(range(1, steps + 1)),
                y=scenario_data[c]["ca_var"],
                name=f"Causation, K={k_causation}, C={c}",
                line=dict(dash='dash', width=2)
            ))
    
    fig_var.update_layout(
        title="ความแปรปรวนของประสิทธิภาพ vs เวลา",
        xaxis_title="รอบเวลา",
        yaxis_title="ความแปรปรวนของประสิทธิภาพ",
        legend_title="กลยุทธ์",
        height=500,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_var, use_container_width=True, key=f"{key_prefix}var_chart")
    
    # สร้างกราฟเปรียบเทียบค่า C
    st.subheader("กราฟเปรียบเทียบค่า C")
    
    # ตรวจสอบว่ามีค่า C มากกว่าหนึ่งค่าหรือไม่
    if len(c_values) > 1:
        # เตรียมข้อมูลสำหรับกราฟแท่ง
        bar_data = []
        for c in c_values:
            if c in scenario_data:
                # ใช้ค่าสุดท้ายของ mean
                bar_data.append({
                    'C': c,
                    'กลยุทธ์': f'Effectuation (K={k_effectuation})',
                    'ค่าเฉลี่ยสุดท้าย': scenario_data[c]["ef_mean"][-1]
                })
                bar_data.append({
                    'C': c,
                    'กลยุทธ์': f'Causation (K={k_causation})',
                    'ค่าเฉลี่ยสุดท้าย': scenario_data[c]["ca_mean"][-1]
                })
        
        df_bar = pd.DataFrame(bar_data)
        
        fig_bar = px.bar(
            df_bar, 
            x='C', 
            y='ค่าเฉลี่ยสุดท้าย', 
            color='กลยุทธ์',
            barmode='group',
            title="เปรียบเทียบประสิทธิภาพสุดท้ายตามค่า C",
            labels={
                'C': 'ระดับการเชื่อมโยงระหว่างเอเจนต์ (C)',
                'ค่าเฉลี่ยสุดท้าย': 'ประสิทธิภาพเฉลี่ยสุดท้าย'
            },
            height=400
        )
        
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}bar_chart")
    else:
        st.info("ต้องมีค่า C มากกว่าหนึ่งค่าเพื่อเปรียบเทียบ")