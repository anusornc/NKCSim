from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union

class NKCModel(ABC):
    """
    คลาสพื้นฐานสำหรับแบบจำลอง NKC
    เป็น Abstract Base Class ที่กำหนดอินเตอร์เฟซสำหรับแบบจำลองทั้งหมด
    """
    
    def __init__(self, name: str, description: str):
        """
        Parameters:
        - name: ชื่อของแบบจำลอง
        - description: คำอธิบายของแบบจำลอง
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def run_simulation(self, *args, **kwargs) -> Dict[str, Any]:
        """
        รันการจำลองและส่งคืนผลลัพธ์
        
        Returns:
        - Dict ที่มีผลลัพธ์การจำลอง
        """
        pass
    
    def get_name(self) -> str:
        """ส่งคืนชื่อของแบบจำลอง"""
        return self.name
    
    def get_description(self) -> str:
        """ส่งคืนคำอธิบายของแบบจำลอง"""
        return self.description
    
    @abstractmethod
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        ส่งคืนข้อมูลพารามิเตอร์สำหรับการแสดงผลใน UI
        
        Returns:
        - Dict ที่มีข้อมูลพารามิเตอร์ในรูปแบบ
          {
              "param_name": {
                  "type": "slider|number|select|multiselect",
                  "label": "คำอธิบายพารามิเตอร์",
                  "min": min_value,  # สำหรับ slider และ number
                  "max": max_value,  # สำหรับ slider และ number
                  "default": default_value,
                  "options": [options],  # สำหรับ select และ multiselect
                  "help": "ข้อความช่วยเหลือ"
              },
              ...
          }
        """
        pass