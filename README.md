# NKCSim

## คำอธิบาย

NKCSim เป็นโปรแกรมจำลองการทำงานของ NKC Model ซึ่งเป็นการจำลอง
จากทฤษฏี NK Model

## การติดตั้ง

1.  **ติดตั้ง Python:** ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้ง Python 3.x บนเครื่องของคุณแล้ว คุณสามารถตรวจสอบได้โดยการเปิด Terminal หรือ Command Prompt และพิมพ์ `python --version` หรือ `python3 --version`
2.  **ติดตั้งไลบรารีที่จำเป็น:** ใช้ pip เพื่อติดตั้งไลบรารีที่จำเป็นสำหรับการรันโปรแกรม
    ```bash
    pip install -r requirements.txt
    ```

## การใช้งาน

1.  **เปิด Terminal หรือ Command Prompt:** ไปยังไดเรกทอรีที่เก็บไฟล์โปรแกรม NKCSim
2.  **รันโปรแกรม:** ใช้คำสั่งต่อไปนี้เพื่อรันโปรแกรม
    ```bash
    streamlit run nkc_sim.py
    ```
3. **หรือใช้:**
    ```bash
   ./venv/bin/python -m streamlit run nkc_sim.py 
    ``` 