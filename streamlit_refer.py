import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc, rcParams
import os

# 한글 폰트 설정 (나눔고딕)
def set_korean_font():
    font_path = os.path.join(os.getcwd(), "NanumGothic.ttf")
    
    if os.path.exists(font_path):
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        # 전역적으로 폰트 설정
        rcParams['font.family'] = font_name
        rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        rcParams['font.size'] = 12  # 폰트 크기 설정
        st.write(f"폰트 {font_name}가 성공적으로 적용되었습니다.")
    else:
        st.error(f"폰트 파일을 찾을 수 없습니다: {font_path}")

# 폰트 설정 적용
set_korean_font()

# 차량 비교 그래프 그리기
def compare_vehicles(vehicle1, vehicle2, vehicle1_specs, vehicle2_specs):
    specs = ['최소가격', '최대가격', '최소연비', '최대연비', '최소출력', '최대출력']
    
    # 스펙 비교 그래프 그리기
    fig, ax = plt.subplots()
    index = np.arange(len(specs))
    bar_width = 0.35

    bar1 = ax.bar(index, vehicle1_specs, bar_width, label=vehicle1)
    bar2 = ax.bar(index + bar_width, vehicle2_specs, bar_width, label=vehicle2)

    # 축과 타이틀에 한글 텍스트 적용
    ax.set_xlabel('스펙', fontsize=14, fontproperties=font_manager.FontProperties(fname="NanumGothic.ttf"))
    ax.set_ylabel('값', fontsize=14, fontproperties=font_manager.FontProperties(fname="NanumGothic.ttf"))
    ax.set_title(f'{vehicle1} vs {vehicle2} 스펙 비교', fontsize=16, fontproperties=font_manager.FontProperties(fname="NanumGothic.ttf"))
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(specs, fontproperties=font_manager.FontProperties(fname="NanumGothic.ttf"))
    ax.legend(prop=font_manager.FontProperties(fname="NanumGothic.ttf"))

    # 그래프 출력
    st.pyplot(fig)

# 예시 데이터로 차량 비교 함수 호출
vehicle1 = '차량 A'
vehicle2 = '차량 B'
vehicle1_specs = [1000, 2000, 15, 20, 150, 200]
vehicle2_specs = [1100, 2100, 14, 18, 160, 190]

compare_vehicles(vehicle1, vehicle2, vehicle1_specs, vehicle2_specs)
