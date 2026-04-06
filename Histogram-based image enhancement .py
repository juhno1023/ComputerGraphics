import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_detailed_metrics(image):
    """단일 채널(Y: 밝기)에 대한 구체적인 통계 수치와 엔트로피를 계산합니다."""
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb_img[:,:,0]

    mean_val = np.mean(y_channel)
    std_val = np.std(y_channel)
    min_val = np.min(y_channel)
    max_val = np.max(y_channel)

    hist = cv2.calcHist([y_channel], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

    return mean_val, std_val, min_val, max_val, entropy

def show_results_with_metrics_and_table(titles, images):
    """이미지, 히스토그램, 수치형 데이터 테이블을 함께 시각화합니다."""
    metrics_data = []
    fig = plt.figure(figsize=(18, 14))

    for i in range(len(images)):
        mean_val, std_val, min_val, max_val, entropy = calculate_detailed_metrics(images[i])

        metrics_data.append({
            "Method": titles[i],
            "Entropy (정보량)": round(entropy, 3),
            "Contrast (Std, 대비)": round(std_val, 2),
            "Mean Intensity (평균 밝기)": round(mean_val, 2),
            "Min Pixel (최소 밝기 값)": min_val,
            "Max Pixel (최대 밝기 값)": max_val
        })

        # 1. 이미지 출력 (상단)
        ax1 = plt.subplot(3, 3, i + 1)
        ax1.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax1.set_title(f"{titles[i]}\nEntropy: {entropy:.2f} | Std: {std_val:.2f}", fontsize=13, fontweight='bold')
        ax1.axis('off')

        # 2. 히스토그램 및 CDF 출력 (중단)
        ax2 = plt.subplot(3, 3, i + 4)
        ycrcb_img = cv2.cvtColor(images[i], cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_img[:,:,0]
        hist = cv2.calcHist([y_channel], [0], None, [256], [0, 256])

        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()

        ax2.bar(np.arange(256), hist.ravel(), color='gray', alpha=0.7)
        ax2.plot(cdf_normalized, color='red', linewidth=2, label='CDF')
        ax2.set_xlim([0, 256])
        if i == 0:
            ax2.set_ylabel("Frequency")
            ax2.legend(loc='upper left')

    # 3. 콘솔 텍스트 출력
    df_metrics = pd.DataFrame(metrics_data)
    print("=" * 70)
    print(" 📊 이미지 변환 전/후 정량적 평가 지표 (Y 채널 기준)")
    print("=" * 70)
    print(df_metrics.to_string(index=False))
    print("=" * 70)

    # 4. 하단 표(Table) 시각화
    ax_table = plt.subplot(3, 1, 3)
    ax_table.axis('off')

    table = ax_table.table(cellText=df_metrics.values,
                           colLabels=df_metrics.columns,
                           cellLoc='center',
                           loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')

    plt.tight_layout()
    plt.show()

def improve_color_standard_he(image):
    """1. 기본 히스토그램 균일화 (Global HE)"""
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)
    y_equalized = cv2.equalizeHist(y)
    ycrcb_equalized = cv2.merge([y_equalized, cr, cb])
    return cv2.cvtColor(ycrcb_equalized, cv2.COLOR_YCrCb2BGR)

def improve_color_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """2. 고급 기법: CLAHE"""
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# --- 메인 실행부 ---
if __name__ == "__main__":
    # Colab에 업로드한 실제 이미지 경로
    image_path = 'test_image3.png'
    img = cv2.imread(image_path)

    # 이미지 로드 실패 시 에러 처리
    if img is None:
        print(f"❌ 오류: '{image_path}' 경로에서 이미지를 찾을 수 없습니다.")
        print("💡 해결 방법: Colab 좌측 폴더 아이콘을 클릭하여 'sample_data' 폴더 안에 'test_image.png' 파일이 정확히 업로드 되어있는지 확인해주세요.")
    else:
        print("✅ 이미지를 성공적으로 불러왔습니다. 분석을 시작합니다...\n")

        # 기법 적용
        standard_he_img = improve_color_standard_he(img)
        clahe_img = improve_color_clahe(img, clip_limit=3.0, tile_grid_size=(16, 16))

        # 결과 시각화
        titles = ['Original Image', 'Standard HE', 'CLAHE']
        images = [img, standard_he_img, clahe_img]

        show_results_with_metrics_and_table(titles, images)