from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import io

# Flask 앱 생성
app = Flask(__name__)

# --- AI 모델 로드 ---
model = MobileNetV2(weights='imagenet')
print("INFO: AI 모델 로드가 완료되었습니다.")


# --------------------E

# 이미지 변환 함수 (이전과 동일)
def prepare_image(img_file):
    img = Image.open(img_file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)


# 테스트용 API (이전과 동일)
@app.route('/', methods=['GET'])
def health_check():
    return "안녕하세요! PetCare AI Python 서버입니다. (모델 로드 완료)"


# --- 👇 2. 분석 API 로직 수정 👇 ---
# (데이터 사전 조회 로직이 모두 삭제되고, 순수 AI 결과만 반환)
@app.route('/analyze', methods=['POST'])
def analyze_breed():
    if 'file' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400

    file = request.files['file']

    try:
        processed_image = prepare_image(file)
        predictions = model.predict(processed_image)
        decoded_top3 = decode_predictions(predictions, top=3)[0]

        # 반환할 결과 리스트
        results = []

        # 상위 3개 결과를 모두 처리
        for (pred_id, breed_name_en, score) in decoded_top3:
            # 2-1. AI가 반환한 순수 영어 이름과 확률만 사용
            result = {
                'breed_name_en': breed_name_en,  # 예: 'Maltese_dog'
                'score': float(score)
            }
            results.append(result)

        print(f"INFO: 분석 완료 (단순): {results}")

        # 2-2. 한국어 이름/이미지 URL이 없는 '순수 AI 결과' 리스트를 반환
        return jsonify(results)

    except Exception as e:
        print(f"ERROR: 이미지 처리 중 오류 발생: {e}")
        return jsonify({'error': f'이미지 처리 중 오류 발생: {e}'}), 500


# --- 👆 2. 분석 API 로직 수정 끝 👆 ---

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)