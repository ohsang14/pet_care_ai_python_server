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
# --------------------

def prepare_image(img_file):
    img = Image.open(img_file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# 테스트용 API
@app.route('/', methods=['GET'])
def health_check():
    return "안녕하세요! PetCare AI Python 서버입니다. (모델 로드 완료)"

# AI 품종 분석 API
@app.route('/analyze', methods=['POST'])
def analyze_breed():
    if 'file' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400

    file = request.files['file']

    try:
        processed_image = prepare_image(file)
        predictions = model.predict(processed_image)

        # 3. 예측 결과를 사람이 읽을 수 있는 형태로 변환합니다. (상위 3개)
        decoded = decode_predictions(predictions, top=3)[0]



        # 4. 상위 3개의 결과를 모두 담을 리스트를 생성합니다.
        results_list = []
        for (id, breed_name, score) in decoded:
            result = {
                'breed_name': breed_name.replace('_', ' '),
                'score': float(score)
            }
            results_list.append(result)

        print(f"INFO: 분석 완료 (상위 3개): {results_list}")

        # 5. 단일 객체가 아닌, '리스트'를 JSON으로 반환합니다.
        return jsonify(results_list)


    except Exception as e:
        print(f"ERROR: 이미지 처리 중 오류 발생: {e}")
        return jsonify({'error': f'이미지 처리 중 오류 발생: {e}'}), 500


# 0.0.0.0: 5001번 포트로 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)