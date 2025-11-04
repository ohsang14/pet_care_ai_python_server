from flask import Flask, request, jsonify

# Flask 앱 생성
app = Flask(__name__)

# 테스트용 API: 서버가 살아있는지 확인
@app.route('/', methods=['GET'])
def health_check():
    return "안녕하세요! PetCare AI Python 서버입니다."

# 품종 분석 API 엔드포인트
@app.route('/analyze', methods=['POST'])
def analyze_breed():
    # 1. Spring Boot 서버로부터 이미지 파일을 받음
    if 'file' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400

    file = request.files['file']

    # 2. (임시) 지금은 AI 모델 대신, 파일 이름만 반환
    # TODO: 여기에 실제 AI 모델 분석 로직이 들어갈 예정
    print(f"이미지 수신 완료: {file.filename}")

    # 3. (임시) 분석 결과라고 가정하고 JSON 반환
    result = {
        'breed': '골든 리트리버 (임시)',
        'score': 0.92
    }

    return jsonify(result)

# 0.0.0.0: 5001번 포트로 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
