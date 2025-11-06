from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import io

BREED_INFO_MAP = {
    # 50가지 대표 품종 정보 (MobileNetV2 기준)
    'Chihuahua': {'ko': '치와와', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/4/4c/Chihuahua1_bv.jpg'},
    'Japanese_spaniel': {'ko': '재패니즈 친', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/5/5a/Japanese_Chin_in_Tallinn.jpg'},
    'Maltese_dog': {'ko': '말티즈', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/d/d3/Maltese_dog_in_Taiwan.jpg'},
    'Pekinese': {'ko': '페키니즈', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/1/15/Pekingese_puppy_in_Tallinn.jpg'},
    'Shih-Tzu': {'ko': '시추', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/e/e0/Shih_Tzu_with_short_hair.jpg'},
    'Blenheim_spaniel': {'ko': '카발리에 킹 찰스 스패니얼 (블렌하임)', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/5/5f/Kavalier-King-Charles-Spaniel-Blenheim.jpg'},
    'papillon': {'ko': '파피용', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/c/c1/Papillon_Dog_Standing.jpg'},
    'toy_terrier': {'ko': '토이 테리어', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/a/a8/Toy_Terrier_2.jpg'},
    'Rhodesian_ridgeback': {'ko': '로디지안 리지백', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/a/a2/Rhodesian_Ridgeback_18_months_old.jpg'},
    'Afghan_hound': {'ko': '아프간 하운드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/0/03/Afghan_Hound_1.jpg'},
    'basset': {'ko': '바셋 하운드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/0/05/Basset_Hound_do_Kastelo_de_Gentil.jpg'},
    'beagle': {'ko': '비글', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/5/55/Beagle_600.jpg'},
    'bloodhound': {'ko': '블러드하운드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/5/57/Bloodhound_pico.jpg'},
    'Walker_hound': {'ko': '워커 하운드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/6/64/Walker_hound.jpg'},
    'English_foxhound': {'ko': '잉글리시 폭스하운드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/3/30/EnglishFoxhound.jpg'},
    'redbone': {'ko': '레드본 쿤하운드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/0/06/Redbone_Coonhound_stacked.jpg'},
    'borzoi': {'ko': '보르조이', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/6/63/Borzoi_male_p_1010195.jpg'},
    'Irish_wolfhound': {'ko': '아이리시 울프하운드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/7/7c/Irish_Wolfhound_fawn.jpg'},
    'whippet': {'ko': '휘핏', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/5/5d/Whippet_grey.jpg'},
    'Norwegian_elkhound': {'ko': '노르웨이 엘크하운드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/c/c5/Norwegian_Elkhound_stacked.jpg'},
    'otterhound': {'ko': '오터하운드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/0/01/Otterhound_CH_Lonestar_Granger_2004.jpg'},
    'Saluki': {'ko': '살루키', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/3/3f/Saluki.jpg'},
    'Scottish_deerhound': {'ko': '스코티시 디어하운드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/a/af/Scottish_deerhound_02.jpg'},
    'Weimaraner': {'ko': '바이마라너', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/b/b2/Weimaraner_silver_gray.jpg'},
    'Staffordshire_bullterrier': {'ko': '스태퍼드셔 불 테리어', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/3/3c/Staffordshire_Bull_Terrier_2.jpg'},
    'American_Staffordshire_terrier': {'ko': '아메리칸 스태퍼드셔 테리어', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/6/66/AmStaff_fawn.jpg'},
    'Bedlington_terrier': {'ko': '베들링턴 테리어', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/c/c8/Bedlington-terrier-rop-show.jpg'},
    'Border_terrier': {'ko': '보더 테리어', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/3/31/Border_Terrier_600.jpg'},
    'Kerry_blue_terrier': {'ko': '케리 블루 테리어', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/c/c4/Kerry_Blue_Terrier_Side.jpg'},
    'Irish_terrier': {'ko': '아이리시 테리어', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/7/74/Irish-terrier-show.jpg'},
    'Norfolk_terrier': {'ko': '노퍽 테리어', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/8/8e/Norfolk_Terrier_mit_Kugel.jpg'},
    'Norwich_terrier': {'ko': '노리치 테리어', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/8/8c/Norwich_Terrier.jpg'},
    'Yorkshire_terrier': {'ko': '요크셔 테리어', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/0/00/Yorkshire_Terrier_Kampi.jpg'},
    'Lakeland_terrier': {'ko': '레이클랜드 테리어', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/a/a1/Lakeland_Terrier_gray.jpg'},
    'Boston_bull': {'ko': '보스턴 테리어', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/3/3d/Boston_Terrier_2.jpg'},
    'schnauzer': {'ko': '슈나우저', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/c/c8/Miniature_Schnauzer_stripping.jpg'},
    'golden_retriever': {'ko': '골든 리트리버', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/b/b8/Golden_Retriever_rainy_day.jpg'},
    'Labrador_retriever': {'ko': '래브라도 리트리버', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/3/34/Labrador_on_Quantock_Hills.jpg'},
    'German_shepherd': {'ko': '저먼 셰퍼드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/d/d0/German_Shepherd_-_DSC_0346_%28100963020-O%29.jpg'},
    'Doberman': {'ko': '도베르만', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/b/b6/Dobermann_orig.jpg'},
    'boxer': {'ko': '복서', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/3/31/Boxer_ hund_ 2.jpg'},
    'Great_Dane': {'ko': '그레이트 데인', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/f/fe/Great-Dane-Harlequin.jpg'},
    'Siberian_husky': {'ko': '시베리안 허스키', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/a/a3/Siberian_Husky_2015.jpg'},
    'Pomeranian': {'ko': '포메라니안', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/c/ca/Pomeranian.JPG'},
    'Samoyed': {'ko': '사모예드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/b/b3/Samoyed_dog_in_snow.jpg'},
    'Newfoundland': {'ko': '뉴펀들랜드', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/2/2c/Newfoundland_dog_Smoky.jpg'},
    'collie': {'ko': '콜리', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/6/6a/Rough_Collie_600.jpg'},
    'Border_collie': {'ko': '보더 콜리', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/5/5e/Border_Collie_in_a_city_park.jpg'},
    'Rottweiler': {'ko': '로트와일러', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/f/f6/Rottweiler_standing_facing_left.jpg'},
    'Lhasa': {'ko': '라사 압소', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/8/8a/Lhasa-Apso-Chiots.jpg'},
    'standard_poodle': {'ko': '스탠더드 푸들', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/f/f8/Full_attention_%284067543110%29.jpg'},
    'miniature_poodle': {'ko': '미니어처 푸들', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/2/21/Caniche_nain_GDF_334.jpg'},
    'toy_poodle': {'ko': '토이 푸들', 'img_url': 'https://upload.wikimedia.org/wikipedia/commons/4/4c/Black_toypoodle.jpg'}
}



# Flask 앱 생성
app = Flask(__name__)
model = MobileNetV2(weights='imagenet')
print("INFO: AI 모델 로드가 완료되었습니다.")

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

# --- 👇 2. 분석 API 로직 (이전과 동일) 👇 ---
# (데이터 사전만 커졌을 뿐, 로직은 동일합니다.)
@app.route('/analyze', methods=['POST'])
def analyze_breed():
    if 'file' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400

    file = request.files['file']

    try:
        processed_image = prepare_image(file)
        predictions = model.predict(processed_image)
        decoded_top3 = decode_predictions(predictions, top=3)[0]

        results = []
        for (pred_id, breed_name_en, score) in decoded_top3:

            breed_info = BREED_INFO_MAP.get(breed_name_en) # 딕셔너리에서 조회

            if breed_info:
                result = {
                    'breed_name_en': breed_name_en.replace('_', ' '),
                    'breed_name_ko': breed_info['ko'],
                    'image_url': breed_info['img_url'],
                    'score': float(score)
                }
            else:
                result = {
                    'breed_name_en': breed_name_en.replace('_', ' '),
                    'breed_name_ko': breed_name_en.replace('_', ' '), # 모를 경우 그냥 영어 이름
                    'image_url': None,
                    'score': float(score)
                }
            results.append(result)

        print(f"INFO: 분석 완료: {results}")
        return jsonify(results)

    except Exception as e:
        print(f"ERROR: 이미지 처리 중 오류 발생: {e}")
        return jsonify({'error': f'이미지 처리 중 오류 발생: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)