from flask import Flask, render_template, request, jsonify
import json
from simple_similarity import SimpleSimilarityCalculator
import traceback

app = Flask(__name__)

# 全局變量存儲計算器實例
calculator = None

def get_calculator():
    """獲取或創建計算器實例"""
    global calculator
    if calculator is None:
        try:
            calculator = SimpleSimilarityCalculator()
        except Exception as e:
            print(f"初始化相似度計算器時出錯: {e}")
            return None
    return calculator

@app.route('/')
def index():
    """主頁面"""
    return render_template('simple_index.html')

@app.route('/calculate', methods=['POST'])
def calculate_similarity():
    """計算文本相似度的API端點"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': '無效的請求數據'}), 400
        
        text1 = data.get('text1', '').strip()
        text2 = data.get('text2', '').strip()
        threshold = float(data.get('threshold', 0.5))
        weight_method = data.get('weight_method', 'equal')
        
        if not text1 or not text2:
            return jsonify({'error': '請輸入兩個文本進行比較'}), 400
        
        if threshold < 0 or threshold > 1:
            return jsonify({'error': '閾值必須在0到1之間'}), 400
        
        # 獲取計算器
        calc = get_calculator()
        if calc is None:
            return jsonify({'error': '相似度計算器初始化失敗'}), 500
        
        # 計算相似度
        result = calc.calculate_f1_score(text1, text2, threshold, weight_method)
        
        # 格式化結果
        formatted_result = {
            'f1_score': round(result['f1_score'], 4),
            'precision': round(result['precision'], 4),
            'recall': round(result['recall'], 4),
            'sentences1': result['sentences1'],
            'sentences2': result['sentences2'],
            'matched_pairs': [
                {
                    'sentence1': pair['sentence1'],
                    'sentence2': pair['sentence2'],
                    'similarity': round(pair['similarity'], 4),
                    'sentence1_idx': pair['sentence1_idx'],
                    'sentence2_idx': pair['sentence2_idx']
                }
                for pair in result['matched_pairs']
            ],
            'sentence_similarities': [
                {
                    'sentence1': sim['sentence1'],
                    'sentence2': sim['sentence2'],
                    'similarity': round(sim['similarity'], 4),
                    'sentence1_idx': sim['sentence1_idx'],
                    'sentence2_idx': sim['sentence2_idx']
                }
                for sim in result['sentence_similarities']
            ],
            'weights1': [round(w, 4) for w in result['weights1']],
            'weights2': [round(w, 4) for w in result['weights2']],
            'threshold': threshold,
            'weight_method': weight_method
        }
        
        return jsonify(formatted_result)
        
    except Exception as e:
        error_msg = f"計算過程中出錯: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/health')
def health_check():
    """健康檢查端點"""
    try:
        calc = get_calculator()
        if calc is None:
            return jsonify({'status': 'error', 'message': '相似度計算器未初始化'}), 500
        return jsonify({'status': 'ok', 'message': '服務正常運行'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """404錯誤處理"""
    return jsonify({'error': '頁面未找到'}), 404

@app.errorhandler(500)
def internal_error(error):
    """500錯誤處理"""
    return jsonify({'error': '服務器內部錯誤'}), 500

if __name__ == '__main__':
    print("正在啟動文本相似度計算服務...")
    print("使用TF-IDF算法進行相似度計算...")
    
    # 預先初始化計算器
    try:
        print("正在初始化相似度計算器...")
        get_calculator()
        print("相似度計算器初始化完成！")
    except Exception as e:
        print(f"計算器初始化失敗: {e}")
        print("請確保已安裝所有依賴")
    
    # 啟動Flask應用
    app.run(debug=True, host='0.0.0.0', port=5001)