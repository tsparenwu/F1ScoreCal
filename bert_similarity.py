import re
import os
import jieba
import nltk
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import torch
import warnings
warnings.filterwarnings('ignore')

# 增加 HuggingFace 下載超時設定，避免連線不穩定導致啟動卡住
os.environ.setdefault("HF_HUB_TIMEOUT", "60")
# 使用鏡像源以提高可用性（若未設定則使用預設鏡像）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 下載NLTK數據（如果需要）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BertSimilarityCalculator:
    def __init__(self, model_name='bert-base-multilingual-cased'):
        """
        初始化BERT相似度計算器
        
        Args:
            model_name: BERT模型名稱，默認使用多語言BERT
        """
        # 優先嘗試離線載入（使用本地快取），避免啟動時因網路逾時而卡住
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                local_files_only=True
            )
        except Exception as e:
            # 若離線載入失敗，退回線上下載並啟用續傳
            print(f"離線載入失敗，改用線上下載: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                resume_download=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                resume_download=True
            )
        self.model.eval()
    
    def detect_language(self, text):
        """
        檢測文本語言（簡單的中英文檢測）
        
        Args:
            text: 輸入文本
            
        Returns:
            'zh' for Chinese, 'en' for English
        """
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        
        if chinese_chars / max(total_chars, 1) > 0.3:
            return 'zh'
        else:
            return 'en'
    
    def sentence_split(self, text, language=None):
        """
        根據語言進行分句
        
        Args:
            text: 輸入文本
            language: 語言類型，如果為None則自動檢測
            
        Returns:
            句子列表
        """
        if language is None:
            language = self.detect_language(text)
        
        if language == 'zh':
            # 中文分句
            sentences = re.split(r'[。！？；\n]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            # 英文分句
            sentences = nltk.sent_tokenize(text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def get_bert_embedding(self, text):
        """
        獲取文本的BERT嵌入向量
        
        Args:
            text: 輸入文本
            
        Returns:
            BERT嵌入向量
        """
        inputs = self.tokenizer(text, return_tensors='pt', 
                               truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]標記的嵌入作為句子表示
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings
    
    def calculate_sentence_similarity(self, sent1, sent2):
        """
        計算兩個句子的相似度
        
        Args:
            sent1: 第一個句子
            sent2: 第二個句子
            
        Returns:
            相似度分數 (0-1)
        """
        emb1 = self.get_bert_embedding(sent1)
        emb2 = self.get_bert_embedding(sent2)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return max(0, similarity)  # 確保非負
    
    def calculate_f1_score(self, text1, text2, threshold=0.5, weight_method='equal'):
        """
        計算兩個文本的F1分數
        
        Args:
            text1: 第一個文本
            text2: 第二個文本
            threshold: 相似度閾值，超過此值認為句子匹配
            weight_method: 權重計算方法 ('equal', 'length', 'position')
            
        Returns:
            字典包含詳細結果
        """
        # 分句
        sentences1 = self.sentence_split(text1)
        sentences2 = self.sentence_split(text2)
        
        if not sentences1 or not sentences2:
            return {
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'sentence_similarities': [],
                'matched_pairs': [],
                'sentences1': sentences1,
                'sentences2': sentences2
            }
        
        # 計算所有句子對的相似度
        similarity_matrix = np.zeros((len(sentences1), len(sentences2)))
        sentence_similarities = []
        
        for i, sent1 in enumerate(sentences1):
            for j, sent2 in enumerate(sentences2):
                sim = self.calculate_sentence_similarity(sent1, sent2)
                similarity_matrix[i][j] = sim
                sentence_similarities.append({
                    'sentence1_idx': i,
                    'sentence2_idx': j,
                    'sentence1': sent1,
                    'sentence2': sent2,
                    'similarity': sim
                })
        
        # 找到最佳匹配（貪心算法）
        matched_pairs = []
        used_i = set()
        used_j = set()
        
        # 按相似度排序
        sorted_similarities = sorted(sentence_similarities, 
                                   key=lambda x: x['similarity'], reverse=True)
        
        for sim_info in sorted_similarities:
            i, j = sim_info['sentence1_idx'], sim_info['sentence2_idx']
            if i not in used_i and j not in used_j and sim_info['similarity'] >= threshold:
                matched_pairs.append(sim_info)
                used_i.add(i)
                used_j.add(j)
        
        # 計算權重
        weights1 = self._calculate_weights(sentences1, weight_method)
        weights2 = self._calculate_weights(sentences2, weight_method)
        
        # 計算加權精確率和召回率
        if len(sentences1) == 0:
            precision = 0.0
        else:
            matched_weight1 = sum(weights1[pair['sentence1_idx']] for pair in matched_pairs)
            total_weight1 = sum(weights1)
            precision = matched_weight1 / total_weight1 if total_weight1 > 0 else 0.0
        
        if len(sentences2) == 0:
            recall = 0.0
        else:
            matched_weight2 = sum(weights2[pair['sentence2_idx']] for pair in matched_pairs)
            total_weight2 = sum(weights2)
            recall = matched_weight2 / total_weight2 if total_weight2 > 0 else 0.0
        
        # 計算F1分數
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'sentence_similarities': sentence_similarities,
            'matched_pairs': matched_pairs,
            'sentences1': sentences1,
            'sentences2': sentences2,
            'similarity_matrix': similarity_matrix.tolist(),
            'weights1': weights1,
            'weights2': weights2
        }
    
    def _calculate_weights(self, sentences, method='equal'):
        """
        計算句子權重
        
        Args:
            sentences: 句子列表
            method: 權重計算方法
            
        Returns:
            權重列表
        """
        n = len(sentences)
        if n == 0:
            return []
        
        if method == 'equal':
            return [1.0] * n
        elif method == 'length':
            lengths = [len(sent) for sent in sentences]
            total_length = sum(lengths)
            return [length / total_length for length in lengths] if total_length > 0 else [1.0/n] * n
        elif method == 'position':
            # 給前面的句子更高權重
            weights = [1.0 / (i + 1) for i in range(n)]
            total_weight = sum(weights)
            return [w / total_weight for w in weights]
        else:
            return [1.0] * n

# 測試函數
def test_similarity_calculator():
    """測試相似度計算器"""
    calculator = BertSimilarityCalculator()
    
    # 中文測試
    text1_zh = "今天天氣很好。我很開心。"
    text2_zh = "今日天氣不錯。我感到高興。"
    
    print("中文測試:")
    result_zh = calculator.calculate_f1_score(text1_zh, text2_zh)
    print(f"F1 Score: {result_zh['f1_score']:.4f}")
    print(f"Precision: {result_zh['precision']:.4f}")
    print(f"Recall: {result_zh['recall']:.4f}")
    print()
    
    # 英文測試
    text1_en = "The weather is nice today. I am happy."
    text2_en = "Today's weather is good. I feel joyful."
    
    print("英文測試:")
    result_en = calculator.calculate_f1_score(text1_en, text2_en)
    print(f"F1 Score: {result_en['f1_score']:.4f}")
    print(f"Precision: {result_en['precision']:.4f}")
    print(f"Recall: {result_en['recall']:.4f}")

if __name__ == "__main__":
    test_similarity_calculator()