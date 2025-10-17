# BERT文本相似度計算工具

基於BERT模型的智能文本相似度分析工具，支援中英文分句和F1分數計算。

## 功能特點

- 🤖 **BERT模型支援**: 使用多語言BERT模型進行文本嵌入
- 🌏 **多語言支援**: 自動檢測並支援中文和英文文本
- 📝 **智能分句**: 根據語言特點進行精確分句
- 📊 **F1分數計算**: 提供精確率、召回率和F1分數
- ⚖️ **多種權重方法**: 支援等權重、長度權重和位置權重
- 🌐 **網頁界面**: 現代化的響應式網頁界面
- 🚀 **本地運行**: 完全本地化部署，保護數據隱私

## 安裝依賴

確保您的系統已安裝Python 3.7+，然後安裝所需依賴：

```bash
pip3 install -r requirements.txt
```

## 使用方法

### 1. 啟動應用

```bash
python3 app.py
```

首次運行時會自動下載BERT模型，請耐心等待。

### 2. 訪問網頁界面

在瀏覽器中打開：http://localhost:5000

### 3. 使用界面

1. 在兩個文本框中分別輸入要比較的文本
2. 調整相似度閾值（0-1之間）
3. 選擇權重計算方法：
   - **等權重**: 所有句子權重相同
   - **長度權重**: 根據句子長度分配權重
   - **位置權重**: 前面的句子權重更高
4. 點擊「計算相似度」按鈕
5. 查看詳細的分析結果

## API使用

### 計算相似度

**POST** `/calculate`

請求體：
```json
{
    "text1": "第一個文本",
    "text2": "第二個文本",
    "threshold": 0.5,
    "weight_method": "equal"
}
```

響應：
```json
{
    "f1_score": 0.8500,
    "precision": 0.9000,
    "recall": 0.8000,
    "sentences1": ["句子1", "句子2"],
    "sentences2": ["句子A", "句子B"],
    "matched_pairs": [
        {
            "sentence1": "句子1",
            "sentence2": "句子A",
            "similarity": 0.85,
            "sentence1_idx": 0,
            "sentence2_idx": 0
        }
    ]
}
```

### 健康檢查

**GET** `/health`

## 技術架構

- **後端**: Flask + PyTorch + Transformers
- **前端**: HTML5 + CSS3 + JavaScript
- **模型**: BERT多語言模型 (bert-base-multilingual-cased)
- **文本處理**: jieba (中文) + NLTK (英文)

## 權重計算方法

### 等權重 (equal)
所有句子的權重相同，適用於句子重要性相等的情況。

### 長度權重 (length)
根據句子長度分配權重，較長的句子獲得更高權重。

### 位置權重 (position)
根據句子在文本中的位置分配權重，前面的句子權重更高。

## 注意事項

1. 首次運行需要下載BERT模型（約400MB），請確保網絡連接正常
2. 計算過程需要一定時間，特別是長文本
3. 建議文本長度不超過512個token以獲得最佳性能
4. 相似度閾值建議設置在0.3-0.7之間

## 系統要求

- Python 3.7+
- 至少2GB可用內存
- 支援的操作系統：Windows、macOS、Linux

## 故障排除

### 模型下載失敗
如果模型下載失敗，可以手動下載或使用鏡像源：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 內存不足
如果遇到內存不足問題，可以：
1. 減少輸入文本長度
2. 關閉其他應用程序
3. 使用更小的BERT模型

### 依賴安裝問題
確保使用正確的Python版本和pip：
```bash
python3 --version
pip3 --version
```

## 開發者

此工具基於Transformers庫和Flask框架開發，支援自定義擴展。