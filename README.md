# 🖼️ Image Caption Generator – Deep Learning Web App

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Build-Passing-brightgreen?style=flat-square)

A modern **AI-powered web application** that generates natural language captions from images using deep learning (CNN + LSTM).

---

## ✨ Demo Preview

> 📸 Upload an image → 🧠 AI analyzes → 📝 Caption is generated instantly
<img width="1063" height="1239" alt="image" src="https://github.com/user-attachments/assets/ba259840-cc0e-4a05-b560-4794b627ac98" />

---

## 🚀 Features

✅ Drag & drop image upload  
✅ Real-time caption generation  
✅ Clean and interactive UI  
✅ CNN + LSTM deep learning model  
✅ Fast inference using pre-trained weights  

---

## 📁 Project Structure

```bash
image-caption-generator/
│── main.py
│── requirements.txt
│── README.md
│
├── outputs/
│   └── models/
│       ├── model.keras
│       ├── feature_extractor.keras
│       └── tokenizer.pkl
│
└── dataset/
    ├── Images/
    └── captions.txt
```

---

## ⚙️ How to Run Locally

1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/image-caption-generator.git
cd image-caption-generator
```



2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Run the App
```bash
streamlit run main.py
```

---

## 🚀 How to Deploy on Streamlit Cloud
1. Push this project to GitHub
2. Go to: https://streamlit.io/cloud
3. Click Create app
4. Connect your GitHub repository
Select:
- Branch: main
- File path: main.py
- Click Deploy

 ---

## 🧑‍💻 How to Use
1. Open the deployed Streamlit app
2. Upload an image
3. Wait a few seconds
4. View the generated caption above the image

---

## 🔗 App Endpoint
Once deployed, your app URL acts as the endpoint:
[App_Link](https://captiontext.streamlit.app/)
https://captiontext.streamlit.app/

---

## 🧩 Model Details

### Model 1 — CNN + LSTM

| Component | Description |
|-----------|-------------|
| Feature Extraction | CNN for image feature extraction |
| Sequence Model | LSTM (Long Short-Term Memory) |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |

### Model 2 — CNN + Transformer

| Component | Description |
|-----------|-------------|
| Feature Extraction | EfficientNetB0 (Pretrained CNN) |
| Decoder | Transformer with Multi-Head Attention |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |


---

## 📊 Evaluation Metrics
1. BLEU-1 to BLEU-4
2. METEOR
3. ROUGE-L
<table>
<tr>

<td valign="top">

### CNN + Transformer

<table>
<tr><th>Metric</th><th>Score</th></tr>
<tr><td>BLEU-1</td><td>0.4412</td></tr>
<tr><td>BLEU-2</td><td>0.2359</td></tr>
<tr><td>BLEU-3</td><td>0.0098</td></tr>
<tr><td>BLEU-4</td><td>0.0019</td></tr>
<tr><td>METEOR</td><td>0.3820</td></tr>
<tr><td>ROUGE-L Precision</td><td>0.3750</td></tr>
<tr><td>ROUGE-L Recall</td><td>0.5111</td></tr>
<tr><td>ROUGE-L F1</td><td>0.4253</td></tr>
</table>

</td>

<td valign="top">

### CNN + LSTM

<table>
<tr><th>Metric</th><th>Score</th></tr>
<tr><td>BLEU-1</td><td>0.504884</td></tr>
<tr><td>BLEU-2</td><td>0.317433</td></tr>
<tr><td>BLEU-3</td><td>0.191181</td></tr>
<tr><td>BLEU-4</td><td>0.113471</td></tr>
<tr><td>METEOR</td><td>0.277254</td></tr>
<tr><td>ROUGE-L Precision</td><td>0.4519</td></tr>
<tr><td>ROUGE-L Recall</td><td>0.3534</td></tr>
<tr><td>ROUGE-L F1</td><td>0.374470</td></tr>
</table>

</td>

</tr>
</table>


---
## 📜 License
This project is licensed under the MIT License.
