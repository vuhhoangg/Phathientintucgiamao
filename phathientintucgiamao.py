import pandas as pd

# Tạo dữ liệu giả lập
data = {
     "text": [
        "Chính phủ đã công bố chính sách kinh tế mới ngày hôm nay.",
        "Người ngoài hành tinh đã hạ cánh xuống New York ngày hôm qua.",
        "Các nhà khoa học phát hiện ra nước trên sao Hỏa.",
        "Người nổi tiếng tuyên bố uống soda có thể chữa khỏi ung thư.",
        "Đội địa phương đã giành chức vô địch hôm qua.",
"Tin nóng: Người đàn ông tuyên bố đã du hành thời gian đến năm 2050.",
"Ngân hàng trung ương giảm 0,5% lãi suất.",
"Tin giả lan truyền về việc một chính trị gia từ chức.",
"Thành phố mở một bệnh viện mới ở khu vực trung tâm.",
"Bài đăng lan truyền khẳng định Trái Đất phẳng và NASA đã nói dối.",
    ],
    "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Lưu thành file CSV
df.to_csv("news.csv", index=False)

print("Đã tạo file news.csv")
df.head()

from matplotlib import pyplot as plt
_df_0['label'].plot(kind='hist', bins=20, title='label')
plt.gca().spines[['top', 'right',]].set_visible(False)

!pip install -q --upgrade transformers datasets accelerate evaluate
!pip install -q evaluate
import os
import random
import numpy as np
import pandas as pd
seed = 42
random.seed(seed)
np.random.seed(seed)

from google.colab import files
files.download("news.csv")

from datasets import load_dataset

def load_or_upload():
    # Thử tải dataset "liar" từ Hugging Face
    try:
        ds = load_dataset("liar")
        # Dataset liar có label dạng 0..5 (6 lớp truthfulness). Ta chuyển thành binary:
        # giả = labels 0,1? (depends) — để đơn giản: label >=3 treat as REAL else FAKE
        def to_binary(example):
            # original label 'label' in liar: 0-5 where 5 = pants-fire (strongly false) etc.
            example['label'] = 0 if example['label'] >= 3 else 1  # 0=real,1=fake
            return example
        ds = ds.map(to_binary)
        df_train = pd.DataFrame(ds['train'])
        df_valid = pd.DataFrame(ds['validation'])
        df_test  = pd.DataFrame(ds['test'])
        # unify column name text
        if 'statement' in df_train.columns:
            df_train = df_train.rename(columns={'statement':'text'})
            df_valid = df_valid.rename(columns={'statement':'text'})
            df_test  = df_test.rename(columns={'statement':'text'})
        df = pd.concat([df_train[['text','label']], df_valid[['text','label']], df_test[['text','label']]], ignore_index=True)
        print("Loaded 'liar' dataset from Hugging Face. Rows:", len(df))
        return df
    except Exception as e:
        print("Không thể tải 'liar' tự động:", e)
        # Fallback: yêu cầu upload file CSV từ local (Colab)
        from google.colab import files
        print("Hãy upload 1 file CSV có 2 cột: 'text' và 'label' (label: 0=real,1=fake)")
        uploaded = files.upload()
        # Lấy file đầu tiên
        fname = list(uploaded.keys())[0]
        df = pd.read_csv(fname)
        print("Uploaded:", fname, " rows:", len(df))
        return df

df = load_or_upload()
df = df.dropna(subset=['text','label']).reset_index(drop=True)
df.label = df.label.astype(int)
df.head()

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df['label'])
print("Train:", len(train_df), "Test:", len(test_df))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# Tfidf
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
X_train = tfidf.fit_transform(train_df['text'].astype(str))
X_test  = tfidf.transform(test_df['text'].astype(str))
y_train = train_df['label'].values
y_test  = test_df['label'].values

# Model
clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
clf.fit(X_train, y_train)

# Predict + Eval
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 (macro):", f1_score(y_test, y_pred, average='macro'))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

def predict_tfidf(text):
    x = tfidf.transform([text])
    p = clf.predict_proba(x)[0]
    label = clf.predict(x)[0]
    return {"label": int(label), "prob_real": float(p[0]), "prob_fake": float(p[1])}

sample = "The new study proves that drinking coffee reduces risk of X."  # thay bằng văn bản bất kỳ
print(predict_tfidf(sample))

# =======================
# 1. Cài thư viện
# =======================
!pip install evaluate

!pip uninstall -y transformers
!pip install transformers==4.56.2
# =======================
# 2. Import thư viện
# =======================
import evaluate
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# =======================
# 3. Giả lập dataset nhỏ (demo)
# =======================
data = {
    "text": [
        "Chính phủ đã công bố chính sách kinh tế mới ngày hôm nay.",
        "Người ngoài hành tinh đã hạ cánh xuống New York ngày hôm qua.",
        "Các nhà khoa học phát hiện ra nước trên sao Hỏa.",
        "Người nổi tiếng tuyên bố uống soda có thể chữa khỏi ung thư.",
        "Đội địa phương đã giành chức vô địch hôm qua.",
"Tin nóng: Người đàn ông tuyên bố đã du hành thời gian đến năm 2050.",
"Ngân hàng trung ương giảm 0,5% lãi suất.",
"Tin giả lan truyền về việc một chính trị gia từ chức.",
"Thành phố mở một bệnh viện mới ở khu vực trung tâm.",
"Bài đăng lan truyền khẳng định Trái Đất phẳng và NASA đã nói dối.",
    ],
    "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Chia train/test
train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)

# =======================
# 4. Chuẩn bị dataset HuggingFace
# =======================
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds_train = Dataset.from_pandas(train_df.reset_index(drop=True))
ds_eval  = Dataset.from_pandas(test_df.reset_index(drop=True))

def preprocess(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

ds_train = ds_train.map(preprocess, batched=True)
ds_eval = ds_eval.map(preprocess, batched=True)

cols = ['input_ids', 'attention_mask', 'label']
ds_train.set_format(type='torch', columns=cols)
ds_eval.set_format(type='torch', columns=cols)

# =======================
# 5. Load model
# =======================
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# =======================
# 6. TrainingArguments
# =======================
training_args = TrainingArguments(
    output_dir="./tfm-fake-news",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    fp16=torch.cuda.is_available()
)

# =======================
# 7. Evaluation Metrics
# =======================
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=preds, references=labels)['accuracy']
    f1_score = f1.compute(predictions=preds, references=labels, average='macro')['f1']
    return {
        "accuracy": acc,
        "f1_macro": f1_score
    }

# =======================
# 8. Trainer
# =======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    compute_metrics=compute_metrics
)

# =======================
# 9. Train + Evaluate
# =======================
trainer.train()
results = trainer.evaluate()
print("Final evaluation:", results)

# =======================
# 10. Dự đoán thử
# =======================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    label = int(np.argmax(probs))
    return {
        "label": label,
        "prob_real": float(probs[0]),
        "prob_fake": float(probs[1])
    }

# =======================
# 11. Test thử dự đoán
# =======================
print(predict("Các nhà khoa học xác nhận loại vắc-xin mới có hiệu quả 100%."))
print(predict("Người ngoài hành tinh được tìm thấy bên trong Nhà Trắng."))

import os
os.environ["HF_API_TOKEN"] = "nhindongdoiduanhauvichutien"

!pip install evaluate

import transformers
print(transformers.__file__)
print(transformers.__version__)
def predict_transformer(text, top_k=2):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    # Đưa model và input lên GPU nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Dự đoán
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    # Lấy label dự đoán cao nhất
    label = int(np.argmax(probs))

    # Nếu bạn muốn lấy top-k class (trong trường hợp num_labels > 2)
    if top_k > 1:
        topk_indices = probs.argsort()[-top_k:][::-1]
        topk_probs = [(int(i), float(probs[i])) for i in topk_indices]
        return {
            "label": label,
            "top_k": topk_probs,
            "prob_real": float(probs[0]),
            "prob_fake": float(probs[1])
        }

    # Mặc định trả về nhị phân
    return {
        "label": label,
        "prob_real": float(probs[0]),
        "prob_fake": float(probs[1])
    }

# Ví dụ test
print(predict_transformer("Scientists confirmed that the new vaccine is 100% effective."))


