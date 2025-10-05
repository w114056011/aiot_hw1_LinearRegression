# Dev Log

## 2025-10-05

### User
建立一個devLog.md，將我們的對話內容記錄下來

### Gemini
Got it. Thanks for the context!

### User
設計一個python sklearn來解決簡單的linear regression問題，並且諄循CRISP-DM步驟

### Gemini
好的，我們將遵循 CRISP-DM 流程，使用 Python 和 scikit-learn 來解決一個簡單的線性回歸問題。

CRISP-DM (跨行業數據挖掘標準流程) 是一個數據挖掘的流程模型，它將數據挖掘的生命週期分為六個階段。我將依序說明並實作每個階段。

**1. 商業理解 (Business Understanding)**
*   **目標：** 預測房價。
*   **問題：** 我們能否根據房子的坪數來預測其價格？這是一個典型的線性回歸問題。

**2. 資料理解 (Data Understanding)**
*   我將生成一些模擬的房價數據，包含「坪數」和「價格」。

**3. 資料準備 (Data Preparation)**
*   我會將數據集分割為訓練集和測試集，以便訓練和評估模型。

**4. 模型建立 (Modeling)**
*   我將使用 `scikit-learn` 的 `LinearRegression` 模型來訓練數據。

**5. 模型評估 (Evaluation)**
*   我將使用均方誤差 (Mean Squared Error) 和 R² 分數來評估模型的性能，並將結果可視化。

**6. 部署 (Deployment)**
*   最後，我會展示如何使用訓練好的模型來預測新的房價。

現在，我將建立一個名為 `linear_regression_crisp_dm.py` 的 Python 腳本來實現這些步驟。

### User
將我們之後的對話紀錄都用新增的方式記錄到devLOog.md裡

### Gemini
好的，我會將我們接下來的對話都記錄到 `devLog.md` 中。

### User
創建一個Todo.md紀錄線性回歸問題的設計步驟，並再創建一個project_plan.md記錄如何實作

### Gemini
好的，我將為您建立 `Todo.md` 和 `project_plan.md`。

### User
修正project_plan.md的內容: 1.我想使用venv創建 python環境。2.允許使用者修改a in ax+b, noise, number of point。3.最後的成果用streamlit框架呈現。

### Gemini
好的，我將根據您的要求修正 `project_plan.md`。

### User
根據project_plan.md完成專案

### Gemini
好的，我將根據 `project_plan.md` 的規劃來完成專案。

### User
重新建立虛擬環境並安裝必要套件後再執行專案

### Gemini
好的，我將再次嘗試建立虛擬環境並執行專案。

### User
修正app.py中 n的範圍100到5000，a的範圍-10到10，noise = N(0, var) where var = 0 to 1000，並顯示Top 5 Outliers

### Gemini
好的，我將根據您的要求修正 `app.py`。

### User
更新 devLog.md。

### Gemini
好的，我現在更新 `devLog.md`。

### User
檢視整個專案，並寫一個markdown report in detail儲存至readme.md

### Gemini
好的，我將檢視整個專案並為您產生一份詳細的 `readme.md` 報告。

### User
add git init and push all files to https://github.com/w114056011/aiot_hw1_LinearRegression.git

### Gemini
好的，我將為您初始化 Git 倉庫，並將所有檔案推送到指定的 GitHub 儲存庫。

### User
專案所使用套件匯出至requirements.txt

### Gemini
好的，我將為您產生 requirements.txt 檔案。

### User
push @requirements.txt to git

### Gemini
好的，您想要將 requirements.txt 推送到 Git 倉庫。

### User
add demo url https://5114056011-aiot-hw1.streamlit.app/ in readme.md

### Gemini
好的，我將把 Demo URL 加入到 readme.md 中。

### User
push to git

### Gemini
好的，您想要將更新後的 readme.md 推送到 Git 倉庫。