# 📈 Streamlit Stock Forecast Dashboard with Chatbot

This project is a Streamlit dashboard that provides:

* **Prophet-based stock price forecasting** with interactive plots.
* **Sentiment-based return prediction** using a pre-trained XGBoost model.
* **Retrieval-augmented chatbot** powered by Google Gemini for context-aware Q\&A on your stock data.

---

## Features

* **Upload CSV**: Upload a CSV file containing `Date` and `Close` columns.
* **Prophet Forecast**: View historical stock prices alongside a future forecast with confidence intervals.
* **Raw Data Viewer**: Inspect the last 250 rows of cleaned input data.
* **Sentiment Return Prediction**: Input sentiment metrics to predict next-day returns.
* **Chatbot**: Ask context-based questions about your data (e.g., "What was the price on 2025-01-01?").

---

## Prerequisites

* Python 3.8 or higher
* A Google Gemini API key (for the chatbot)

---

## Setup & Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/<your-username>/streamlit-stock-dashboard.git
   cd streamlit-stock-dashboard
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .\.venv\\Scripts\\activate  # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Place the sentiment model**:

   * Ensure `sentiment_xgb_model.pkl` is in the project root (used for sentiment prediction).

---

## Configuration

The app reads your Gemini API key from either:

* **Environment variable**: `GEMINI_API_KEY`
* **Sidebar input**: Paste your key in the Streamlit sidebar at runtime.

To set it in your shell:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"   # macOS/Linux
set GEMINI_API_KEY="YOUR_GEMINI_API_KEY"      # Windows PowerShell
```

---

## Running Locally

```bash
streamlit run streamlit_stock_dashboard_with_chatbot.py
```

Open your browser at `http://localhost:8501`.

---

## Deployment

### Streamlit Community Cloud

1. Push your code to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app from your repo.
3. Set the `GEMINI_API_KEY` in **Secrets**.
4. Deploy—updates automatically on each push.

### Heroku

1. Add a `Procfile`:

   ```plain
   web: streamlit run streamlit_stock_dashboard_with_chatbot.py --server.port=$PORT
   ```
2. (Optional) Pin Python in `runtime.txt`.
3. Commit & push to Heroku:

   ```bash
   heroku create your-app-name
   heroku config:set GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   git push heroku main
   heroku open
   ```

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

