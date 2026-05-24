# myfinancialapp

## Equity assistant without temporary Gradio links

The app now serves the equity Q&A directly from Flask, so you no longer need a constantly changing `*.gradio.live` URL in `equity.html`.

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Add your data files

Put your source files under:

```text
./data
```

### 3) Build the persistent vector store (one-time or when data changes)

```bash
python scripts_build_vector_store.py
```

This creates:

```text
./chroma_db
```

### 4) Set Gemini API key

```bash
export GEMINI_API_KEY="your_key_here"
```

### 5) Run Flask

```bash
python app.py
```

Use the `/equity` page, which now calls `/equity/query` directly.
