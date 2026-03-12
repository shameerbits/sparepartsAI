# sparepartsAI

## AI Spare Parts Sales Assistant

This is a simple Streamlit app that helps a spare parts shop salesman identify
and sell auto parts using a small "RAG‑Lite" search over a local Excel
inventory plus optional image recognition via OpenAI Vision.

### Features

* Load your Excel inventory (must have columns `item_name`, `item_cd`,
  `cat_name`, `hsncode`, `clsng_bal` etc.)
* Free‑text search with fuzzy matching (rapidfuzz) across name, code, category,
  and HSN code
* Upload a photo of a part and let the OpenAI Vision model suggest keywords
* Mechanic Mode: AI explains matches in simple Malayalam for the salesperson
* All logic lives in a **single file** (`app.py`)

### Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Export your OpenAI key:
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

### Inventory format

The Excel should contain at least the following columns (additional columns
are ignored):

* `item_name` – spare part name
* `item_cd` – part code
* `cat_name` – category name
* `hsncode` – HSN code
* `clsng_bal` – quantity available

The app builds a combined `search_text` field for fuzzy queries so you may
search for things like "toyota brake pad", "h4 bulb" or "EHLB43TP".
