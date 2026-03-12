import os
import base64
import imghdr
import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
from openai import OpenAI

# initialize client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

st.title("AI Spare Parts Sales Assistant")

# --- inventory helpers -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_inventory(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df = df.fillna("")
    # ensure important columns exist as strings
    for col in ["item_name", "item_cd", "cat_name", "hsncode", "clsng_bal"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
        else:
            df[col] = ""
    # create searchable field
    df["search_text"] = (
        df["item_name"] + " " + df["item_cd"] + " " + df["cat_name"] + " " + df["hsncode"]
    ).str.lower()
    return df


def search_inventory(df: pd.DataFrame, query: str, top_n: int = 5) -> pd.DataFrame:
    if not query or df.empty:
        return pd.DataFrame()
    query = query.lower()
    choices = df["search_text"].tolist()
    results = process.extract(query, choices, scorer=fuzz.WRatio, limit=20)
    rows = []
    for match, score, idx in results:
        row = df.iloc[idx].copy()
        row["_score"] = score
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    res_df = pd.DataFrame(rows)
    res_df["clsng_bal_num"] = pd.to_numeric(res_df.get("clsng_bal", 0), errors="coerce").fillna(0)
    res_df = res_df.sort_values(by=["clsng_bal_num", "_score"], ascending=[False, False])
    return res_df.head(top_n)


def rows_to_text(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    subset = df[["item_name", "item_cd", "cat_name", "clsng_bal"]].copy()
    subset.columns = ["Item Name", "Part Code", "Category", "Qty"]
    return subset.to_string(index=False)


def analyze_image(uploaded_file) -> str:
    try:
        img_bytes = uploaded_file.read()
        ext = imghdr.what(None, img_bytes) or "png"
        b64 = base64.b64encode(img_bytes).decode()
        data_url = f"data:image/{ext};base64,{b64}"
        prompt = (
            "You are an automobile spare parts expert.\n\n"
            "Identify the spare part in this image.\n"
            "Return:\n"
            "* Part name\n"
            "* Category\n"
            "* Possible vehicle systems\n"
            "* Common keywords used to search this part.\n"
        )
        resp = client.responses.create(
            model="gpt-4o-mini-vision",
            input=[
                {"role": "user", "content": prompt},
                {"type": "input_image", "image_url": data_url, "detail": "high"},
            ],
            max_output_tokens=500,
        )
        if hasattr(resp, "output_text"):
            return resp.output_text.strip()
        # fallback: iterate output items
        text_parts = []
        for item in getattr(resp, "output", []):
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            text_parts.append(c.get("text", ""))
                elif isinstance(content, str):
                    text_parts.append(content)
        return "\n".join(text_parts).strip()
    except Exception as e:
        st.error(f"Image analysis failed: {e}")
        return ""


def mechanic_explanation(matches_text: str, user_query: str) -> str:
    prompt = f"""
You are a senior automobile mechanic helping a spare parts shop salesman.

Explain the parts in a simple way.

Inventory Matches:
{matches_text}

Customer Query:
{user_query}

Respond in Malayalam as much as possible.

Explain:
1. Stock status
2. Best matching part
3. Part code
4. Quantity available
5. What the part is used for in the vehicle
6. Which car system it belongs to
7. Suggest related parts if helpful

Keep explanation short and clear for a salesman.
"""
    try:
        r = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"(error generating explanation: {e})"

# --- Streamlit UI -----------------------------------------------------------
uploaded_file = st.file_uploader("Upload Excel inventory", type=["xls", "xlsx"])
df = None
if uploaded_file:
    df = load_inventory(uploaded_file)
    st.success("Inventory loaded.")
else:
    st.info("Please upload inventory to begin.")

query = st.text_input("Enter your search query:")
image_file = st.file_uploader("Upload spare part image", type=["png", "jpg", "jpeg"])

if st.button("Search"):
    if df is None:
        st.warning("Upload inventory first.")
    elif not query and not image_file:
        st.warning("Enter a text query or upload an image.")
    else:
        search_terms = ""
        if image_file:
            st.info("Analyzing image...")
            img_desc = analyze_image(image_file)
            st.write("**Image recognition output:**")
            st.write(img_desc)
            search_terms += img_desc
        if query:
            search_terms = f"{query} {search_terms}" if search_terms else query
        matches = search_inventory(df, search_terms)
        if matches.empty:
            st.write("No matching items found.")
        else:
            display_df = matches[["item_name", "item_cd", "cat_name", "clsng_bal"]].copy()
            display_df.columns = ["Item Name", "Part Code", "Category", "Qty"]
            st.dataframe(display_df)
            matches_text = rows_to_text(matches)
            explanation = mechanic_explanation(matches_text, query or "image lookup")
            st.text_area("Mechanic Explanation (Malayalam)", explanation, height=300)

