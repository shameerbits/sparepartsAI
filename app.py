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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Image analysis failed: {e}")
        return ""


def mechanic_explanation_english(matches_text: str, user_query: str) -> str:
    prompt = f"""
You are a Senior Automobile Mechanic and Parts Consultant helping a spare parts shop salesman.

Your job is to translate the customer's vague request into the correct spare part and explain it clearly so the salesman can confidently sell it.

Use the inventory matches below as the PRIMARY source of truth for stock availability.

Inventory Matches From Shop:
{matches_text}

Customer Query:
{user_query}


IMPORTANT RULES

1. ONLY mark parts as "In Stock" if they appear in the inventory list above.
2. DO NOT invent stock items.
3. You MAY use general automobile knowledge to explain the part, vehicle system, symptoms, OEM part numbers, and market information.
4. If the vehicle model is mentioned, try to identify the ORIGINAL OEM PART NUMBER used by manufacturers like:
   - Maruti Suzuki
   - Hyundai
   - Toyota
   - Honda
   - Tata
   - Mahindra
5. If OEM part number is uncertain, say "Possible OEM Reference".
6. Keep explanation simple so a salesman with little mechanical knowledge can understand.


OUTPUT FORMAT


STOCK STATUS
List each matched item with:
• Item Name
• Part Code
• Quantity Available
• Sale Price (if available)


BEST MATCH FOR CUSTOMER
Part Name:
Part Code:
Available Quantity:
Why this is the best match:


OEM / ORIGINAL PART NUMBER
Provide OEM part number if known (especially for Maruti Suzuki or common Indian cars).


PART EXPLANATION
Explain in simple words:
• What the part does
• Where it is located in the vehicle
• Which vehicle system it belongs to
  (Engine / Brake / Suspension / Electrical / Cooling / Transmission)


COMMON SYMPTOMS OF FAILURE
Explain what happens when this part fails so salesman can ask the customer.


RELATED PARTS TO SUGGEST
Suggest related parts commonly replaced together
BUT only mark them as "Available" if they exist in the inventory above.


MARKET KNOWLEDGE
Provide helpful information such as:
• Typical market price range
• OEM vs aftermarket difference
• Which brands mechanics prefer


SALESMAN LEARNING TIP
Teach the salesman something useful such as:
• Alternative names mechanics use for this part
• Quick way to identify the part
• Common customer language for this part


MALAYALAM QUICK EXPLANATION
Provide a short explanation in Malayalam so the salesman can easily explain to local customers.


Keep the answer structured, clear, and practical for a spare parts shop.
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


def mechanic_explanation_malayalam(english_explanation: str, matches_text: str) -> str:
    prompt = f"""
Translate and explain the following mechanic explanation to Malayalam.
Maintain accuracy and clarity. Translate ONLY, do not add new information.

Original English explanation:
{english_explanation}

Inventory data (for reference):
{matches_text}

Provide the Malayalam translation clearly.
"""
    try:
        r = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"(error translating to Malayalam: {e})"

# --- Streamlit UI -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_default_inventory():
    """Load default stock.xlsx from the directory if it exists."""
    if os.path.exists("stock.xlsx"):
        return load_inventory("stock.xlsx")
    return None

# Load default inventory
default_df = load_default_inventory()

# Optional file upload
uploaded_file = st.file_uploader("Upload Excel inventory (Optional - replaces default stock)", type=["xls", "xlsx"])
df = None

if uploaded_file:
    df = load_inventory(uploaded_file)
    st.success("✓ Custom inventory loaded.")
elif default_df is not None:
    df = default_df
    st.info("ℹ️  Using default stock.xlsx")
else:
    st.warning("⚠️  No inventory loaded. Please upload a stock file or ensure stock.xlsx exists in the directory.")

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
            
            # Generate English explanation first (accurate)
            english_exp = mechanic_explanation_english(matches_text, query or "image lookup")
            
            # Translate to Malayalam
            malayalam_exp = mechanic_explanation_malayalam(english_exp, matches_text)
            
            # Display one above the other with reasonable size
            st.subheader("English Explanation")
            st.text_area("English", english_exp, height=200, disabled=True, key="eng_exp")
            st.subheader("Malayalam വിവരണം")
            st.text_area("Malayalam", malayalam_exp, height=200, disabled=True, key="mal_exp")

