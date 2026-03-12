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
            "Identify the spare part shown in this image.\n"
            "Provide:\n"
            "* Part name\n"
            "* What it does in the vehicle\n"
            "* Relevant vehicle system (Engine, Brake, Suspension, Electrical, Cooling, Exhaust, Body)\n"
            "* Keywords to search for this part\n\n"
            "Be concise and practical for spare parts shop use.\n"
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
You are a Senior Automobile Mechanic and Spare Parts Expert helping a junior spare parts shop salesman.

Your job is to understand the customer's request and explain the most relevant spare part from the shop inventory.

The salesman is still learning about automobile parts, so provide helpful explanations.

-------------------------
CUSTOMER REQUEST
-------------------------
{user_query}

-------------------------
INVENTORY MATCHES FROM SHOP
-------------------------
{matches_text}

IMPORTANT RULES

1. The inventory list above is the ONLY source of stock availability.
2. DO NOT invent items that are not in the inventory.
3. Some items may be loosely matched by the search system. If an item is clearly unrelated to the customer request, ignore it.
   Example: If the customer asked for a "lamp" and inventory contains "silencer", do NOT select silencer as best match.
4. Prefer parts whose names contain keywords related to the request.
5. Use your automotive knowledge to explain the part and provide OEM references when possible.
6. If the vehicle model is mentioned (Swift, Alto, etc.), try to provide the possible OEM part number used by manufacturers such as:
   - Maruti Suzuki
   - Hyundai
   - Toyota
   - Honda
   - Tata
   - Mahindra
7. If OEM number is uncertain, write: "Possible OEM Reference".

-------------------------
OUTPUT FORMAT
-------------------------

STOCK STATUS
List all relevant items from the inventory with:
• Item Name
• Part Code
• Quantity Available
• Sale Price (if available)

BEST MATCH FOR CUSTOMER
Part Name:
Part Code:
Quantity Available:

Reason why this is the best match.

OEM / ORIGINAL PART NUMBER
Provide OEM reference if known (especially for Maruti Suzuki vehicles).

PART EXPLANATION (ENGLISH)
Explain clearly:
• What this part does
• Where it is located in the vehicle
• Which vehicle system it belongs to
  (Engine / Brake / Suspension / Electrical / Cooling / Exhaust / Body)

COMMON FAILURE SYMPTOMS
Explain symptoms when this part fails so the salesman can ask the customer.

RELATED PARTS TO SUGGEST
Suggest parts commonly replaced together.
Mark them as "Available in Shop" ONLY if they exist in the inventory list above.

MARKET KNOWLEDGE
Provide helpful mechanic knowledge such as:
• Typical market price range
• OEM vs aftermarket differences
• Popular brands mechanics prefer

SALESMAN LEARNING TIP
Teach the salesman useful knowledge such as:
• Other names mechanics use for this part
• How to visually identify it
• Common customer words for this part

-------------------------
MALAYALAM EXPLANATION (MANDATORY)
-------------------------

Provide a short explanation in Malayalam so the salesman can easily explain to local customers.

The Malayalam explanation must include:
• What the part is
• What it does in the vehicle

Use simple spoken Malayalam used in Kerala spare parts shops.

Example style:
"ഇത് കാർ ഹെഡ് ലൈറ്റ് ബൾബ് ആണ്. രാത്രിയിൽ ലൈറ്റ് നൽകാൻ ഉപയോഗിക്കുന്ന ഭാഗമാണ്."

The Malayalam explanation MUST always be provided.
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
            
            # Generate comprehensive explanation (includes Malayalam)
            explanation = mechanic_explanation_english(matches_text, query or "image lookup")
            
            # Display explanation
            st.subheader("Spare Parts Analysis")
            st.text_area("Analysis", explanation, height=400, disabled=True, key="explanation")

