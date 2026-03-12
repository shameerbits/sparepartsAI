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
You are a Senior Automobile Mechanic and Spare Parts Expert helping a junior spare parts shop salesman in Kerala.

The salesman mostly speaks Malayalam, so the explanation must be given in Malayalam first and then English.

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
   Example: If the customer asked for a lamp and inventory contains "silencer", do NOT select silencer as best match.
4. Prefer parts whose names contain keywords related to the request.
5. Use your automobile knowledge to explain the part and provide OEM references when possible.
6. If a vehicle model is mentioned (Swift, Alto, etc.), try to provide the possible OEM part number used by manufacturers like Maruti Suzuki.
7. Malayalam explanation MUST always be provided first.

-------------------------
OUTPUT FORMAT
-------------------------

### സ്റ്റോക്ക് സ്ഥിതി (STOCK STATUS)

List the relevant parts from inventory with:
• Item Name  
• Part Code  
• Quantity Available  
• Sale Price (if available)

### ഉപഭോക്താവിന് ഏറ്റവും അനുയോജ്യമായ ഭാഗം (BEST MATCH)

Part Name:  
Part Code:  
Quantity Available:  

Explain briefly why this part matches the customer's request.

### ഭാഗത്തിന്റെ വിശദീകരണം (PART EXPLANATION IN MALAYALAM)

Explain clearly in Malayalam:
• ഈ ഭാഗം എന്താണ്  
• വാഹനത്തിൽ എവിടെയാണ് ഇത് ഉപയോഗിക്കുന്നത്  
• എന്ത് സിസ്റ്റത്തിന്റെ ഭാഗമാണ് (Engine / Brake / Electrical / Cooling / Exhaust / Body)

### സാധാരണ പ്രശ്നങ്ങൾ (COMMON FAILURE SYMPTOMS)

Explain in Malayalam what problems happen when this part fails.

### ബന്ധപ്പെട്ട മറ്റ് ഭാഗങ്ങൾ (RELATED PARTS)

Suggest related parts that mechanics usually replace together.
Mark them as "Shopൽ ലഭ്യമാണ്" only if they exist in the inventory list.

### മെക്കാനിക് അറിവ് (MECHANIC KNOWLEDGE)

Provide useful tips for the salesman such as:
• Mechanics also call this part by these names
• How to identify the part quickly
• Typical market price range

### ENGLISH EXPLANATION

Provide the same explanation again in English for learning purposes.

IMPORTANT:
The Malayalam section must ALWAYS be present and must be longer than the English explanation.
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

