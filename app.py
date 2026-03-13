import os
import base64
import imghdr
import re
import streamlit as st
import pandas as pd
import requests
from rapidfuzz import process, fuzz
from bs4 import BeautifulSoup
from openai import OpenAI
from urllib.parse import quote

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
You are an experienced automobile mechanic helping a junior spare parts shop salesman.

Your job is to understand the customer's request and identify the correct spare part from the shop inventory. 
Explain the part clearly so the salesman can confidently sell it and also learn about automobile parts.

----------------------------------
CUSTOMER REQUEST
----------------------------------
{user_query}

----------------------------------
SHOP INVENTORY MATCHES
----------------------------------
{matches_text}

IMPORTANT RULES

1. The inventory list above is the ONLY source of stock availability.
2. DO NOT invent parts that are not present in the inventory.
3. Some search results may be weak matches. Ignore items that are clearly unrelated to the customer request.
   Example: If the customer asked for a lamp but the inventory contains "silencer", ignore the silencer.
4. Prefer parts whose names strongly match the customer request.
5. Think like a real mechanic and infer the most likely part the customer needs.
6. If a vehicle model is mentioned (Swift, Alto, WagonR, etc.), consider typical parts used in that vehicle.
7. If possible, provide the original OEM part number used by manufacturers such as:
   - Maruti Suzuki
   - Hyundai
   - Toyota
   - Honda
   - Tata
   - Mahindra
8. If the OEM number is uncertain, write "Possible OEM Reference".

----------------------------------
OUTPUT FORMAT
----------------------------------

STOCK STATUS
List relevant parts from the inventory:

• Item Name
• Part Code
• Quantity Available
• Sale Price (if available)

Ignore inventory items that clearly do not match the request.

BEST MATCH FOR CUSTOMER

Part Name:
Part Code:
Quantity Available:

Explain briefly why this is the best match for the customer's request.

PART EXPLANATION

Explain clearly:

• What the part does
• Where it is located in the vehicle
• Which vehicle system it belongs to
  (Engine / Brake / Electrical / Cooling / Exhaust / Suspension / Body)

COMMON FAILURE SYMPTOMS

Explain typical symptoms when this part fails so the salesman can confirm with the customer.

Example symptoms:
• Light not working
• Brake noise
• Engine overheating
• Vehicle not starting

RELATED PARTS TO SUGGEST

Suggest parts that mechanics usually replace together with this part.

Mark parts as "Available in Shop" ONLY if they appear in the inventory list above.

OEM / ORIGINAL PART NUMBER

Provide OEM reference numbers if known.

If uncertain, write:
Possible OEM Reference.

MECHANIC TIP FOR SALESMAN

Teach the salesman useful knowledge such as:

• Other names mechanics use for this part
• How to visually identify it
• Typical market price range
• Popular brands mechanics prefer
"""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"(error generating explanation: {e})"


def _extract_part_name_from_image_desc(image_desc: str) -> str:
    if not image_desc:
        return ""
    lines = [ln.strip(" -*\t") for ln in image_desc.splitlines() if ln.strip()]
    # Prefer explicit labels like "Part name: ..."
    for line in lines:
        m = re.search(r"part\s*name\s*[:\-]\s*(.+)", line, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Fallback to first concise non-empty line.
    return lines[0] if lines else ""


def _parse_part_result_from_text(text: str, fallback_query: str = ""):
    if not text:
        return None
    normalized = " ".join(text.split())
    if len(normalized) < 10:
        return None

    part_no_match = re.search(
        r"(?:part\s*(?:no|number|#)\s*[:\-]?\s*)?([A-Z0-9][A-Z0-9\-]{5,})",
        normalized,
        flags=re.IGNORECASE,
    )
    price_match = re.search(
        r"(?:MRP|Rs\.?|INR)\s*[:\-]?\s*([0-9][0-9,]*(?:\.\d{1,2})?)",
        normalized,
        flags=re.IGNORECASE,
    )

    # Build a practical candidate part name from the front of the snippet.
    part_name = normalized
    for token in ["Part No", "Part Number", "MRP", "Rs.", "INR"]:
        idx = part_name.lower().find(token.lower())
        if idx > 0:
            part_name = part_name[:idx].strip(" -:|,")
            break
    if not part_name and fallback_query:
        part_name = fallback_query

    if not part_no_match and not price_match and not fallback_query:
        return None

    return {
        "Part Name": part_name[:120] if part_name else (fallback_query or "Unknown"),
        "Part Number": part_no_match.group(1).upper() if part_no_match else "N/A",
        "Possible MRP": f"Rs {price_match.group(1)}" if price_match else "N/A",
    }


def maruti_direct_search(query: str, max_items: int = 10):
    base_url = "https://www.marutisuzuki.com/genuine-parts/query"
    query = (query or "").strip()
    if not query:
        return f"{base_url}/", pd.DataFrame(), "No search query provided for Maruti lookup."

    url = f"{base_url}/{quote(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return url, pd.DataFrame(), f"Maruti search returned HTTP {response.status_code}."

        soup = BeautifulSoup(response.text, "html.parser")
        records = []
        seen = set()

        # Parse table rows first (if present).
        for tr in soup.find_all("tr"):
            text = tr.get_text(" ", strip=True)
            item = _parse_part_result_from_text(text, fallback_query=query)
            if not item:
                continue
            key = (item["Part Name"], item["Part Number"], item["Possible MRP"])
            if key in seen:
                continue
            seen.add(key)
            records.append(item)
            if len(records) >= max_items:
                break

        # Fallback parse across likely result containers.
        if not records:
            for el in soup.find_all(["li", "article", "div"], limit=700):
                text = el.get_text(" ", strip=True)
                if not text or len(text) < 18:
                    continue
                if not re.search(r"part|mrp|rs\.?|inr|number|genuine", text, flags=re.IGNORECASE):
                    continue
                item = _parse_part_result_from_text(text, fallback_query=query)
                if not item:
                    continue
                key = (item["Part Name"], item["Part Number"], item["Possible MRP"])
                if key in seen:
                    continue
                seen.add(key)
                records.append(item)
                if len(records) >= max_items:
                    break

        if not records:
            return url, pd.DataFrame(), "No structured part details found from Maruti direct search page."

        return url, pd.DataFrame(records), ""
    except Exception as e:
        return url, pd.DataFrame(), f"Maruti search failed: {e}"




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
        img_desc = ""
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
            
            # Display explanation as a styled webpage section
            st.subheader("Spare Parts Analysis")
            st.markdown(explanation, unsafe_allow_html=True)

        # Maruti Genuine Parts direct search using identified part name/query.
        image_part_name = _extract_part_name_from_image_desc(img_desc)
        maruti_query = (query or "").strip()
        if not maruti_query and image_part_name:
            maruti_query = image_part_name
        if not maruti_query and not matches.empty:
            maruti_query = str(matches.iloc[0].get("item_name", "")).strip()

        st.subheader("Maruti Genuine Parts Direct Search")
        maruti_url, maruti_df, maruti_msg = maruti_direct_search(maruti_query)
        st.write(f"Search URL: {maruti_url}")
        st.write(f"Search Query Used: {maruti_query or 'N/A'}")

        if maruti_msg:
            st.info(maruti_msg)
        if maruti_df.empty:
            st.write("No Maruti direct search results captured for this query.")
        else:
            st.dataframe(maruti_df)

