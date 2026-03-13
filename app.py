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

PART_SYNONYMS = {
    "headlight": "head lamp",
    "head light": "head lamp",
    "head assembly": "head lamp assembly",
    "front light": "head lamp",
    "back light": "tail lamp",
    "rear light": "tail lamp",
    "side mirror": "orvm mirror",
    "mirror": "orvm mirror",
    "fan": "cooling fan",
    "radiator fan": "radiator cooling fan",
}

SEARCH_STOP_WORDS = {"for", "with", "the", "part", "car"}

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


def normalize_query(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip().lower())
    if not q:
        return ""

    # Replace phrase-level synonyms first (longer keys first).
    for src in sorted(PART_SYNONYMS.keys(), key=len, reverse=True):
        q = re.sub(rf"\b{re.escape(src)}\b", PART_SYNONYMS[src], q)

    return re.sub(r"\s+", " ", q).strip()


def interpret_query(query: str) -> str:
    cleaned = (query or "").strip()
    if not cleaned:
        return ""

    prompt = f"""
You are an automobile spare-part query normalizer.

Convert customer wording into a standardized spare part search phrase.

Examples:
- head assembly -> head lamp assembly
- front light -> head lamp
- back light -> tail lamp
- side mirror -> orvm mirror
- radiator fan -> radiator cooling fan assembly

Rules:
- Return ONLY the normalized part name phrase.
- No explanations, no punctuation list, no extra words.
- If the query is already clear, return it unchanged.

Customer query: {cleaned}
"""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0,
        )
        interpreted = (r.choices[0].message.content or "").strip()
        interpreted = re.sub(r"\s+", " ", interpreted).strip(" \n\t\"'")
        return interpreted or cleaned
    except Exception:
        return cleaned


def search_inventory(df: pd.DataFrame, query: str, top_n: int = 5) -> pd.DataFrame:
    if not query or df.empty:
        return pd.DataFrame()
    query = query.lower()

    # Fuzzy match across full search text.
    choices = df["search_text"].tolist()
    fuzzy_results = process.extract(query, choices, scorer=fuzz.WRatio, limit=max(30, top_n * 6))

    score_by_idx = {}
    for _match, score, idx in fuzzy_results:
        score_by_idx[idx] = max(score_by_idx.get(idx, 0), float(score))

    # Token regex match to catch phrase variations and partial terms.
    tokens = [t for t in re.split(r"\W+", query) if t and t not in SEARCH_STOP_WORDS]
    if tokens:
        token_pattern = "|".join(re.escape(t) for t in tokens)
        token_mask = df["search_text"].astype(str).str.contains(token_pattern, case=False, regex=True, na=False)
        token_indices = df[token_mask].index.tolist()
        for idx in token_indices:
            row_text = str(df.at[idx, "search_text"])
            hit_count = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", row_text))
            token_score = 50 + (50 * hit_count / max(1, len(tokens)))
            score_by_idx[idx] = max(score_by_idx.get(idx, 0), float(token_score))

    if not score_by_idx:
        return pd.DataFrame()

    rows = []
    for idx, score in score_by_idx.items():
        row = df.iloc[idx].copy()
        row["_score"] = score
        rows.append(row)

    res_df = pd.DataFrame(rows)
    res_df["clsng_bal_num"] = pd.to_numeric(res_df.get("clsng_bal", 0), errors="coerce").fillna(0)
    res_df = res_df.sort_values(by=["_score", "clsng_bal_num"], ascending=[False, False])
    return res_df.head(top_n)


def rows_to_text(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    subset = df[["item_name", "item_cd", "cat_name", "clsng_bal"]].copy()
    subset.columns = ["Item Name", "Part Code", "Category", "Qty"]
    return subset.to_string(index=False)


def build_related_inventory_context(
    df: pd.DataFrame,
    primary_matches: pd.DataFrame,
    user_query: str,
    max_items: int = 15,
) -> pd.DataFrame:
    """Build a broader but relevant inventory slice for LLM reasoning.

    Includes: primary matches + same categories + query-token fuzzy matches.
    """
    if df.empty or primary_matches.empty:
        return primary_matches

    primary = primary_matches.copy()
    primary["_priority"] = 0
    chunks = [primary]

    # Pull more items from categories already matched (e.g., electrical for head lamp).
    if "cat_name" in df.columns and "cat_name" in primary_matches.columns:
        cats = [str(x).strip().lower() for x in primary_matches["cat_name"].tolist() if str(x).strip()]
        cats = list(dict.fromkeys(cats))
        if cats:
            cat_mask = df["cat_name"].astype(str).str.lower().isin(cats)
            cat_df = df[cat_mask].copy()
            cat_df["_priority"] = 1
            chunks.append(cat_df)

    # Add keyword-based matches from the query terms.
    tokens = [t for t in re.split(r"\W+", (user_query or "").lower()) if len(t) >= 3]
    tokens = [t for t in tokens if t not in {"for", "and", "with", "the", "part", "car"}]
    if tokens:
        token_pattern = "|".join(re.escape(t) for t in tokens)
        tok_mask = df["search_text"].astype(str).str.contains(token_pattern, case=False, regex=True, na=False)
        tok_df = df[tok_mask].copy()
        tok_df["_priority"] = 1
        chunks.append(tok_df)

    combined = pd.concat(chunks, ignore_index=True)
    dedupe_cols = [c for c in ["item_cd", "item_name", "cat_name"] if c in combined.columns]
    if dedupe_cols:
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="first")
    else:
        combined = combined.drop_duplicates(keep="first")

    if "_score" not in combined.columns:
        combined["_score"] = 0
    if "_priority" not in combined.columns:
        combined["_priority"] = 1
    combined["clsng_bal_num"] = pd.to_numeric(combined.get("clsng_bal", 0), errors="coerce").fillna(0)
    combined = combined.sort_values(by=["_priority", "_score", "clsng_bal_num"], ascending=[True, False, False])
    return combined.head(max_items)


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

Your job is to explain inventory results already filtered by the system.
Do not perform part selection beyond the provided inventory list.
Explain clearly so the salesman can confidently sell and learn.

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
3. DO NOT override or re-rank inventory results generated by system search.
4. Only explain parts shown in the provided inventory list.
5. Choose the most relevant item ONLY from this list.
6. If possible, provide the original OEM part number used by manufacturers such as:
   - Maruti Suzuki
   - Hyundai
   - Toyota
   - Honda
   - Tata
   - Mahindra
7. If the OEM number is uncertain, write "Possible OEM Reference".
8. For the RELATED PARTS section, suggest common co-replacement parts even if not found in stock,
   but clearly mark each as either "Available in Shop" or "Not found in current inventory".
9. Explain in simple language suitable for a spare parts shop salesman.
10. Include Malayalam wherever helpful for clarity.

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

For each suggested related part, add an availability tag:
• Available in Shop (only if present in the inventory list above)
• Not found in current inventory

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
        "Category": "N/A",
        "Product URL": "N/A",
        "Source": "Fallback Parse",
    }


def _extract_datalayer_field(text: str, field_name: str) -> str:
    if not text:
        return ""
    match = re.search(rf"{re.escape(field_name)}\s*:\s*'([^']+)'", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


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

        # Parse official listing cards from the main listing container first.
        listing_root = soup.find("div", class_="listingPageMain") or soup
        cards = listing_root.select("div.listingMain div.sliderBox")
        if not cards:
            cards = listing_root.find_all("div", class_="sliderBox")

        for card in cards:
            try:
                name_el = card.find("h3")
                strong_el = card.find("strong")
                price_el = card.find("div", class_="price")
                link_el = card.find("a", href=re.compile(r"^/genuine-parts/")) or card.find("a", href=True)

                onclick_blob = " ".join(
                    el.get("onclick", "")
                    for el in card.find_all(attrs={"onclick": True})
                )

                dl_name = _extract_datalayer_field(onclick_blob, "item_name")
                dl_id = _extract_datalayer_field(onclick_blob, "item_id")
                dl_price = _extract_datalayer_field(onclick_blob, "price")

                name = name_el.get_text(strip=True) if name_el else (dl_name or query)
                part_number = strong_el.get_text(strip=True) if strong_el else (dl_id or "N/A")
                price = price_el.get_text(" ", strip=True) if price_el else (f"MRP: ₹ {dl_price}" if dl_price else "N/A")
                category = card.get("data-category") or "N/A"
                part_url = (
                    f"https://www.marutisuzuki.com{link_el['href']}"
                    if link_el and str(link_el["href"]).startswith("/")
                    else (link_el["href"] if link_el else "N/A")
                )

                query_score = fuzz.WRatio(query.lower(), name.lower()) if query and name else 0

                item = {
                    "Part Name": name,
                    "Part Number": part_number,
                    "Possible MRP": price,
                    "Category": category,
                    "Product URL": part_url,
                    "Source": "Official Card",
                    "_query_score": query_score,
                }

                key = (
                    item["Part Name"].strip().lower(),
                    item["Part Number"].strip().upper(),
                    item["Possible MRP"].strip().lower(),
                )
                if key in seen:
                    continue
                seen.add(key)
                records.append(item)
            except Exception:
                continue

        if records:
            records = sorted(records, key=lambda x: x.get("_query_score", 0), reverse=True)
            records = records[:max_items]
            seen = {
                (
                    i["Part Name"].strip().lower(),
                    i["Part Number"].strip().upper(),
                    i["Possible MRP"].strip().lower(),
                )
                for i in records
            }

        # Parse table rows first (if present).
        for tr in listing_root.find_all("tr"):
            text = tr.get_text(" ", strip=True)
            item = _parse_part_result_from_text(text, fallback_query=query)
            if not item:
                continue
            key = (
                item["Part Name"].strip().lower(),
                item["Part Number"].strip().upper(),
                item["Possible MRP"].strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(item)
            if len(records) >= max_items:
                break

        # Fallback parse across likely result containers.
        if len(records) < max_items:
            for el in listing_root.find_all(["li", "article", "div"], limit=700):
                text = el.get_text(" ", strip=True)
                if not text or len(text) < 18:
                    continue
                if not re.search(r"part|mrp|rs\.?|inr|number|genuine", text, flags=re.IGNORECASE):
                    continue
                item = _parse_part_result_from_text(text, fallback_query=query)
                if not item:
                    continue
                key = (
                    item["Part Name"].strip().lower(),
                    item["Part Number"].strip().upper(),
                    item["Possible MRP"].strip().lower(),
                )
                if key in seen:
                    continue
                seen.add(key)
                records.append(item)
                if len(records) >= max_items:
                    break

        if not records:
            return url, pd.DataFrame(), "No structured part details found from Maruti direct search page."

        output_df = pd.DataFrame(records)
        if "_query_score" in output_df.columns:
            output_df = output_df.drop(columns=["_query_score"])
        return url, output_df, ""
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
        img_desc = ""
        image_part_name = ""
        if image_file:
            st.info("Analyzing image...")
            img_desc = analyze_image(image_file)
            st.write("**Image recognition output:**")
            st.write(img_desc)
            image_part_name = _extract_part_name_from_image_desc(img_desc)

        raw_customer_query = (query or "").strip() or image_part_name
        normalized_query = normalize_query(raw_customer_query)
        interpreted_query = interpret_query(normalized_query) if normalized_query else ""
        deterministic_search_query = interpreted_query or normalized_query or raw_customer_query

        if not deterministic_search_query and img_desc:
            deterministic_search_query = normalize_query(_extract_part_name_from_image_desc(img_desc) or img_desc[:80])

        if deterministic_search_query:
            st.caption(f'System interpreted query as: "{deterministic_search_query}"')

        matches = search_inventory(df, deterministic_search_query)
        if matches.empty:
            st.write("No matching items found.")
        else:
            display_df = matches[["item_name", "item_cd", "cat_name", "clsng_bal"]].head(5).copy()
            display_df.columns = ["Item Name", "Part Code", "Category", "Qty"]
            st.dataframe(display_df)

            # Give LLM slightly broader relevant stock context for related-part suggestions.
            explanation_context_df = build_related_inventory_context(
                df=df,
                primary_matches=matches,
                user_query=deterministic_search_query,
                max_items=15,
            )
            matches_text = rows_to_text(explanation_context_df)
            
            # Generate comprehensive explanation (includes Malayalam)
            explanation_input = deterministic_search_query or query or image_part_name or "image lookup"
            explanation = mechanic_explanation_english(matches_text, explanation_input)
            
            # Display explanation as a styled webpage section
            st.subheader("Spare Parts Analysis")
            st.markdown(explanation, unsafe_allow_html=True)

        # Maruti Genuine Parts direct search using identified part name/query.
        maruti_query = (deterministic_search_query or "").strip()
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
            if "Source" in maruti_df.columns:
                maruti_df = maruti_df.sort_values(by="Source", ascending=True)
                official_count = int((maruti_df["Source"] == "Official Card").sum())
                st.caption(f"Official Maruti cards found: {official_count} / {len(maruti_df)}")

            ordered_cols = [
                "Part Name",
                "Part Number",
                "Possible MRP",
                "Category",
                "Product URL",
                "Source",
            ]
            display_cols = [c for c in ordered_cols if c in maruti_df.columns]
            st.dataframe(maruti_df[display_cols] if display_cols else maruti_df)

