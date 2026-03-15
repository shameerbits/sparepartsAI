import os
import base64
import imghdr
import re
import json
import streamlit as st
import pandas as pd
import requests
from rapidfuzz import fuzz
from bs4 import BeautifulSoup
from openai import OpenAI
from urllib.parse import quote

# initialize client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# model selection
SEARCH_JSON_NORMALIZATION_MODEL = os.environ.get(
    "SEARCH_JSON_NORMALIZATION_MODEL",
    os.environ.get("PART_QUERY_MODEL", "gpt-5"),
)
PART_QUERY_FALLBACK_MODEL = os.environ.get("PART_QUERY_FALLBACK_MODEL", "gpt-4.1")
IMAGE_ANALYSIS_MODEL = os.environ.get("IMAGE_ANALYSIS_MODEL", "gpt-4.1-mini")
MECHANIC_RESPONSE_MODEL = os.environ.get("MECHANIC_RESPONSE_MODEL", "gpt-4o-mini")

st.title("AI Spare Parts Sales Assistant")
debug_mode = st.sidebar.checkbox("Enable debug", value=False)
debug_window = st.expander("Debug Window", expanded=False) if debug_mode else None

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
        df["item_name"] + " " +
        df["item_cd"] + " " +
        df["cat_name"] + " " +
        df["hsncode"]
    ).str.lower()

    # normalize common spare part abbreviations
    df["search_text"] = df["search_text"].str.replace("assy", "assembly")
    df["search_text"] = df["search_text"].str.replace("lamp assy", "lamp assembly")
    df["search_text"] = df["search_text"].str.replace("brg", "bearing")
    return df


def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", (text or "").strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace(" ", "-")
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned


def _safe_phrase(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", (text or "").strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


CATALOG_TERM_MAP = {
    "exterior": "outer",
    "interior": "inner",
    "outside": "outer",
    "inside": "inner",
}


def _normalize_catalog_phrase(text: str) -> str:
    normalized = _safe_phrase(text)
    if not normalized:
        return ""
    for src, dst in sorted(CATALOG_TERM_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        normalized = re.sub(rf"\b{re.escape(src)}\b", dst, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _remove_year_tokens(text: str) -> str:
    tokens = [t for t in re.split(r"\W+", (text or "").lower()) if t]
    tokens = [t for t in tokens if not re.fullmatch(r"(19|20)\d{2}", t)]
    return " ".join(tokens).strip()


def _empty_parsed_query() -> dict:
    return {
        "model": "",
        "part_name": "",
        "side": "",
        "type": "",
        "part_number": "",
        "part": "",
        "category": "",
        "hsn": "",
    }


def _build_inventory_rag_query(parsed: dict) -> str:
    fields_order = [
        "model",
        "part_name",
        "side",
        "type",
        "part_number",
        "part",
        "category",
        "hsn",
    ]

    terms = []
    for key in fields_order:
        value = _safe_phrase(str(parsed.get(key, "")))
        if value:
            terms.append(value)

    if parsed.get("part_number"):
        terms.append("part code")

    # de-dupe while preserving order
    seen = set()
    deduped = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)

    return " ".join(deduped).strip()


def build_actionable_search_query(user_query: str) -> dict:
    """Use GPT once to normalize and derive both stock and website search queries."""
    cleaned = re.sub(r"\s+", " ", (user_query or "").strip().lower())
    if not cleaned:
        return {
            "parsed_query": _empty_parsed_query(),
            "normalized_query": "",
            "inventory_query": "",
            "maruti_query": "",
            "parse_error": "empty user query",
        }

    prompt = f"""
You are an automotive spare parts query interpreter.

Your job is to convert a customer's natural language spare-parts search query into a structured JSON format that can be used for accurate RAG search.

Extract information only if it is clearly mentioned or confidently inferred from the query. If a field cannot be determined, leave it as an empty string "".

Return ONLY valid JSON. Do not include explanations.

Fields to extract:

model: vehicle model name (example: swift, baleno, alto)
part_name: official catalogue style name if possible (example: fog lamp assembly, head lamp assembly)
side: LH or RH if left or right is mentioned
type: part version like type1, type2, type3 if mentioned
part_number: infer likely OEM/reference part number from model + part context even if not explicitly mentioned
part: generic part name (example: fog lamp, head lamp, bumper)
category: vehicle system category (examples: lighting, body, engine, electrical, cooling)
hsn: infer HSN code from mapped part/category even when not explicitly mentioned

Catalogue naming preference rules:
- Use standard Indian spare-parts wording.
- Prefer "outer" over "exterior".
- Prefer "inner" over "interior".


Inference rules:
- Prefer confident inference over leaving blank for part_number and hsn.
- If multiple plausible part numbers exist and confidence is low, choose the most common reference format for that part/model.
- Use empty string only when no credible inference is possible.

HSN mapping rules:
head lamp -> 85122010
fog lamp -> 85122020
tail lamp -> 85122090
horn -> 85123000
wiper -> 85124000

Category mapping rules:
head lamp, fog lamp, tail lamp -> lighting
bumper, grille, fender -> body
radiator -> cooling
air filter -> intake
brake pad -> braking

Side mapping rules:
left, driver side -> LH
right, passenger side -> RH

Example:

Customer Query:
"swift 2012 left fog lamp"

Response:
{{
 "model": "swift",
 "part_name": "fog lamp assembly",
 "side": "LH",
 "type": "",
 "part_number": "",
 "part": "fog lamp",
 "category": "lighting",
 "hsn": "85122020"
}}

Customer Query:
"baleno headlight right"

Response:
{{
 "model": "baleno",
 "part_name": "head lamp assembly",
 "side": "RH",
 "type": "",
 "part_number": "",
 "part": "head lamp",
 "category": "lighting",
 "hsn": "85122010"
}}

Now convert the following customer query into the JSON structure.

Customer Query:
{cleaned}
"""

    fallback_normalized = _safe_slug(_remove_year_tokens(cleaned))
    fallback_inventory = _safe_phrase(_remove_year_tokens(cleaned))
    fallback_maruti = _safe_phrase(_remove_year_tokens(cleaned))

    def _parse_with_model(model_name: str) -> dict:
        r = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        payload = json.loads((r.choices[0].message.content or "").strip())
        return {
            "model": _safe_phrase(str(payload.get("model", ""))),
            "part_name": _normalize_catalog_phrase(str(payload.get("part_name", ""))),
            "side": _safe_phrase(str(payload.get("side", ""))),
            "type": _safe_phrase(str(payload.get("type", ""))),
            "part_number": _safe_phrase(str(payload.get("part_number", ""))),
            "part": _normalize_catalog_phrase(str(payload.get("part", ""))),
            "category": _safe_phrase(str(payload.get("category", ""))),
            "hsn": _safe_phrase(str(payload.get("hsn", ""))),
        }

    parsed = None
    parse_errors = []

    for model_name in [SEARCH_JSON_NORMALIZATION_MODEL, PART_QUERY_FALLBACK_MODEL]:
        try:
            parsed = _parse_with_model(model_name)
            break
        except Exception as e:
            parse_errors.append(f"{model_name}: {e}")

    if parsed is None:
        return {
            "parsed_query": _empty_parsed_query(),
            "normalized_query": fallback_normalized,
            "inventory_query": fallback_inventory,
            "maruti_query": fallback_maruti,
            "parse_error": " | ".join(parse_errors) if parse_errors else "unknown parse error",
        }

    normalized_source = " ".join(
        x
        for x in [parsed.get("model", ""), parsed.get("part", "") or parsed.get("part_name", "")]
        if x
    ).strip() or fallback_normalized
    normalized = _safe_slug(normalized_source) or fallback_normalized

    inventory_query = _build_inventory_rag_query(parsed) or fallback_inventory

    maruti_query = _safe_phrase(
        " ".join(
            x
            for x in [
                parsed.get("model", ""),
                parsed.get("part_name", "") or parsed.get("part", ""),
                parsed.get("side", ""),
                parsed.get("type", ""),
            ]
            if x
        )
    ) or fallback_maruti
    maruti_query = _safe_phrase(_remove_year_tokens(maruti_query))

    return {
        "parsed_query": parsed,
        "normalized_query": normalized,
        "inventory_query": inventory_query,
        "maruti_query": maruti_query,
        "parse_error": "",
    }


def search_inventory(df: pd.DataFrame, query: str, top_n: int = 7) -> pd.DataFrame:

    if not query or df.empty:
        return pd.DataFrame()

    query = query.lower()

    # Break query into meaningful tokens
    tokens = [t for t in re.split(r"\W+", query) if len(t) > 2]

    scores = []

    for idx, row in df.iterrows():

        text = str(row["search_text"]).lower()

        score = 0

        # --- Strong phrase match ---
        if query in text:
            score += 120

        # --- Token based matching ---
        for token in tokens:

            if token in text:

                if token in row["item_name"].lower():
                    score += 40      # very strong match

                elif token in row["cat_name"].lower():
                    score += 25      # category match

                else:
                    score += 10      # weak match

        # --- Fuzzy fallback ---
        fuzzy = fuzz.partial_ratio(query, text)
        score += fuzzy * 0.5

        scores.append((idx, score))

    if not scores:
        return pd.DataFrame()

    # Convert to dataframe
    score_df = pd.DataFrame(scores, columns=["idx", "score"])

    score_df = score_df.sort_values("score", ascending=False)

    top_indices = score_df.head(top_n)["idx"].tolist()

    result = df.loc[top_indices].copy()

    result["_score"] = score_df.head(top_n)["score"].values

    result["clsng_bal_num"] = pd.to_numeric(result.get("clsng_bal", 0), errors="coerce").fillna(0)

    result = result.sort_values(by=["_score", "clsng_bal_num"], ascending=[False, False])

    return result


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
        uploaded_file.seek(0)
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
            model=IMAGE_ANALYSIS_MODEL,
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
Return only the two sections below. Do not add any other sections, headings, bullets, or summaries.

PART EXPLANATION
Explain clearly:
- what the part does
- where it is located in the vehicle
- which vehicle system it belongs to

COMMON FAILURE SYMPTOMS
Explain typical symptoms when this part fails so the salesman can confirm with the customer.
"""
    try:
        r = client.chat.completions.create(
            model=MECHANIC_RESPONSE_MODEL,
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

        def _looks_like_noise(text: str) -> bool:
            t = (text or "").strip().lower()
            if not t:
                return True
            return bool(
                re.search(
                    r"showing\s+\d|results\s+for|filters?\s+clear|reset\s+all|apply\s+categories|\bcategories\b",
                    t,
                    flags=re.IGNORECASE,
                )
            )

        # Parse official listing cards from the main listing container first.
        listing_root = soup.find("div", class_="listingPageMain") or soup
        show_result_text = ""
        show_result_el = soup.select_one("div.showResult")
        if show_result_el:
            show_result_text = show_result_el.get_text(" ", strip=True)

        declared_total = None
        if show_result_text:
            m = re.search(r"of\s+(\d+)\s+results", show_result_text, flags=re.IGNORECASE)
            if m:
                declared_total = int(m.group(1))

        effective_limit = max_items
        if declared_total is not None:
            effective_limit = max(0, min(max_items, declared_total))

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
                if _looks_like_noise(name):
                    continue
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
            records = records[:effective_limit]
            seen = {
                (
                    i["Part Name"].strip().lower(),
                    i["Part Number"].strip().upper(),
                    i["Possible MRP"].strip().lower(),
                )
                for i in records
            }

            # Prefer official cards and avoid noisy fallback parsing when cards are available.
            output_df = pd.DataFrame(records)
            if "_query_score" in output_df.columns:
                output_df = output_df.drop(columns=["_query_score"])
            return url, output_df, ""

        # Parse table rows first (if present).
        for tr in listing_root.find_all("tr"):
            text = tr.get_text(" ", strip=True)
            if _looks_like_noise(text):
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

        # Fallback parse across likely result containers.
        if len(records) < max_items:
            for el in listing_root.find_all(["li", "article", "div"], limit=700):
                text = el.get_text(" ", strip=True)
                if not text or len(text) < 18:
                    continue
                if not re.search(r"part|mrp|rs\.?|inr|number|genuine", text, flags=re.IGNORECASE):
                    continue
                if _looks_like_noise(text):
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

        if effective_limit >= 0:
            records = records[:effective_limit]

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


def _on_search_click():
    # Capture current input for this search, then clear widget value safely via callback.
    st.session_state["last_typed_query"] = st.session_state.get("search_query_input", "").strip()
    st.session_state["search_query_input"] = ""

query = st.text_input("Enter your search query:", key="search_query_input")
image_file = st.file_uploader("Upload spare part image", type=["png", "jpg", "jpeg"])
stock_results_limit = st.slider("Stock results to show", min_value=1, max_value=20, value=7, step=1)
if image_file is not None:
    st.image(image_file, caption="Uploaded image thumbnail", width=240)

if st.button("Search", on_click=_on_search_click):
    typed_query = st.session_state.get("last_typed_query", "").strip()

    if df is None:
        st.warning("Upload inventory first.")
    elif not typed_query and not image_file:
        st.warning("Enter a text query or upload an image.")
    else:
        img_desc = ""
        image_part_name = ""
        if image_file:
            if debug_mode:
                with debug_window:
                    st.info("Analyzing image...")
            img_desc = analyze_image(image_file)
            if debug_mode:
                with debug_window:
                    st.write("**Image recognition output:**")
                    st.write(img_desc)
            image_part_name = _extract_part_name_from_image_desc(img_desc)

        # If image exists, use image-derived part first and append typed text as extra narrowing context.
        if image_part_name and typed_query:
            raw_customer_query = f"{image_part_name} {typed_query}".strip()
        elif image_part_name:
            raw_customer_query = image_part_name
        else:
            raw_customer_query = typed_query

        query_bundle = build_actionable_search_query(raw_customer_query)
        parse_error = query_bundle.get("parse_error", "")
        normalized_query = query_bundle.get("normalized_query", "")
        inventory_rag_query = query_bundle.get("inventory_query", "")
        maruti_search_query = query_bundle.get("maruti_query", "")

        if not inventory_rag_query and img_desc:
            fallback_image_query = _extract_part_name_from_image_desc(img_desc) or img_desc[:80]
            if typed_query:
                fallback_image_query = f"{fallback_image_query} {typed_query}".strip()
            query_bundle = build_actionable_search_query(fallback_image_query)
            parse_error = query_bundle.get("parse_error", "")
            normalized_query = query_bundle.get("normalized_query", "")
            inventory_rag_query = query_bundle.get("inventory_query", "")
            maruti_search_query = query_bundle.get("maruti_query", "")

        if debug_mode and (inventory_rag_query or normalized_query):
            with debug_window:
                st.caption("Understanding messy customer language and converting to inventory-searchable item name")
                st.write("Parsed Query JSON:")
                st.json(query_bundle.get("parsed_query", {}))
                if parse_error:
                    st.warning(f"Parsed query fallback used: {parse_error}")
                st.write(f"User Query -> {raw_customer_query or 'N/A'}")
                st.write(f"GPT Normalized Query -> {normalized_query or raw_customer_query or 'N/A'}")
                st.write(f"Inventory RAG Query -> {inventory_rag_query or 'N/A'}")
                st.write(f"Maruti Website Query -> {maruti_search_query or 'N/A'}")

        inventory_search_query = inventory_rag_query or normalized_query or raw_customer_query

        matches = search_inventory(df, inventory_search_query, top_n=stock_results_limit)
        if matches.empty:
            st.write("No matching items found.")
        else:
            display_df = matches[["item_name", "item_cd", "cat_name", "clsng_bal"]].head(stock_results_limit).copy()
            display_df.columns = ["Item Name", "Part Code", "Category", "Qty"]
            st.dataframe(display_df)

            # Give LLM slightly broader relevant stock context for related-part suggestions.
            explanation_context_df = build_related_inventory_context(
                df=df,
                primary_matches=matches,
                user_query=inventory_rag_query or normalized_query or inventory_search_query,
                max_items=15,
            )
            matches_text = rows_to_text(explanation_context_df)
            
            # Generate comprehensive explanation (includes Malayalam)
            explanation_input = inventory_rag_query or normalized_query or inventory_search_query or typed_query or image_part_name or "image lookup"
            explanation = mechanic_explanation_english(matches_text, explanation_input)
            
            # Display explanation as a styled webpage section
            st.subheader("Spare Parts Analysis")
            st.markdown(explanation, unsafe_allow_html=True)

        # Maruti Genuine Parts direct search using identified part name/query.
        maruti_query = (maruti_search_query or "").strip()
        if not maruti_query:
            maruti_query = _remove_year_tokens(inventory_search_query)
        if not maruti_query and image_part_name:
            maruti_query = _remove_year_tokens(image_part_name)
        if not maruti_query and not matches.empty:
            maruti_query = _remove_year_tokens(str(matches.iloc[0].get("item_name", "")).strip())

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

