import os
import streamlit as st
import pandas as pd
import openai

# Set OpenAI API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

st.title("AI Spare Parts Sales Assistant")

# Function to load and cache inventory from an uploaded Excel file
@st.cache_data(show_spinner=False)
def load_inventory(file) -> pd.DataFrame:
    # read the excel file into a DataFrame
    df = pd.read_excel(file)
    return df

# Function to convert DataFrame rows into plain text lines
def df_to_text(df: pd.DataFrame) -> str:
    lines = []
    for idx, row in df.iterrows():
        # build a simple description of each row
        parts = []
        # try to access common columns if they exist
        if "Part Number" in row and pd.notna(row["Part Number"]):
            parts.append(str(row["Part Number"]))
        if "Item Name" in row and pd.notna(row["Item Name"]):
            parts.append(str(row["Item Name"]))
        if "Rack" in row and pd.notna(row["Rack"]):
            parts.append(f"Rack {row['Rack']}")
        if "Quantity" in row and pd.notna(row["Quantity"]):
            parts.append(f"Qty {row['Quantity']}")
        if "Price" in row and pd.notna(row["Price"]):
            parts.append(f"Price {row['Price']}")
        # fallback: include all columns generically
        if not parts:
            parts = [f"{k} {v}" for k, v in row.items() if pd.notna(v)]
        line = " - ".join(parts)
        lines.append(line)
    return "\n".join(lines)

# upload inventory section
uploaded_file = st.file_uploader("Upload Inventory Excel", type=["xls", "xlsx"])

inventory_text = ""
if uploaded_file is not None:
    df = load_inventory(uploaded_file)
    inventory_text = df_to_text(df)
    st.success("Inventory loaded and cached in memory.")

# search input and button
query = st.text_input("Enter your search query:")
if st.button("Search"):
    if not query:
        st.warning("Please enter a search query.")
    elif not inventory_text:
        st.warning("Please upload an inventory file first.")
    else:
        # build the prompt for ChatGPT
        prompt = f"""
You are an expert automobile spare parts assistant.

Customer asked for:
{query}

Here is the shop inventory:
{inventory_text}

Tasks:

1. Identify matching items in the inventory
2. Show rack location
3. Show quantity
4. Explain the part
5. Mention compatible vehicles if known
6. Provide approximate online market price
7. Mention common OEM brands

Return structured response useful for a spare parts salesman.
"""
        try:
            with st.spinner("Contacting AI..."):
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                )
            answer = response.choices[0].message.content
            st.text_area("AI Response", answer, height=300)
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")

# end of file
