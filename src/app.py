
# # # synonyms = {

# # #     # Crops / Plants
# # #     "plant": "crops",
# # #     "plants": "crops",
# # #     "crop": "crops",
# # #     "harvest": "harvesting",
# # #     "harvesting": "harvesting",
# # #     "seedling": "seed",
# # #     "seeds": "seed",
# # #     "farming": "farm",
# # #     "agriculture": "farm",

# # #     # Water / Irrigation
# # #     "irrigation": "water",
# # #     "watering": "water",
# # #     "pump": "machine_water_pump",
# # #     "handpump": "handpump_water",

# # #     # Farmers / Farm
# # #     "farms": "farm",
# # #     "farmers": "farmer",
# # #     "cultivation": "farm",

# # #     # Machines
# # #     "tractoring": "tractor",
# # #     "plowing": "plough",
# # #     "ploughing": "plough",
# # #     "machine": "machine_water_pump",
# # #     "harvester": "harvester",

# # #     # Animals
# # #     "buffaloes": "buffalo",
# # #     "cows": "cow",
# # #     "goats": "goat",
# # #     "hens": "chicken",

# # #     # Agriculture Materials
# # #     "fertilizers": "fertilizer",
# # #     "pesticides": "pesticide",
# # #     "organic": "organic_food",

# # #     # Weather / Soil
# # #     "climate": "weather",
# # #     "rain": "weather",
# # #     "mud": "soil",
# # #     "land": "soil"
# # # }



# import streamlit as st
# import pandas as pd
# import os
# import string

# synonyms = {
#     "plants": "crops",
#     "crop": "crops",
#     "farmers": "farmer",
#     "cows": "cow",
#     "goats": "goat",
#     "buffaloes": "buffalo",
#     "hens": "chicken",
#     "seeds": "seed",
#     "fertilizers": "fertilizer",
#     "pesticides": "pesticide",
#     "tractors": "tractor",
#     "machines": "machine"
# }



import streamlit as st
import pandas as pd

synonyms = {
    "plants": "crops",
    "crop": "crops",
    "plant": "crops",
    "harvest": "harvesting",
    "farmers": "farmer",
    "cows": "cow",
    "goats": "goat",
    "buffaloes": "buffalo",
    "hens": "chicken",
    "seeds": "seed",
    "fertilizers": "fertilizer",
    "pesticides": "pesticide",
    "tractors": "tractor",
    "machines": "machine",
    "farms": "farm",
    "watering": "water",
    "irrigation": "water"
}

# Load dataset
df = pd.read_csv("dataset/agro_terms.csv")

st.title("🌾 AgroSign AI")

# Initialize session state
if "clear_trigger" not in st.session_state:
    st.session_state.clear_trigger = 0
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# Form with dynamic key that changes on clear
with st.form(key=f"form_{st.session_state.clear_trigger}"):
    text = st.text_input("Enter agriculture text:")

if text:
    words = text.lower().split()

    st.subheader("Detected Signs:")

    for word in words:   # ✅ NOW INSIDE

        # Basic normalization
        if word.endswith("ing"):
            word = word[:-3]
        elif word.endswith("s") and len(word) > 2:
            word = word[:-1]

        # Synonym mapping
        if word in synonyms:
            word = synonyms[word]

        # Try exact match first
        match = df[
            (df["keyword"].str.lower() == word) |
            (df["synonyms"].str.lower().str.split('|').apply(
                lambda x: word in [s.strip() for s in x] if isinstance(x, list) else False
            ))
        ]

        if not match.empty:
            video_path = match.iloc[0]["video_path"]
            st.write(f"👉 {word}")
            st.video(video_path)