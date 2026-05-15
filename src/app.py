
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



import streamlit as st
import pandas as pd
import os
import string

synonyms = {
    "plants": "crops",
    "crop": "crops",
    "farmers": "farmer",
    "cows": "cow",
    "goats": "goat",
    "buffaloes": "buffalo",
    "hens": "chicken",
    "seeds": "seed",
    "fertilizers": "fertilizer",
    "pesticides": "pesticide",
    "tractors": "tractor",
    "machines": "machine"
}

df = pd.read_csv("../dataset/agro_terms.csv")

st.title("🌾 AgroSign AI")

# SAFE STATE (only for clearing output)
if "show_output" not in st.session_state:
    st.session_state.show_output = True

# CLEAR FUNCTION (NO TEXT STATE MODIFICATION)
def clear_all():
    st.session_state.show_output = False

# FORM (ENTER KEY WORKS HERE)
with st.form("form"):
    text = st.text_input("Enter agriculture text:")

    col1, col2 = st.columns(2)

    submit = col1.form_submit_button("Submit")
    col2.form_submit_button("Clear", on_click=clear_all)

# CLEAR OUTPUT AREA ONLY (SAFE RESET)
if not st.session_state.show_output:
    st.session_state.show_output = True
    st.rerun()

# PROCESS INPUT
if submit and text and st.session_state.show_output:

    words = text.lower().split()

    st.subheader("Detected Signs:")

    for word in words:

        word = word.strip(string.punctuation)

        if word in synonyms:
            word = synonyms[word]

        match = df[
            (df["keyword"].str.lower() == word) |
            (df["synonyms"].str.lower().str.contains(word, na=False))
        ]

        if not match.empty:

            keyword = match.iloc[0]["keyword"]
            video_path = os.path.join("..", match.iloc[0]["video_path"])

            st.write(f"👉 {keyword}")

            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.error(f"Video not found: {video_path}")

        else:
            st.warning(f"No sign found for: {word}")