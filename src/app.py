
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

# Initialize session state
if "clear_trigger" not in st.session_state:
    st.session_state.clear_trigger = 0
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# Form with dynamic key that changes on clear
with st.form(key=f"form_{st.session_state.clear_trigger}"):
    text = st.text_input("Enter agriculture text:")

    col1, col2 = st.columns(2)

    submit = col1.form_submit_button("Submit")
    clear = col2.form_submit_button("Clear")

# Handle clear button
if clear:
    st.session_state.clear_trigger += 1
    st.session_state.last_text = ""
    st.rerun()

# Process input on submit
if submit and text:
    st.session_state.last_text = text
    
    words = text.lower().split()

    st.subheader("Detected Signs:")

    for word in words:

        word = word.strip(string.punctuation)

        if word in synonyms:
            word = synonyms[word]

        # FIXED: Use exact word matching with word boundaries
        match = df[
            (df["keyword"].str.lower() == word) |
            (df["synonyms"].str.lower().str.split('|').apply(
                lambda x: word in [s.strip() for s in x] if isinstance(x, list) else False
            ))
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