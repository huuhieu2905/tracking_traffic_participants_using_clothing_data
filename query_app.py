import streamlit as st
import os
import base64
import time

from PIL import Image


# 📁 Đường dẫn thư mục log và ảnh
LOG_DIR = "logs"
IMAGE_DIR = "results_images"
CLASSES_CLOTHES = {'shirt': 0, 't-shirt': 1, 'sweater': 2,
                   'coat': 3, 'dress': 4}


def load_log_file(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


def classify_by_choice(image_paths):
    choice_dict = {clothes: [] for clothes in CLASSES_CLOTHES.keys()}
    for path in image_paths:
        for clothes in CLASSES_CLOTHES.keys():
            if path.split("_")[-2] == clothes:
                path = path + ".jpg"
                choice_dict[clothes].append(path)

    return choice_dict


def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def query_app():
    # ------------------ Streamlit UI ---------------------
    st.title("🗂️ Query Log")

    # 1. Select file log
    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".txt")]
    start = time.time()
    selected_log = st.selectbox("📄 Select a log file", log_files)

    if selected_log:
        selected_log_name = selected_log.split(".")[0]
        log_path = os.path.join(LOG_DIR, selected_log)

        
        image_paths = load_log_file(log_path)
        classified_images = classify_by_choice(image_paths)

        # 2. Select clothes
        choice = st.selectbox("🔢 Choose clothes", [
            "shirt", "t-shirt", "sweater", "coat", "dress"])

        # 3. Show list images

        selected_images = classified_images.get(choice, [])
        end = time.time() - start
        print("Time:", end)
        images_per_page = 5
        total_pages = (len(selected_images) - 1) // images_per_page + 1

        if "page" not in st.session_state:
            st.session_state.page = 1

        col1, col2, col3 = st.columns([2, 6, 2])

        with col1:
            if st.button("⬅️ Previous") and st.session_state.page > 1:
                st.session_state.page -= 1

        with col3:
            if st.button("Next ➡️") and st.session_state.page < total_pages:
                st.session_state.page += 1

        st.markdown(f"### Page {st.session_state.page}/{total_pages}")

        start_idx = (st.session_state.page - 1) * images_per_page
        end_idx = start_idx + images_per_page

        if selected_images:
            st.markdown(f"### 🖼️ Images for {choice}")
            for img_rel_path in selected_images[start_idx:end_idx]:
                # Image path
                img_path = os.path.join(
                    f"{IMAGE_DIR}/{selected_log_name}", img_rel_path)
                col1, col2 = st.columns([5, 1])
                with col2:
                    st.image(img_path, width=60)

                with col1:
                    with st.expander(f"📷 {img_rel_path}"):
                        # st.image(img_path, width=300,
                        #          use_container_width=True, clamp=True)
                        st.markdown(
                            f'<img src="data:image/jpeg;base64,{base64.b64encode(open(img_path, "rb").read()).decode()}" style="max-height:300px;">',
                            unsafe_allow_html=True,
                        )

        else:
            st.info("No image for this choice.")


if __name__ == "__main__":
    query_app()
