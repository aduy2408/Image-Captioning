import streamlit as st
from PIL import Image
from model.image_captioning_models import test_blip2, test_blip, test_git_model, test_vit_gpt2, get_device
from deep_translator import GoogleTranslator

def translate_to_vietnamese(text):
    try:
        translator = GoogleTranslator(source='en', target='vi')
        return translator.translate(text)
    except Exception as e:
        return f"Translation error: {str(e)}"

# Set page config
st.set_page_config(
    page_title="Image Captioning Prototype",
    layout="centered"
)

# Title and description
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1>Image Captioning</h1>
    <p style="font-size: 18px; color: #666;">Upload an image and let the model describes!</p>
</div>
""", unsafe_allow_html=True)



# Model mapping
MODELS = {
    "BLIP": test_blip,
    "BLIP-2": test_blip2,
    "GIT": test_git_model,
    "ViT-GPT2": test_vit_gpt2
}

# Main content area - centered layout
st.markdown("<br>", unsafe_allow_html=True)

# Large image upload box with custom styling
st.markdown("""
<style>
.uploadedFile {
    padding: 2rem !important;
}
div[data-testid="stFileUploader"] > div {
    padding: 3rem 2rem !important;
    border: 3px dashed #cccccc !important;
    border-radius: 15px !important;
    background-color: #f8f9fa !important;
    text-align: center !important;
}
div[data-testid="stFileUploader"] label {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an image to generate caption",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Drag and drop or click to upload an image file"
)

# Display uploaded image if available
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Center the image 
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_container_width=True)

# Model selection and generate button in the same row
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    selected_model = st.selectbox(
        "Select Model:",
        list(MODELS.keys()),
        index=0,
        help="Choose the AI model for caption generation"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  
    generate_button = st.button("Generate Caption", type="primary", use_container_width=True, disabled=(uploaded_file is None))

st.markdown("<br>", unsafe_allow_html=True)

if generate_button and uploaded_file is not None:
    with st.spinner(f"Generating caption using {selected_model}..."):
        try:
            caption_function = MODELS[selected_model]
            caption = caption_function(image)

            # Generate Vietnamese translation
            with st.spinner("Translating to Vietnamese..."):
                vietnamese_caption = translate_to_vietnamese(caption)

            st.markdown("### Generated Caption:")

            # English caption
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #4CAF50;
                margin: 10px 0;
            ">
                <p style="font-size: 16px; margin: 0; font-weight: 600; color: #2E7D32;">🇺🇸 English:</p>
                <p style="font-size: 18px; margin: 5px 0 0 0; font-style: italic;">"{caption}"</p>
            </div>
            """, unsafe_allow_html=True)

            # Vietnamese translation
            st.markdown(f"""
            <div style="
                background-color: #e8f5e8;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #2196F3;
                margin: 10px 0;
            ">
                <p style="font-size: 16px; margin: 0; font-weight: 600; color: #1565C0;">🇻🇳 Tiếng Việt:</p>
                <p style="font-size: 18px; margin: 5px 0 0 0; font-style: italic; color: #000000; font-weight: 500;">"{vietnamese_caption}"</p>
            </div>
            """, unsafe_allow_html=True)

            device = get_device()
            st.success(f"Caption generated and translated successfully using {device.upper()}!")

        except Exception as e:
            st.error(f"Error generating caption: {str(e)}")
            st.error("Please try again or select a different model.")

