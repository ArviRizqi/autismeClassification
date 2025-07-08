import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN

# =============== CONFIG ===============
st.title("Autism Detection using MobileViT + MTCNN (Face Crop)")

# Load MTCNN face detector
@st.cache_resource
def load_mtcnn():
    mtcnn = MTCNN(keep_all=False)  # keep_all=False agar hanya crop wajah terbesar
    return mtcnn

mtcnn = load_mtcnn()

# Load TorchScript model
@st.cache_resource
def load_model():
    model = torch.jit.load("mobilevit_traced.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# =============== UI ===============
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ======== Detect and Crop Face ========
    with torch.no_grad():
        face = mtcnn(image)

    if face is None:
        st.warning("No face detected. Please upload a clear frontal face image.")
    else:
        # Convert to PIL Image for visualization
        face_image = transforms.ToPILImage()(face)
        st.image(face_image, caption="Cropped Face", use_column_width=False, width=224)

        # Preprocess cropped face
        input_tensor = preprocess(face_image).unsqueeze(0)  # Add batch dimension

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item()

        # Map label (0/1) ke nama class
        labels = {0: "Non-Autistic", 1: "Autistic"}
        result = labels.get(predicted_class, "Unknown")

        # Display result
        st.markdown(f"### Prediction: **{result}** ({confidence*100:.2f}% confidence)")
