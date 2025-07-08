import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
import timm

# ---------------------------
# ⚡️ Load MTCNN
mtcnn = MTCNN(keep_all=False, device='cpu')

# ---------------------------
# ⚡️ Load Model
@st.cache_resource
def load_model():
    model = timm.create_model('mobilevit_s', pretrained=False, num_classes=2)
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        'https://huggingface.co/Artz-03/autismeClassification/resolve/main/best_model_phase2.pt',
        map_location='cpu'
    ))
    model.eval()
    return model

model = load_model()

# ---------------------------
# ⚡️ Streamlit UI
st.title("Autism Detection from Face")

uploaded_file = st.file_uploader("Upload face image", type=['jpg','png'])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Crop face with MTCNN
    face = mtcnn(image)

    if face is not None:
        # Ensure face tensor is valid (C,H,W)
        if face.shape[0] == 3:
            # Add batch dimension (B,C,H,W)
            face = face.unsqueeze(0)

            # Resize tensor directly
            face = F.interpolate(face, size=(224,224), mode='bilinear', align_corners=False)

            # Normalize manually (mean/std ImageNet)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
            face = (face - mean) / std

            with torch.no_grad():
                outputs = model(face)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0, pred].item()

            labels = {0: "Non-Autistic", 1: "Autistic"}
            result = labels.get(pred, "Unknown")

            st.write(f"Prediction: **{result}** ({conf*100:.2f}%)")
        else:
            st.warning("Face detection failed. Please upload a clear face image.")
    else:
        # Optional: classify as Non-Autistic by default if no face detected
        st.warning("No face detected. Classified as Non-Autistic by default.")
        st.write(f"Prediction: **Non-Autistic** (100.00%)")
