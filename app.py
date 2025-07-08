import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
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
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        face = transform(face).unsqueeze(0)

        # Predict
        outputs = model(face)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()

        labels = {0: "Non-Autistic", 1: "Autistic"}
        result = labels.get(pred, "Unknown")

        st.write(f"Prediction: **{result}** ({conf*100:.2f}%)")
    else:
        st.warning("No face detected.")
