import streamlit as st
import torch
import numpy as np
from PIL import Image
from UNet import UNet
from deeplab import DeepLabV3Plus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet = UNet(in_channels=3, out_channels=1, depth=4).to(device)
unet_ckpt = torch.load('unet.pt', map_location=device)
unet.load_state_dict(unet_ckpt['state_dict'])
unet.eval()

deeplab = DeepLabV3Plus(in_channels=3, out_channels=1).to(device)
deeplab_ckpt = torch.load('deeplab.pt', map_location=device)
deeplab.load_state_dict(deeplab_ckpt['state_dict'])
deeplab.eval()

st.title("Brain MRI Segmentation")
st.write(
    "This application demonstrates brain tumor segmentation using two state-of-the-art models: **UNet** and **DeepLab v3+**. "
    "Upload a brain MRI image, and the application will display segmentation results."
)

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "tif", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    img_resized = img.resize((256, 256))
    img_array = np.array(img_resized)
    img_normalized = img_array / 255.0
    img_input = torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        unet_pred = unet(img_input)
        deeplab_pred = deeplab(img_input)

    unet_pred_binary = (unet_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    deeplab_pred_binary = (deeplab_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    unet_overlay = np.array(img_resized).copy()
    unet_overlay[unet_pred_binary == 1] = [0, 255, 0]

    deeplab_overlay = np.array(img_resized).copy()
    deeplab_overlay[deeplab_pred_binary == 1] = [0, 255, 0]

    st.header("Segmentation Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("UNet")
        st.image(unet_pred.squeeze().cpu().numpy(), caption="UNet Raw Prediction", use_column_width=True, clamp=True)
        st.image(unet_overlay, caption="UNet Overlay", use_column_width=True)

    with col2:
        st.subheader("DeepLab v3+")
        st.image(deeplab_pred.squeeze().cpu().numpy(), caption="DeepLab v3+ Raw Prediction", use_column_width=True, clamp=True)
        st.image(deeplab_overlay, caption="DeepLab v3+ Overlay", use_column_width=True)

    st.markdown("---")
