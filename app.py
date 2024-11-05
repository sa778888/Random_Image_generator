import streamlit as st
import torch
from torchvision.utils import save_image
from io import BytesIO
from model import Generator  # Import your model classes

# Load the model
@st.cache(allow_output_mutation=True)
def load_generator():
    latent_dim = 100
    device = torch.device('cpu')  # Use 'cuda' if you have a GPU
    generator = Generator(latent_dim=latent_dim, channels=3).to(device)
    checkpoint = torch.load("./checkpoint_epoch_450.pt", map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    return generator

generator = load_generator()

# Streamlit app interface
st.title("GAN Image Generator")
st.write("Click the button to generate a new image.")

if st.button("Generate Image"):
    noise = torch.randn(1, 100, 1, 1)  # Generate random noise
    with torch.no_grad():
        fake_image = generator(noise)
        fake_image = (fake_image + 1) / 2  # Normalize to [0, 1]

    # Save image to a BytesIO buffer
    buffer = BytesIO()
    save_image(fake_image, buffer, format="PNG", normalize=True)
    buffer.seek(0)

    # Display image in Streamlit
    st.image(buffer, caption="Generated Image", use_column_width=True)
