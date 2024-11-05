import streamlit as st
import torch
from torchvision.utils import make_grid, save_image
from io import BytesIO
from model import Generator  # Import your model classes

# Load the model
@st.cache(allow_output_mutation=True)
def load_generator():
    latent_dim = 100
    device = torch.device('cpu')  # Use 'cuda' if you have a GPU
    generator = Generator(latent_dim=latent_dim, channels=3).to(device)
    checkpoint = torch.load("checkpoint_epoch_450.pt", map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    return generator

generator = load_generator()

# Streamlit app interface
st.title("GAN Image Generator")
st.write("Click the button to generate a 8x8 grid of images.")

if st.button("Generate 8x8 Image Grid"):
    num_images = 64  # 8x8 grid
    latent_dim = 100
    noise = torch.randn(num_images, latent_dim, 1, 1)  # Generate random noise for 64 images

    with torch.no_grad():
        fake_images = generator(noise)
        fake_images = (fake_images + 1) / 2  # Normalize to [0, 1]

    # Create a grid of images
    grid_image = make_grid(fake_images, nrow=8, padding=2, normalize=True)

    # Save grid image to a BytesIO buffer
    buffer = BytesIO()
    save_image(grid_image, buffer, format="PNG")
    buffer.seek(0)

    # Display image in Streamlit
    st.image(buffer, caption="Generated 8x8 Image Grid", use_column_width=True)
