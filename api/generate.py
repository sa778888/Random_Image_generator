from fastapi import FastAPI, Response
import torch
from io import BytesIO
from model import Generator  # Assume your model is in model.py
from torchvision.utils import save_image

app = FastAPI()

# Parameters for model and image generation
latent_dim = 100
device = 'cpu'  # Use CPU for Vercel serverless functions

# Load the model in the function so it's loaded on each request (not ideal, but works for Vercel)
def load_generator():
    checkpoint_path = '../checkpoint_epoch_450.pt'
    generator = Generator(latent_dim=latent_dim, channels=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
    generator.eval()
    return generator

@app.get("/generate")
async def generate_image():
    generator = load_generator()
    noise = torch.randn(1, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        fake_image = generator(noise)
        fake_image = (fake_image + 1) / 2  # Normalize to [0, 1]

    buffer = BytesIO()
    save_image(fake_image, buffer, format="PNG", normalize=True)
    buffer.seek(0)
    
    return Response(content=buffer.getvalue(), media_type="image/png")
