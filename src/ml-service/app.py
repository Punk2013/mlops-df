from .pipeline import IncrementalTrainingPipeline
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torch

if __name__ == "__main__":
  # pipeline = IncrementalTrainingPipeline()
  # pipeline.train_person("Tom Hanks", "data/Tom Hanks")

  # pipeline.train_person("Nicole Kidman", "data/Nicole Kidman")

  pipeline = IncrementalTrainingPipeline()
  # Load trained models for this person
  model = pipeline.load_person_models("Tom Hanks", encoder_path=pipeline._get_latest_encoder_path())
  model.encoder.eval()
  model.decoder.eval()

  # Load and preprocess source image
  transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])

  source_img = Image.open("data/Nicole Kidman/033_8bed6926.jpg").convert('RGB')
  source_tensor = transform(source_img).unsqueeze(0).to(model.device)

  # Encode and reconstruct with person's decoder
  with torch.no_grad():
    encoded = model.encoder(source_tensor)
    reconstructed = model.decoder(encoded)

  # Convert back to image
  def denormalize(tensor):
    return tensor * 0.5 + 0.5

  output_img = transforms.ToPILImage()(denormalize(reconstructed[0].cpu()))
  output_path = "nicole-hanks.jpg"
  output_img.save(output_path)

  print(f"Image saved to {output_path}")
