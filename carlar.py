import os
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# La Tesla T4 en modo TCC solo es visible para CUDA (DirectML no la ve)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Dispositivo en uso: CUDA GPU ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("CUDA no disponible, usando CPU.")

# Cargar procesador y modelo
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)
model.eval()

# Cargar imagen
image_path = Path(__file__).resolve().parent / "muestras_venus/cheque_1.jpg"  # Ajusta el nombre
image = Image.open(image_path).convert("RGB")

# Inferencia
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

with torch.no_grad():
    generated_output = model.generate(
        pixel_values,
        return_dict_in_generate=True,
        output_scores=True,
    )

generated_ids = generated_output.sequences
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

confidence = None
scores = getattr(generated_output, "scores", None)
if scores:
    log_probs = torch.stack([torch.log_softmax(s, dim=-1) for s in scores], dim=1)
    new_tokens = generated_ids[:, 1:]
    token_scores = log_probs.gather(2, new_tokens.unsqueeze(-1)).squeeze(-1)[0]
    mask = new_tokens[0] != model.generation_config.pad_token_id
    token_scores = token_scores[mask]
    if token_scores.numel() > 0:
        confidence = float(torch.exp(token_scores.mean()).item())

print("Texto extraído:", generated_text)
if confidence is not None:
    print(f"Confianza de la extracción: {confidence:.2%}")