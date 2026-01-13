import torch
import torch.nn as nn
import torchvision.models as models
import json
import re
from pathlib import Path
from PIL import Image
from torchvision import transforms
from io import BytesIO
from fastapi import FastAPI
from pydantic import BaseModel
import base64

def load_model(model_registry_dir):
    registry_path = Path(model_registry_dir)

    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry directory not found: {model_registry_dir}")

    model_files = list(registry_path.glob("model_v*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No models found in registry: {model_registry_dir}")
    
    def get_version(filepath):
        match = re.search(r'model_v(\d+)\.pth', filepath.name)
        return int(match.group(1)) if match else 0
        
    latest_model = max(model_files, key=get_version)
    version = get_version(latest_model)
        
    print(f"Loading model version {version} from {latest_model}")
    
    # Load metadata
    metadata_path = registry_path / f"model_v{version}_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {metadata_path}")
    else:
        print(f"WARNING: Metadata not found at {metadata_path}")
        print("Using default metadata. Model will still work, but with default settings.")
        
        default_classes = ['barred_owl', 'bear', 'coyote', 'hummingbird', 'osprey', 'sandhill_crane']
        metadata = {
            "version": f"v{version}",
            "classes": default_classes,
            "class_to_idx": {cls: i for i, cls in enumerate(default_classes)},
            "preprocessing": {
                "input_size": [224, 224],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "model_architecture": "MobileNetV3",
            "training_metrics": {
                "note": "Metadata not found - using defaults"
            }
        }
        print(f"Using default classes: {default_classes}")

    # Load model
    model = models.mobilenet_v3_small(weights=None)
    num_classes = len(metadata['classes'])
    in_features = 576

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )

    model.load_state_dict(torch.load(latest_model, map_location='cpu'))
    # Set to evaluation mode
    model.eval() 
    
    print(f"Model loaded successfully!")
    print(f"  Classes: {metadata['classes']}")
    print(f"  Architecture: {metadata['model_architecture']}")
    
    return model, metadata


def preprocess_image(image_path_or_bytes, metadata):
    # Load image
    if isinstance(image_path_or_bytes, bytes):
        # Image data is bytes (from base64)
        image = Image.open(BytesIO(image_path_or_bytes)).convert('RGB')
    elif isinstance(image_path_or_bytes, (str, Path)):
        # Image data is file path
        image = Image.open(image_path_or_bytes).convert('RGB')
    else:
        raise ValueError("image_path_or_bytes must be bytes, str, or Path")
    
    # Get preprocessing parameters from metadata
    preprocessing = metadata['preprocessing']
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=preprocessing['mean'],
            std=preprocessing['std']
        )
    ])
    
    # Apply transform
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict(model, image_tensor, metadata):
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        
        # Get probabilities using softmax
        probs = torch.softmax(outputs, dim=1)
        
        # Get top prediction
        confidence, predicted_idx = torch.max(probs, dim=1)
        
        # Convert index to class name
        class_name = metadata['classes'][predicted_idx.item()]
        confidence_value = confidence.item()
        
        # Get all class probabilities as dictionary
        all_probs = probs[0].tolist()
        confidence_dict = {
            metadata['classes'][i]: prob 
            for i, prob in enumerate(all_probs)
        }
    
    return class_name, confidence_value, confidence_dict

app = FastAPI(title="Animal Classification Inference API")

class InferenceRequest(BaseModel):
    image_data: str

class InferenceResponse(BaseModel):
    tags: list[str]
    confidence: dict[str, float] = {}

model = None
metadata = None

@app.on_event("startup")
async def load_model_at_startup():
    """Load model when API starts."""
    global model, metadata
    try:
        script_dir = Path(__file__).parent.parent.parent 
        model_registry_path = script_dir / "model_registry"
        
        model, metadata = load_model(str(model_registry_path))
        print("Model loaded successfully at startup!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": metadata['classes'] if metadata else None
    }

@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    if model is None or metadata is None:
        return {"error": "Model not loaded", "tags": [], "confidence": {}}
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_data)
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes, metadata)
        
        # Get prediction
        class_name, confidence, confidence_dict = predict(model, image_tensor, metadata)
        
        # Return in format expected
        return InferenceResponse(
            tags=[class_name],
            confidence=confidence_dict
        )
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return InferenceResponse(
            tags=[],
            confidence={},
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
