from transformers import SegformerForSemanticSegmentation

def get_segformer_model(num_classes=6):  # Adjust based on classes
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    return model

if __name__ == "__main__":
    model = get_segformer_model()
    print(f"Model loaded: {model.__class__.__name__}")