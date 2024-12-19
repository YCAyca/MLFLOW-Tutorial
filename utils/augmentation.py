import yaml
from torchvision import transforms

# Helper function to create transforms dynamically
def create_transforms(params, input_size):
    transform_list = []
    # Add base transforms
    transform_list.append(transforms.ToPILImage())
    transform_list.append(transforms.Resize(input_size))

    # Add transforms based on config
    if "resize" in params:
        transform_list.append(transforms.Resize(params["resize"]))
    
    if params.get("random_horizontal_flip", False):
        transform_list.append(transforms.RandomHorizontalFlip())
    
    if "random_rotation" in params:
        transform_list.append(transforms.RandomRotation(params["random_rotation"]))
    
    if "color_jitter" in params:
        cj_params = params["color_jitter"]
        transform_list.append(
            transforms.ColorJitter(
                brightness=cj_params.get("brightness", 0),
                contrast=cj_params.get("contrast", 0),
                saturation=cj_params.get("saturation", 0),
                hue=cj_params.get("hue", 0)
            )
        )
    
    transform_list.append(transforms.ToTensor())
    
    if "normalize" in params:
        norm_params = params["normalize"]
        transform_list.append(
            transforms.Normalize(
                mean=norm_params.get("mean", [0.0, 0.0, 0.0]),
                std=norm_params.get("std", [1.0, 1.0, 1.0])
            )
        )
    
    return transforms.Compose(transform_list)

if __name__ == "__main__":
    # Load configuration from YAML
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Create transforms for train and validation using the config
    data_transforms = {
        "train": create_transforms(config["data_transforms"]["train"], (299,299)),
        "val": create_transforms(config["data_transforms"]["val"], (299,299)),
    }

    # Verify the transforms
    print(data_transforms["train"])
    print(data_transforms["val"])
