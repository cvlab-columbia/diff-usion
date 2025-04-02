import pyrallis
from datasets import get_cls_dataset_by_name
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
from pathlib import Path
import pdb
from textual_inversion_config import ClassifierTrainConfig
import torchvision.transforms.v2 as transforms

class CLIPClassifier(torch.nn.Module):
    def __init__(self, embedding_dim=1280, num_classes=1):
        super(CLIPClassifier, self).__init__()
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", subfolder="image_processor"
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", subfolder="image_encoder"
        )
        self.linear = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            inputs = self.image_processor(
                images=x, return_tensors="pt", do_rescale=False
            ).to(x.device)
            image_embeds = self.image_encoder(**inputs).image_embeds

        output = self.linear(image_embeds)
        return output


@pyrallis.wrap()
def main(cfg: ClassifierTrainConfig):

    device_id = cfg.device

    early_stop_counter = 0

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    
    torch.manual_seed(42)
    val_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )


    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    cfg.dataset.file_list_path = None

    train_ds, val_ds = get_cls_dataset_by_name(cfg.dataset, dataset_transforms=[transform, transform])
    data_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)

    classifier = CLIPClassifier().to(device)

    class_0_embeds_list = []
    class_1_embeds_list = []

    counter = 0
    
    for batch in tqdm(data_loader):
        if len(batch) == 4:
            images, labels, mask, mask_larger = batch
        else:
            images, labels, paths = batch

        images = images.to(device)
        labels = labels.to(device)
        #pdb.set_trace()
        inputs = classifier.image_processor(
            images=images, return_tensors="pt", do_rescale=False
        ).to(device)
        #pdb.set_trace()
        with torch.no_grad():
            image_embeds = classifier.image_encoder(**inputs).image_embeds
            
            # Separate embeddings based on labels
            class_0_mask = (labels == 0)
            class_1_mask = (labels != 0)
            
            class_0_embeds_list.append(image_embeds[class_0_mask])
            class_1_embeds_list.append(image_embeds[class_1_mask])
        counter = counter + 1

    class_0_embeds = torch.cat(class_0_embeds_list)
    class_1_embeds = torch.cat(class_1_embeds_list)

    save_dir = Path(cfg.clip_image_embeds_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    #pdb.set_trace()
    torch.save(class_0_embeds, save_dir / "0_val.pt")
    torch.save(class_1_embeds, save_dir / "1_val.pt")

if __name__ == "__main__":
    main()
