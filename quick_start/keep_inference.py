from transformers import PretrainedConfig, PreTrainedModel, BertModel, AutoConfig, AutoModel, BertConfig, AutoTokenizer
import timm
import torch.nn as nn
import torch
import numpy
from torchvision import transforms
from PIL import Image

class KEEPConfig(PretrainedConfig):
    model_type = "keep"  # 

    def __init__(
        self,
        vision_config=None,  # Vision Encoder
        text_config=None,    # Text Encoder
        projection_dim=768,  
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config
        self.text_config = text_config
        self.projection_dim = projection_dim
        

class KEEPModel(PreTrainedModel):
    config_class = KEEPConfig  # 

    def __init__(self, config):
        super().__init__(config)

        # Vision Encoder 
        self.visual = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )

        self.visual_head = nn.Sequential(
                    nn.Linear(self.visual.num_features, config.projection_dim),
                    nn.GELU(),
                    nn.Linear(config.projection_dim, config.projection_dim)
                )

        # Text Encoder
        text_config =  BertConfig(**config.text_config)
        self.text = BertModel(text_config)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * numpy.log(1 / 0.04))

    def encode_image(self, image_inputs):
        vision_features = self.visual(image_inputs)  # [batch_size, vision_dim]
        vision_features =  torch.nn.functional.normalize(self.visual_head(vision_features), dim=-1)  # [batch_size, projection_dim]
        
        return vision_features
    
    def encode_text(self, text_inputs):
        text_features = torch.nn.functional.normalize(self.text(**text_inputs).pooler_output, dim=-1)  # [batch_size, text_dim]
        return text_features
    
    
    def forward(self, image_inputs, text_inputs):
        vision_features = self.encode_image(image_inputs)
        
        text_features = self.encode_text(text_inputs)

        return {
            "vision_features": vision_features,  
            "text_features": text_features       
        }

AutoConfig.register("keep", KEEPConfig)
AutoModel.register(KEEPConfig, KEEPModel)


model_path = '../KEEP_release/'
config = AutoConfig.from_pretrained(model_path + 'config.json')
model = AutoModel.from_config(config)
state_dict = torch.load(model_path + 'pytorch_model.bin', map_location='cpu')
model.load_state_dict(state_dict,strict=True)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True, local_files_only=True)
transform = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

example_image_path = './example.tif'
example_text = ['an H&E image of breast invasive carcinoma.', 'an H&E image of normal tissue.', 'an H&E image of lung adenocarcinoma.']

img_input =  transform(Image.open(example_image_path).convert('RGB')).unsqueeze(0)
token_input = tokenizer(example_text,max_length=256,padding='max_length',truncation=True, return_tensors='pt')

img_feature = model.encode_image(img_input)
text_feature = model.encode_text(token_input)

print(img_feature@text_feature.T)
