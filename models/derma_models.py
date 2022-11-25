## Most of these codes are taken from https://drive.google.com/drive/folders/1oQ53WH_Tp6rcLZjRp_-UBOQcMl-b1kkP and the DDI paper.

import torch
import torch.nn as nn
import os
import torchvision


# google drive paths to models
MODEL_WEB_PATHS = {
'HAM10000_INCEPTION':'https://drive.google.com/uc?id=1ToT8ifJ5lcWh8Ix19ifWlMcMz9UZXcmo',
}

# thresholds determined by maximizing F1-score on the test split of the train 
#   dataset for the given algorithm
MODEL_THRESHOLDS = {
    'HAM10000_INCEPTION':0.733,
}

def load_model(backbone_name, save_dir="./models", download=True):
    # Taken from the DDI repo https://drive.google.com/drive/folders/1oQ53WH_Tp6rcLZjRp_-UBOQcMl-b1kkP
    """Load the model and download if necessary. Saves model to provided save 
    directory."""

    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{backbone_name.lower()}.pth")
    if not os.path.exists(model_path):
        if not download:
            raise Exception("Model not downloaded and download option not"\
                            " enabled.")
        else:
            # Requires installation of gdown (pip install gdown)
            import gdown
            gdown.download(MODEL_WEB_PATHS[backbone_name], model_path)
    model = torchvision.models.inception_v3(init_weights=False, pretrained=False, transform_input=True)
    model.fc = torch.nn.Linear(2048, 2)
    model.AuxLogits.fc = torch.nn.Linear(768, 2)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model._ddi_name = backbone_name
    model._ddi_threshold = MODEL_THRESHOLDS[backbone_name]
    model._ddi_web_path = MODEL_WEB_PATHS[backbone_name]
    return model


class InceptionBottom(nn.Module):
    def __init__(self, original_model, layer="penultimate"):
        super(InceptionBottom, self).__init__()
        layer_dict = {"penultimate": -2,
                      "block_6": -4,
                      "block_5": -5,
                      "block_4": -6}
        until_layer = layer_dict[layer]
        self.layer = layer
        all_children = list(original_model.children())
        all_children.insert(-1, nn.Flatten(1))
        self.features = nn.Sequential(*all_children[:until_layer])
        self.model = original_model

    def _transform_input(self, x):
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        x = self._transform_input(x)
        x = self.model.Conv2d_1a_3x3(x)
        # N x 32 model.x 149 x 149
        x = self.model.Conv2d_2a_3x3(x)
        # N x 32 model.x 147 x 147
        x = self.model.Conv2d_2b_3x3(x)
        # N x 64 model.x 147 x 147
        x = self.model.maxpool1(x)
        # N x 64 model.x 73 x 73
        x = self.model.Conv2d_3b_1x1(x)
        # N x 80 model.x 73 x 73
        x = self.model.Conv2d_4a_3x3(x)
        # N x 192model. x 71 x 71
        x = self.model.maxpool2(x)
        # N x 192model. x 35 x 35
        x = self.model.Mixed_5b(x)
        # N x 256model. x 35 x 35
        x = self.model.Mixed_5c(x)
        # N x 288model. x 35 x 35
        x = self.model.Mixed_5d(x)
        # N x 288model. x 35 x 35
        x = self.model.Mixed_6a(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6b(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6c(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6d(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6e(x)
        # N x 768model. x 17 x 17
        # N x 768model. x 17 x 17
        x = self.model.Mixed_7a(x)
        # N x 128model.0 x 8 x 8
        x = self.model.Mixed_7b(x)
        # N x 204model.8 x 8 x 8
        x = self.model.Mixed_7c(x)
        # N x 204model.8 x 8 x 8
        # Adaptivmodel.e average pooling
        x = self.model.avgpool(x)
        # N x 204model.8 x 1 x 1
        x = self.model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        return x


class InceptionTop(nn.Module):
    def __init__(self, original_model, layer="penultimate"):
        super(InceptionTop, self).__init__()
        layer_dict = {"penultimate": -2,
                      "block_6": -4,
                      "block_5": -5,
                      "block_4": -6}
        until_layer = layer_dict[layer]
        all_children = list(original_model.children())
        all_children.insert(-1, nn.Flatten(1))
        self.layer = layer
        self.features = nn.Sequential(*all_children[until_layer:])
    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def get_derma_model(args, backbone_name="ham10000"):
    model = load_model(backbone_name.upper(), save_dir=args.out_dir)
    model = model.to("cuda")
    model = model.eval()
    model_bottom, model_top = InceptionBottom(model), InceptionTop(model)
    return model, model_bottom, model_top
