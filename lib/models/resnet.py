import torchvision.models as models
import torch


def load_config(yaml_file):
    """
    @description  :加载配置文件
    ---------
    @yaml_file  :配置文件存放路径
    -------
    @Returns  :
    -------
    """
    import yaml
    
    with open(yaml_file, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def debug_model():

    from torchsummary import summary
    cfg = load_config("/home/yangpeng/Subject/defocus/OIDMENet/config/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50()

    model = model.to(device)

    summary(model, input_size=(3, 256, 256), device='cuda')
if __name__ == "__main__":
    debug_model()