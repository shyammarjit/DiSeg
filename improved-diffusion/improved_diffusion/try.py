import torch
from transformers import (
    SegformerModel,
    SegformerConfig,
    SegformerForSemanticSegmentation,
)
import torch.nn as nn
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)


# # Initializing a SegFormer nvidia/segformer-b0-finetuned-ade-512-512 style configuration
# configuration = SegformerConfig(output_hidden_states=True)
# # print(configuration)
# # exit()


# # Initializing a model from the nvidia/segformer-b0-finetuned-ade-512-512 style configuration
# model = SegformerModel(configuration)

# # Accessing the model configuration
# # configuration = model.config


# x = torch.rand(1, 3, 256, 256)

# s = model(x)

# print(x.shape)
# print(type(s))


# print(s["last_hidden_state"].shape)
# # torch.Size([1, 256, 8, 8])

# print("-----------------------------------")
# print(len(s["hidden_states"]))

# print(s["hidden_states"][0].shape)
# print(s["hidden_states"][1].shape)
# print(s["hidden_states"][2].shape)
# print(s["hidden_states"][3].shape)
# # torch.Size([1, 32, 64, 64])
# # torch.Size([1, 64, 32, 32])
# # torch.Size([1, 160, 16, 16])
# # torch.Size([1, 256, 8, 8])

# # print(model.config)


# configuration = SegformerConfig(
#     id2label={
#         "0": "wall",
#         "1": "building",
#         "2": "sky",
#         "3": "floor",
#     },
#     label2id={
#         "wall": 0,
#         "building": 1,
#         "sky": 2,
#         "floor": 3,
#     },
# )

# model = SegformerForSemanticSegmentation(configuration)
# config = model.config

# # print(config)
# # x = torch.rand(1, 3, 256, 256)

# # s = model(x)

# # print(x.shape)
# # print(s["logits"].shape)

# model1 = SegformerForSemanticSegmentation.from_pretrained(
#     "nvidia/segformer-b0-finetuned-ade-512-512"
# )
# config1 = model1.config

# # print(config1)

# x = torch.rand(1, 3, 224, 224)

# s = model(x)
# s1 = model1(x)


# print(s["logits"].shape)
# # torch.Size([1, 4, 56, 56])

# print(s1["logits"].shape)
# # torch.Size([1, 150, 56, 56])


nn.Sequential(
    linear(model_channels, time_embed_dim),
    SiLU(),
    linear(time_embed_dim, time_embed_dim),
)
