import torch
from transformers import (
    SegformerModel,
    SegformerConfig,
    SegformerForSemanticSegmentation,
)
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import argparse


# # Initializing a SegFormer nvidia/segformer-b0-finetuned-ade-512-512 style configuration
# configuration = SegformerConfig(output_hidden_states=True)
# # print(configuration)
# # exit()


# # SegformerConfig = {
# #     "attention_probs_dropout_prob": 0.0,
# #     "classifier_dropout_prob": 0.1,
# #     "decoder_hidden_size": 256,
# #     "depths": [2, 2, 2, 2],
# #     "drop_path_rate": 0.1,
# #     "hidden_act": "gelu",
# #     "hidden_dropout_prob": 0.0,
# #     "hidden_sizes": [32, 64, 160, 256],
# #     "initializer_range": 0.02,
# #     "layer_norm_eps": 1e-06,
# #     "mlp_ratios": [4, 4, 4, 4],
# #     "model_type": "segformer",
# #     "num_attention_heads": [1, 2, 5, 8],
# #     "num_channels": 3,
# #     "num_encoder_blocks": 4,
# #     "output_hidden_states": true,
# #     "patch_sizes": [7, 3, 3, 3],
# #     "reshape_last_stage": true,
# #     "semantic_loss_ignore_index": 255,
# #     "sr_ratios": [8, 4, 2, 1],
# #     "strides": [4, 2, 2, 2],
# #     "transformers_version": "4.26.1",
# # }

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

# # SegformerConfig = {
# #     # "attention_probs_dropout_prob": 0.0,
# #     # "classifier_dropout_prob": 0.1,
# #     # "decoder_hidden_size": 256,
# #     # "depths": [2, 2, 2, 2],
# #     # "drop_path_rate": 0.1,
# #     # "hidden_act": "gelu",
# #     # "hidden_dropout_prob": 0.0,
# #     # "hidden_sizes": [32, 64, 160, 256],
# #     # "initializer_range": 0.02,
# #     # "layer_norm_eps": 1e-06,
# #     # "mlp_ratios": [4, 4, 4, 4],
# #     # "model_type": "segformer",
# #     # "num_attention_heads": [1, 2, 5, 8],
# #     # "num_channels": 3,
# #     # "num_encoder_blocks": 4,
# #     # "patch_sizes": [7, 3, 3, 3],
# #     # "reshape_last_stage": true,
# #     # "semantic_loss_ignore_index": 255,
# #     # "sr_ratios": [8, 4, 2, 1],
# #     # "strides": [4, 2, 2, 2],
# #     # "transformers_version": "4.26.1",
# # }

# # SegformerConfig1 = {
# #     "_name_or_path": "nvidia/segformer-b0-finetuned-ade-512-512",
# #     "architectures": ["SegformerForSemanticSegmentation"],
# #     # "attention_probs_dropout_prob": 0.0,
# #     # "classifier_dropout_prob": 0.1,
# #     # "decoder_hidden_size": 256,
# #     # "depths": [2, 2, 2, 2],
# #     "downsampling_rates": [1, 4, 8, 16],
# #     # "drop_path_rate": 0.1,
# #     # "hidden_act": "gelu",
# #     # "hidden_dropout_prob": 0.0,
# #     # "hidden_sizes": [32, 64, 160, 256],
# #     "id2label": {
# #         "0": "wall",
# #         "1": "building",
# #         "2": "sky",
# #         "3": "floor",
# #         "4": "tree",
# #         "5": "ceiling",
# #         "6": "road",
# #         "7": "bed ",
# #         "8": "windowpane",
# #         "9": "grass",
# #         "10": "cabinet",
# #         "11": "sidewalk",
# #         "12": "person",
# #         "13": "earth",
# #         "14": "door",
# #         "15": "table",
# #         "16": "mountain",
# #         "17": "plant",
# #         "18": "curtain",
# #         "19": "chair",
# #         "20": "car",
# #         "21": "water",
# #         "22": "painting",
# #         "23": "sofa",
# #         "24": "shelf",
# #         "25": "house",
# #         "26": "sea",
# #         "27": "mirror",
# #         "28": "rug",
# #         "29": "field",
# #         "30": "armchair",
# #         "31": "seat",
# #         "32": "fence",
# #         "33": "desk",
# #         "34": "rock",
# #         "35": "wardrobe",
# #         "36": "lamp",
# #         "37": "bathtub",
# #         "38": "railing",
# #         "39": "cushion",
# #         "40": "base",
# #         "41": "box",
# #         "42": "column",
# #         "43": "signboard",
# #         "44": "chest of drawers",
# #         "45": "counter",
# #         "46": "sand",
# #         "47": "sink",
# #         "48": "skyscraper",
# #         "49": "fireplace",
# #         "50": "refrigerator",
# #         "51": "grandstand",
# #         "52": "path",
# #         "53": "stairs",
# #         "54": "runway",
# #         "55": "case",
# #         "56": "pool table",
# #         "57": "pillow",
# #         "58": "screen door",
# #         "59": "stairway",
# #         "60": "river",
# #         "61": "bridge",
# #         "62": "bookcase",
# #         "63": "blind",
# #         "64": "coffee table",
# #         "65": "toilet",
# #         "66": "flower",
# #         "67": "book",
# #         "68": "hill",
# #         "69": "bench",
# #         "70": "countertop",
# #         "71": "stove",
# #         "72": "palm",
# #         "73": "kitchen island",
# #         "74": "computer",
# #         "75": "swivel chair",
# #         "76": "boat",
# #         "77": "bar",
# #         "78": "arcade machine",
# #         "79": "hovel",
# #         "80": "bus",
# #         "81": "towel",
# #         "82": "light",
# #         "83": "truck",
# #         "84": "tower",
# #         "85": "chandelier",
# #         "86": "awning",
# #         "87": "streetlight",
# #         "88": "booth",
# #         "89": "television receiver",
# #         "90": "airplane",
# #         "91": "dirt track",
# #         "92": "apparel",
# #         "93": "pole",
# #         "94": "land",
# #         "95": "bannister",
# #         "96": "escalator",
# #         "97": "ottoman",
# #         "98": "bottle",
# #         "99": "buffet",
# #         "100": "poster",
# #         "101": "stage",
# #         "102": "van",
# #         "103": "ship",
# #         "104": "fountain",
# #         "105": "conveyer belt",
# #         "106": "canopy",
# #         "107": "washer",
# #         "108": "plaything",
# #         "109": "swimming pool",
# #         "110": "stool",
# #         "111": "barrel",
# #         "112": "basket",
# #         "113": "waterfall",
# #         "114": "tent",
# #         "115": "bag",
# #         "116": "minibike",
# #         "117": "cradle",
# #         "118": "oven",
# #         "119": "ball",
# #         "120": "food",
# #         "121": "step",
# #         "122": "tank",
# #         "123": "trade name",
# #         "124": "microwave",
# #         "125": "pot",
# #         "126": "animal",
# #         "127": "bicycle",
# #         "128": "lake",
# #         "129": "dishwasher",
# #         "130": "screen",
# #         "131": "blanket",
# #         "132": "sculpture",
# #         "133": "hood",
# #         "134": "sconce",
# #         "135": "vase",
# #         "136": "traffic light",
# #         "137": "tray",
# #         "138": "ashcan",
# #         "139": "fan",
# #         "140": "pier",
# #         "141": "crt screen",
# #         "142": "plate",
# #         "143": "monitor",
# #         "144": "bulletin board",
# #         "145": "shower",
# #         "146": "radiator",
# #         "147": "glass",
# #         "148": "clock",
# #         "149": "flag",
# #     },
# #     "image_size": 224,
# #     # "initializer_range": 0.02,
# #     "label2id": {
# #         "airplane": 90,
# #         "animal": 126,
# #         "apparel": 92,
# #         "arcade machine": 78,
# #         "armchair": 30,
# #         "ashcan": 138,
# #         "awning": 86,
# #         "bag": 115,
# #         "ball": 119,
# #         "bannister": 95,
# #         "bar": 77,
# #         "barrel": 111,
# #         "base": 40,
# #         "basket": 112,
# #         "bathtub": 37,
# #         "bed ": 7,
# #         "bench": 69,
# #         "bicycle": 127,
# #         "blanket": 131,
# #         "blind": 63,
# #         "boat": 76,
# #         "book": 67,
# #         "bookcase": 62,
# #         "booth": 88,
# #         "bottle": 98,
# #         "box": 41,
# #         "bridge": 61,
# #         "buffet": 99,
# #         "building": 1,
# #         "bulletin board": 144,
# #         "bus": 80,
# #         "cabinet": 10,
# #         "canopy": 106,
# #         "car": 20,
# #         "case": 55,
# #         "ceiling": 5,
# #         "chair": 19,
# #         "chandelier": 85,
# #         "chest of drawers": 44,
# #         "clock": 148,
# #         "coffee table": 64,
# #         "column": 42,
# #         "computer": 74,
# #         "conveyer belt": 105,
# #         "counter": 45,
# #         "countertop": 70,
# #         "cradle": 117,
# #         "crt screen": 141,
# #         "curtain": 18,
# #         "cushion": 39,
# #         "desk": 33,
# #         "dirt track": 91,
# #         "dishwasher": 129,
# #         "door": 14,
# #         "earth": 13,
# #         "escalator": 96,
# #         "fan": 139,
# #         "fence": 32,
# #         "field": 29,
# #         "fireplace": 49,
# #         "flag": 149,
# #         "floor": 3,
# #         "flower": 66,
# #         "food": 120,
# #         "fountain": 104,
# #         "glass": 147,
# #         "grandstand": 51,
# #         "grass": 9,
# #         "hill": 68,
# #         "hood": 133,
# #         "house": 25,
# #         "hovel": 79,
# #         "kitchen island": 73,
# #         "lake": 128,
# #         "lamp": 36,
# #         "land": 94,
# #         "light": 82,
# #         "microwave": 124,
# #         "minibike": 116,
# #         "mirror": 27,
# #         "monitor": 143,
# #         "mountain": 16,
# #         "ottoman": 97,
# #         "oven": 118,
# #         "painting": 22,
# #         "palm": 72,
# #         "path": 52,
# #         "person": 12,
# #         "pier": 140,
# #         "pillow": 57,
# #         "plant": 17,
# #         "plate": 142,
# #         "plaything": 108,
# #         "pole": 93,
# #         "pool table": 56,
# #         "poster": 100,
# #         "pot": 125,
# #         "radiator": 146,
# #         "railing": 38,
# #         "refrigerator": 50,
# #         "river": 60,
# #         "road": 6,
# #         "rock": 34,
# #         "rug": 28,
# #         "runway": 54,
# #         "sand": 46,
# #         "sconce": 134,
# #         "screen": 130,
# #         "screen door": 58,
# #         "sculpture": 132,
# #         "sea": 26,
# #         "seat": 31,
# #         "shelf": 24,
# #         "ship": 103,
# #         "shower": 145,
# #         "sidewalk": 11,
# #         "signboard": 43,
# #         "sink": 47,
# #         "sky": 2,
# #         "skyscraper": 48,
# #         "sofa": 23,
# #         "stage": 101,
# #         "stairs": 53,
# #         "stairway": 59,
# #         "step": 121,
# #         "stool": 110,
# #         "stove": 71,
# #         "streetlight": 87,
# #         "swimming pool": 109,
# #         "swivel chair": 75,
# #         "table": 15,
# #         "tank": 122,
# #         "television receiver": 89,
# #         "tent": 114,
# #         "toilet": 65,
# #         "towel": 81,
# #         "tower": 84,
# #         "trade name": 123,
# #         "traffic light": 136,
# #         "tray": 137,
# #         "tree": 4,
# #         "truck": 83,
# #         "van": 102,
# #         "vase": 135,
# #         "wall": 0,
# #         "wardrobe": 35,
# #         "washer": 107,
# #         "water": 21,
# #         "waterfall": 113,
# #         "windowpane": 8,
# #     },
# #     # "layer_norm_eps": 1e-06,
# #     # "mlp_ratios": [4, 4, 4, 4],
# #     # "model_type": "segformer",
# #     # "num_attention_heads": [1, 2, 5, 8],
# #     # "num_channels": 3,
# #     # "num_encoder_blocks": 4,
# #     # "patch_sizes": [7, 3, 3, 3],
# #     # "reshape_last_stage": true,
# #     # "semantic_loss_ignore_index": 255,
# #     # "sr_ratios": [8, 4, 2, 1],
# #     # "strides": [4, 2, 2, 2],
# #     "torch_dtype": "float32",
# #     # "transformers_version": "4.26.1",
# # }

# x = torch.rand(1, 3, 224, 224)

# s = model(x)
# s1 = model1(x)


# print(s["logits"].shape)
# # torch.Size([1, 4, 56, 56])

# print(s1["logits"].shape)
# # torch.Size([1, 150, 56, 56])
def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


args = create_argparser().parse_args()

model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)

# print(model)

configuration = SegformerConfig(
    id2label={
        "0": "zero",
        "1": "one",
        "2": "two",
    },
    label2id={
        "zero": 0,
        "one": 1,
        "two": 2,
    },
    image_size=256,
)

model = SegformerForSemanticSegmentation(configuration)


x = torch.rand(2, 3, 256, 256)
t = torch.rand(2)

# ou = model(x, t)
# print(type(ou))
# print(ou.shape)
# # <class 'torch.Tensor'>
# # torch.Size([2, 3, 256, 256])

ou = model(x)
print(type(ou))
print(type(ou["logits"]))
print(ou["logits"].shape)
