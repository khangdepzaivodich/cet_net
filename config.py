"""
MuscleAU-Net Configuration
All hyperparameters in one place.
"""


class Config:
    # ----- Model Architecture -----
    # Backbone
    backbone = "dinov2_vitb14"  # options: "dinov2_vitb14", "vit_b_16", "swin_b"
    backbone_dim = 768          # ViT-B output dimension
    input_size = 256            # input image size (256x256)
    patch_size = 14             # DINOv2 uses 14x14 patches
    num_patches = 256           # (256/14)^2 ≈ 332, but DINOv2 uses 518→37x37=1369... we'll handle dynamically

    # Feature dimensions
    hidden_dim = 256            # common projection dimension for multi-modal repo
    texture_dim = 128           # ROI texture CNN output dimension
    num_rois = 7                # number of ROI regions

    # Muscles
    num_muscles = 18            # number of facial muscles
    num_aus = 12                # number of Action Units (DISFA: 12)

    # Geometry
    num_geometry_predicates = 8
    num_texture_predicates = 5

    # Graph
    gat_heads = 4               # number of attention heads in GAT
    gat_layers = 2              # number of GAT layers

    # Cross-attention
    cross_attn_heads = 8
    cross_attn_layers = 2

    # ----- AU Indices (DISFA) -----
    au_indices = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
    au_names = [
        "AU1", "AU2", "AU4", "AU5", "AU6", "AU9",
        "AU12", "AU15", "AU17", "AU20", "AU25", "AU26",
    ]

    # ----- Loss Weights -----
    lambda_au = 1.0
    lambda_logic = 0.5
    lambda_counterfactual = 0.3
    lambda_graph = 0.3
    lambda_attention = 0.1

    # ----- Training -----
    lr = 1e-4
    backbone_lr_factor = 0.1
    weight_decay = 1e-5
    batch_size = 8              # small for RTX 3050 (4GB)
    epochs = 30
    num_workers = 0             # 0 for Windows compatibility

    # ----- Paths -----
    checkpoint_dir = "checkpoints"

    # ----- Dataset -----
    dataset = "disfa"
    disfa_root = r"C:\Users\khang\OneDrive\Desktop\DISFA_Data"
    disfa_intensity_threshold = 2
    disfa_train_subjects = [
        "SN001", "SN002", "SN003", "SN004", "SN005", "SN006",
        "SN007", "SN008", "SN009", "SN010", "SN011", "SN012",
        "SN013", "SN016", "SN017", "SN018", "SN021", "SN023",
        "SN024", "SN025", "SN026",
    ]
    disfa_val_subjects = [
        "SN027", "SN028", "SN029", "SN030", "SN031", "SN032",
    ]

    bp4d_root = "data/BP4D"
    bp4d_train_subjects = []
    bp4d_val_subjects = []
