"""
CET-Net Configuration
All hyperparameters in one place.
"""


class Config:
    # ----- Model Architecture -----
    num_aus = 12  # number of Action Units to detect
    num_expressions = 7  # number of expression classes
    backbone_feat_dim = 2048  # ResNet-50 conv5_x output channels
    spatial_size = 7  # spatial resolution after backbone (7x7)
    hidden_dim = 256  # internal feature dimension for AU processing
    gnn_layers = 2  # number of belief transport iterations

    # ----- AU Indices (for reference) -----
    # 0:AU1, 1:AU2, 2:AU4, 3:AU6, 4:AU7, 5:AU10,
    # 6:AU12, 7:AU14, 8:AU15, 9:AU17, 10:AU23, 11:AU24
    au_names = [
        "AU1", "AU2", "AU4", "AU6", "AU7", "AU10",
        "AU12", "AU14", "AU15", "AU17", "AU23", "AU24",
    ]

    # ----- Expression Indices (for reference) -----
    expression_names = [
        "Happy", "Sad", "Fear", "Disgust", "Anger", "Surprise", "Neutral",
    ]

    # ----- Loss Weights -----
    lambda_au = 1.0       # AU supervision weight
    lambda_expr = 1.0     # expression supervision weight
    lambda_rule = 0.5     # rule violation penalty weight
    lambda_cf = 0.3       # counterfactual consistency weight

    # ----- Training -----
    lr = 1e-4
    backbone_lr_factor = 0.1  # backbone uses lr * this factor
    weight_decay = 1e-5
    batch_size = 32
    epochs = 30
    num_workers = 0

    # ----- Paths -----
    checkpoint_dir = "checkpoints"

    # ----- Dataset -----
    # Set to "bp4d" or "disfa"
    dataset = "disfa"

    # BP4D paths and splits
    bp4d_root = "data/BP4D"  # change to your BP4D path
    bp4d_train_subjects = [
        "F001", "F002", "F003", "F004", "F005", "F006", "F007", "F008",
        "F009", "F010", "F011", "F012", "F013", "F014", "F015", "F016",
        "F017", "F018", "F019", "F020", "F021", "F022", "F023",
        "M001", "M002", "M003", "M004", "M005", "M006", "M007", "M008",
        "M009", "M010", "M011", "M012", "M013", "M014",
    ]
    bp4d_val_subjects = [
        "M015", "M016", "M017", "M018",
    ]

    # DISFA paths and splits
    disfa_root = r"C:\Users\khang\OneDrive\Desktop\DISFA_Data"  # actual path
    disfa_intensity_threshold = 2  # AU intensity >= this is "active"
    disfa_train_subjects = [
        "SN001", "SN002", "SN003", "SN004", "SN005", "SN006",
        "SN007", "SN008", "SN009", "SN010", "SN011", "SN012",
        "SN013", "SN016", "SN017", "SN018", "SN021", "SN023",
        "SN024", "SN025", "SN026",
    ]
    disfa_val_subjects = [
        "SN027", "SN028", "SN029", "SN030", "SN031", "SN032",
    ]

    # ----- Zero-Shot Learning -----
    # Which expressions are "unseen" during training
    # By default all 7 are seen; change this for ZSL experiments
    zsl_unseen_expressions = []  # e.g., ["Fear", "Disgust"] to hold out
    zsl_temperature = 10.0  # cosine similarity scaling
