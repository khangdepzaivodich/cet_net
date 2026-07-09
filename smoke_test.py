"""
MuscleAU-Net Smoke Test

Verifies that all modules produce correct tensor shapes
and that gradients flow through the entire pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_backbone():
    print("\n--- Stage 1: ViT Backbone ---")
    from models.backbone import ViTBackbone

    # Use a smaller backbone for testing if DINOv2 isn't available
    try:
        backbone = ViTBackbone("dinov2_vitb14").to(device)
        print("  Loaded: DINOv2 ViT-B/14")
    except Exception as e:
        print(f"  DINOv2 failed ({e}), trying vit_b_16...")
        backbone = ViTBackbone("vit_b_16").to(device)
        print("  Loaded: torchvision ViT-B/16")

    x = torch.randn(2, 3, 224, 224, device=device)
    cls_token, patch_tokens = backbone(x)
    print(f"  cls_token: {cls_token.shape}")
    print(f"  patch_tokens: {patch_tokens.shape}")
    assert cls_token.shape[0] == 2
    assert patch_tokens.shape[0] == 2
    assert cls_token.shape[1] == patch_tokens.shape[2]  # same embed dim
    print("  [OK] Backbone OK")
    return backbone.embed_dim, patch_tokens.shape[1]


def test_geometry_branch():
    print("\n--- Stage 2: Geometry Branch ---")
    from models.geometry_branch import GeometryBranch

    geo = GeometryBranch(num_predicates=8, embed_dim=256).to(device)
    x = torch.randn(2, 3, 224, 224, device=device)
    landmarks = torch.rand(2, 468, 3, device=device)

    embedding, predicates = geo(x, landmarks)
    print(f"  geometry_embedding: {embedding.shape}")
    print(f"  geometry_predicates: {predicates.shape}")
    assert embedding.shape == (2, 256)
    assert predicates.shape == (2, 8)
    assert (predicates >= 0).all() and (predicates <= 1).all()
    print("  [OK] Geometry Branch OK")


def test_texture_branch():
    print("\n--- Stage 3: Texture Branch ---")
    from models.texture_branch import TextureBranch

    tex = TextureBranch(texture_dim=128, num_predicates=5).to(device)
    x = torch.randn(2, 3, 224, 224, device=device)
    landmarks = torch.rand(2, 468, 3, device=device)

    roi_features, tex_predicates = tex(x, landmarks)
    print(f"  roi_features: {roi_features.shape}")
    print(f"  texture_predicates: {tex_predicates.shape}")
    assert roi_features.shape == (2, 7, 128)
    assert tex_predicates.shape == (2, 5)
    print("  [OK] Texture Branch OK")


def test_multi_modal_repo(vit_dim, num_patches):
    print("\n--- Stage 4: Multi-Modal Repository ---")
    from models.multi_modal_repo import MultiModalRepository

    repo = MultiModalRepository(
        vit_dim=vit_dim, geo_predicates=8, tex_dim=128, common_dim=256
    ).to(device)

    patch_tokens = torch.randn(2, num_patches, vit_dim, device=device)
    geo_preds = torch.rand(2, 8, device=device)
    roi_feats = torch.randn(2, 7, 128, device=device)

    vit_pool, geo_pool, tex_pool = repo(patch_tokens, geo_preds, roi_feats)
    print(f"  vit_pool: {vit_pool.shape}")
    print(f"  geo_pool: {geo_pool.shape}")
    print(f"  tex_pool: {tex_pool.shape}")
    assert vit_pool.shape == (2, num_patches, 256)
    assert geo_pool.shape == (2, 8, 256)
    assert tex_pool.shape == (2, 7, 256)
    print("  [OK] Multi-Modal Repository OK")


def test_muscle_query_attention():
    print("\n--- Stage 5: Muscle Query Cross-Attention ---")
    from models.muscle_query_attention import MuscleQueryAttention

    mqa = MuscleQueryAttention(num_muscles=18, dim=256, num_heads=8, num_layers=2).to(device)

    vit_pool = torch.randn(2, 100, 256, device=device)
    geo_pool = torch.randn(2, 8, 256, device=device)
    tex_pool = torch.randn(2, 7, 256, device=device)

    muscle_emb, gate_info = mqa(vit_pool, geo_pool, tex_pool)
    print(f"  muscle_embeddings: {muscle_emb.shape}")
    print(f"  gate_info: {gate_info}")
    assert muscle_emb.shape == (2, 18, 256)
    print("  [OK] Muscle Query Attention OK")


def test_muscle_activation():
    print("\n--- Stage 6: Muscle Activation ---")
    from models.muscle_activation import MuscleActivationHead

    head = MuscleActivationHead(embed_dim=256, num_muscles=18).to(device)
    emb = torch.randn(2, 18, 256, device=device)
    act = head(emb)
    print(f"  muscle_activations: {act.shape}")
    assert act.shape == (2, 18)
    assert (act >= 0).all() and (act <= 1).all()
    print("  [OK] Muscle Activation OK")


def test_muscle_graph():
    print("\n--- Stage 7: Muscle Graph Transformer ---")
    from models.muscle_graph import MuscleGraphTransformer
    from knowledge.muscle_anatomy import get_muscle_adjacency

    graph = MuscleGraphTransformer(dim=256, num_heads=4, num_layers=2).to(device)
    adj = get_muscle_adjacency().to(device)
    emb = torch.randn(2, 18, 256, device=device)

    refined = graph(emb, adj)
    print(f"  refined_embeddings: {refined.shape}")
    assert refined.shape == (2, 18, 256)
    print("  [OK] Muscle Graph OK")


def test_reasoning():
    print("\n--- Stage 8: Symbolic Reasoning ---")
    from reasoning.predicates import PredicateStore
    from reasoning.rules import SymbolicRuleEngine
    from reasoning.counterfactual import CounterfactualEngine

    engine = SymbolicRuleEngine(num_aus=12).to(device)
    cf_engine = CounterfactualEngine(num_muscles=18).to(device)

    geo_preds = torch.rand(2, 8, device=device)
    tex_preds = torch.rand(2, 5, device=device)
    muscle_act = torch.rand(2, 18, device=device)

    preds = PredicateStore(geo_preds, tex_preds, muscle_act)
    au_preds, rules, compat = engine(preds, muscle_act)
    cf_loss = cf_engine(muscle_act, engine)

    print(f"  au_predictions: {au_preds.shape}")
    print(f"  num_rules: {len(rules)}")
    print(f"  num_compat: {len(compat)}")
    print(f"  cf_loss: {cf_loss.item():.4f}")
    assert au_preds.shape == (2, 12)
    print("  [OK] Symbolic Reasoning OK")


def test_full_model():
    print("\n--- Full Model Integration ---")
    from models.muscleaunet import MuscleAUNet

    try:
        model = MuscleAUNet(backbone_name="dinov2_vitb14").to(device)
    except Exception:
        model = MuscleAUNet(backbone_name="vit_b_16").to(device)

    x = torch.randn(2, 3, 224, 224, device=device)
    landmarks = torch.rand(2, 468, 3, device=device)
    output = model(x, landmarks=landmarks)

    print(f"  au_preds: {output['au_preds'].shape}")
    print(f"  muscle_activations: {output['muscle_activations'].shape}")
    print(f"  geometry_predicates: {output['geometry_predicates'].shape}")
    print(f"  texture_predicates: {output['texture_predicates'].shape}")
    print(f"  num_rules: {len(output['rule_satisfactions'])}")
    print(f"  cf_loss: {output['cf_loss'].item():.4f}")

    assert output["au_preds"].shape == (2, 12)
    assert output["muscle_activations"].shape == (2, 18)

    # Test gradient flow
    target = torch.randint(0, 2, (2, 12), device=device).float()
    loss = torch.nn.functional.binary_cross_entropy(output["au_preds"], target)
    loss.backward()

    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_p = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Gradient flow: {has_grad}/{total_p} params received gradients")
    assert has_grad > 0, "No gradients flowing!"
    print("  [OK] Full Model OK")


if __name__ == "__main__":
    print("=" * 60)
    print("MuscleAU-Net Smoke Test")
    print("=" * 60)

    try:
        vit_dim, num_patches = test_backbone()
        test_geometry_branch()
        test_texture_branch()
        test_multi_modal_repo(vit_dim, num_patches)
        test_muscle_query_attention()
        test_muscle_activation()
        test_muscle_graph()
        test_reasoning()
        test_full_model()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[X] FAILED: {e}")
        import traceback
        traceback.print_exc()
