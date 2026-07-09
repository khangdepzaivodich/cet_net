# Research Proposal Prompt

You are an expert researcher in computer vision, neuro-symbolic AI, graph neural networks, differentiable logic, and facial Action Unit (AU) recognition.

Your task is to design and implement a **complete research framework** for Action Unit Recognition that is sufficiently novel for a top-tier conference (CVPR / ICCV / ECCV / NeurIPS).

The target datasets are:

* BP4D
* DISFA

The implementation should be in PyTorch and designed as a modular research codebase.

---

# Objective

Current AU recognition methods (including SymGraphAU) mainly learn

Image → Feature → AU

or

Image → Graph → AU.

The proposed framework should instead learn

Image → Visual Evidence → Latent Muscle Representation → Neuro-Symbolic Reasoning → AU

The core hypothesis is:

Facial muscles are the hidden causal variables that generate

* landmark movement
* wrinkle formation
* local texture
* global appearance

Therefore the symbolic reasoning should operate over **latent muscle activations**, not directly over AU labels.

The framework must NOT simply add a logic loss on AU predictions.

Instead, logic should operate throughout the perception pipeline.

---

# Overall Architecture

Design the following architecture.

RGB Image

↓

Global Appearance Encoder (Vision Transformer)

↓

Global Tokens

*

Geometry Branch

↓

Landmarks

↓

Geometric Features

*

Local Texture Branch

↓

ROI Features

↓

Muscle Query Cross Attention

↓

Muscle Embeddings

↓

Anatomical Muscle Graph Transformer

↓

Neuro-Symbolic Reasoning Layer

↓

AU Predictions

↓

Losses

The implementation should be modular.

---

# Stage 1 — Global Appearance Encoder

Use a Vision Transformer.

Candidate backbones:

* ViT-B/16
* Swin Transformer
* EVA-02
* DINOv2

Input:

256×256 RGB face.

Output:

Global tokens

T_global ∈ R^(N×D)

The implementation should allow replacing the backbone.

---

# Stage 2 — Geometry Branch

Use MediaPipe Face Mesh or HRNet.

Extract facial landmarks.

Compute differentiable geometric predicates.

Examples:

* mouth corner displacement
* eyebrow height
* eyebrow distance
* lip distance
* eye openness
* cheek displacement
* nose width
* chin displacement

Do NOT use handcrafted thresholds.

Instead normalize these values into continuous values between 0 and 1.

Output:

Geometry embedding.

---

# Stage 3 — Local Texture Branch

Instead of using only global ViT features, extract local high-frequency facial textures.

Use landmarks to define ROIs.

Example ROIs:

* left eye
* right eye
* forehead
* nose
* mouth
* left cheek
* right cheek

Each ROI should be processed by a lightweight CNN.

Output:

ROI texture embeddings.

The CNN should specialize in learning:

* crow's feet
* forehead wrinkles
* glabellar wrinkles
* nasolabial folds
* chin wrinkles

---

# Stage 4 — Multi-modal Repository

Do NOT concatenate features immediately.

Maintain separate feature pools.

Repository:

* Global ViT tokens
* ROI texture embeddings
* Geometry embeddings

These become the keys and values for cross-attention.

---

# Stage 5 — Muscle Query Cross Attention

This is the core innovation.

Create one learnable query per facial muscle.

Example muscles:

* Frontalis medialis
* Frontalis lateralis
* Corrugator supercilii
* Depressor supercilii
* Orbicularis oculi (orbital)
* Orbicularis oculi (palpebral)
* Levator palpebrae superioris
* Levator labii superioris
* Levator labii superioris alaeque nasi
* Zygomaticus major
* Zygomaticus minor
* Risorius
* Buccinator
* Depressor anguli oris
* Depressor labii inferioris
* Mentalis
* Orbicularis oris
* Nasalis

Each muscle query attends to every modality.

Example:

Q_Zygomaticus attends to

* mouth ViT tokens
* cheek ViT tokens
* mouth ROI texture
* cheek ROI texture
* mouth geometry

Output:

One embedding per muscle.

These embeddings represent latent muscle activations.

---

# Stage 6 — Muscle Activation Estimation

Each muscle embedding

↓

MLP

↓

Probability

Example

Zygomaticus = 0.91

Orbicularis = 0.83

Frontalis = 0.11

These muscle activations are latent variables.

No muscle supervision exists.

They must be learned jointly.

---

# Stage 7 — Anatomical Muscle Graph Transformer

Construct a graph whose nodes are muscles.

Do NOT construct an AU graph.

Use anatomical adjacency.

Examples:

Frontalis ↔ Corrugator

Corrugator ↔ Depressor Supercilii

Zygomaticus Major ↔ Risorius

Levator Labii Superioris ↔ LLSAN

Orbicularis Oris ↔ Mentalis

Orbicularis Oculi ↔ Levator Palpebrae

Implement using Graph Attention Network or Graph Transformer.

The graph should refine latent muscle embeddings.

---

# Stage 8 — Neuro-Symbolic Reasoning

Implement differentiable symbolic reasoning.

Use ONE formalism consistently.

Preferred:

Logic Tensor Networks (LTN)

or

Probabilistic Soft Logic (PSL)

or

Differentiable fuzzy logic.

Do NOT use hard rules.

Every predicate must produce a differentiable truth value.

---

# Predicates

Implement three categories of predicates.

## Geometry predicates

Examples:

CornerUp

CornerDown

EyeClosed

BrowRaised

UpperLipRaised

LipSeparated

CheekRaised

These are computed from normalized landmark geometry.

---

## Texture predicates

Examples:

CrowFeet

ForeheadWrinkle

GlabellarWrinkle

NasolabialFold

ChinWrinkle

These are outputs of the ROI CNN.

---

## Muscle predicates

Examples:

Zygomaticus

Orbicularis

Corrugator

Mentalis

Frontalis

These come from the muscle-query transformer.

---

# Symbolic Rule Base

Implement the following differentiable rules.

## Geometry → Muscle

CornerUp

→

Zygomaticus Major

EyeClosing

→

Orbicularis Oculi

InnerBrowRaise

→

Frontalis Medialis

OuterBrowRaise

→

Frontalis Lateralis

BrowsTogether

→

Corrugator

UpperLipRaise

→

Levator Labii Superioris

NostrilRaise

→

LLSAN

CornerDown

→

Depressor Anguli Oris

LipStretch

→

Risorius

ChinRaise

→

Mentalis

---

## Texture → Muscle

CrowFeet

→

Orbicularis Oculi

ForeheadWrinkle

→

Frontalis

GlabellarWrinkle

→

Corrugator

NasolabialFold

→

Zygomaticus Major

NoseWrinkle

→

LLSAN

ChinWrinkle

→

Mentalis

---

## Muscle → AU

Frontalis Medialis

→ AU1

Frontalis Lateralis

→ AU2

Corrugator

→ AU4

Levator Palpebrae

→ AU5

Orbicularis Oculi

→ AU6

Orbicularis Palpebralis

→ AU7

LLSAN

→ AU9

Levator Labii Superioris

→ AU10

Zygomaticus Minor

→ AU11

Zygomaticus Major

→ AU12

Buccinator

→ AU14

Depressor Anguli Oris

→ AU15

Depressor Labii Inferioris

→ AU16

Mentalis

→ AU17

Risorius

→ AU20

Orbicularis Oris

→ AU22–24

---

## Multiple Evidence → Muscle

CornerUp

AND

NasolabialFold

→

Zygomaticus

EyeClosing

AND

CrowFeet

→

Orbicularis

BrowRaise

AND

ForeheadWrinkle

→

Frontalis

UpperLipRaise

AND

NoseWrinkle

→

LLSAN

ChinRaise

AND

ChinWrinkle

→

Mentalis

---

## Multiple Muscle → AU

Corrugator

AND

Depressor Supercilii

→ AU4

LLSAN

AND

Levator Labii Superioris

→ AU10

Zygomaticus Major

AND

Orbicularis Oculi

→ Duchenne smile

Orbicularis Oris

AND

Mentalis

→ AU18–24 family

---

## Muscle Compatibility Rules

Implement anatomical consistency.

Examples

High Zygomaticus

AND

High DAO

↓

Penalty

High Frontalis

AND

High Corrugator

↓

Weak compatibility

High Orbicularis

↓

Likely CheekRaise

High Mentalis

↓

Likely ChinWrinkle

Implement these as differentiable constraints.

---

# Logic-Guided Attention

Implement a feedback mechanism.

Instead of

Feature

↓

Logic

only,

logic should influence feature extraction.

Modify attention

Attention = Softmax(QKᵀ + LogicBias)

LogicBias should be computed from the symbolic reasoning module.

This allows symbolic reasoning to guide attention toward anatomically relevant regions.

---

# Counterfactual Learning

Implement intervention.

Example

Original

Zygomaticus = 0.95

↓

AU12 = True

Intervene

do(Zygomaticus = 0)

Expected

AU12 decreases.

Repeat for every muscle.

Compute a counterfactual consistency loss.

---

# Loss Function

Implement

L =

L_AU

*

λ1 L_logic

*

λ2 L_counterfactual

*

λ3 L_graph

*

λ4 L_attention

Design appropriate mathematical formulations for each loss.

---

# Training

Support

BP4D

DISFA

Provide:

* dataloaders
* preprocessing
* evaluation
* F1 score
* per-AU metrics
* ablation studies

---

# Ablation Studies

Implement experiments removing:

* texture branch
* geometry branch
* muscle graph
* logic layer
* logic-guided attention
* counterfactual loss
* muscle compatibility rules

---

# Deliverables

Produce:

1. Complete mathematical formulation for every module.
2. Network architecture diagrams.
3. Detailed explanation of tensor shapes through the network.
4. Formal definitions of all predicates and symbolic rules.
5. Exact implementation of differentiable logic.
6. Complete PyTorch implementation.
7. Training scripts.
8. Evaluation scripts for BP4D and DISFA.
9. Configuration files.
10. Clear justification of why the proposed framework is novel compared with SymGraphAU and other state-of-the-art AU recognition methods.

Do not simplify any module. Before implementing, critically evaluate the entire framework for technical correctness, novelty, computational feasibility, differentiability, and potential weaknesses. If any component is inconsistent, biologically implausible, redundant, or unlikely to survive peer review at CVPR/ICCV/NeurIPS, redesign it while preserving the central idea of anatomically grounded neuro-symbolic muscle reasoning. Explain every design decision with references to relevant literature where appropriate.
