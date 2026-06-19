Novel framework: Causal Evidence Transport Network (CET-Net)
1. First principles

A face image contains three things:

Localized evidence: tiny muscle movements.
Latent causes: AUs.
Composed outcomes: expressions.

So the model should not directly classify expressions from pixels, and it should not treat rules as fixed post-processing. It should infer a latent AU state, then let expressions emerge as a consequence of that state.

That gives one clean principle:

Image → evidence → AU beliefs → expression beliefs

Knowledge is not a separate module. It is a prior over how evidence is allowed to flow.

2. Core idea

Represent the face as a field of AU evidence, then run belief transport through a factor graph.

Instead of:

local graph + global attention + symbolic graph + ZSL head

use:

evidence extraction
uncertainty-calibrated AU inference
causal AU–expression factor passing
composition-based unseen class synthesis

This is more novel because the model is built around belief propagation under uncertainty, not around a stack of attention blocks.

3. Architecture
3.1 AU evidence field

The backbone produces a dense feature map F∈R
H×W×C
.

For each AU k, the model predicts:

an evidence score e
k
	​

an uncertainty score u
k
	​

a soft spatial mask m
k
	​


So each AU is not just a node embedding. It is a belief variable with confidence.

z
k
	​

=σ(g
k
	​

(F))

where z
k
	​

 is the soft AU belief.

3.2 Causal AU factor graph

The AU layer is a factor graph with three factor types:

local evidence factors: how strongly the image supports AU k
interaction factors: how AUs reinforce or suppress each other
structural factors: known FACS logic and co-activation tendencies

The key difference from a normal graph network is this:

edges are not just learned similarity; they are interpreted as belief constraints

So the update is not only message passing, but belief transport:

z
k
t+1
	​

=Update(z
k
t
	​

,{z
j
t
	​

}
j

=k
	​

,F,Ω)

where Ω is the rule set.

3.3 Expression as a higher-order consequence

Expression is not predicted independently.

It is computed from the AU belief vector z through a compositional operator:

p(y
expr
∣z)=Softmax(h(z))

But h(⋅) should not be a plain MLP. It should be a composition layer that learns which AU groups define which expression, while staying grounded in AU semantics.

This makes expressions a consequence of AU structure, not a separate label head.

3.4 Uncertainty-gated rule activation

This is where the framework becomes distinct.

A rule should not be applied just because it exists. It should be applied only when the model is uncertain or inconsistent.

For each rule r
m
	​

, define a confidence gate:

γ
m
	​

=sigmoid(a⋅viol(r
m
	​

)+b⋅uncertainty(z))

So:

if the model is already confident and consistent, the rule contributes little
if the model is uncertain or contradicting known structure, the rule contributes more

This is better than static symbolic regularization because the knowledge base becomes adaptive per sample.

4. Zero-shot transfer mechanism

Unseen expressions should be constructed as compositions of AU atoms, not as new class embeddings learned from labels.

For an unseen class c
u
	​

, define its AU descriptor a
c
u
	​

	​

.

Then synthesize its prototype by:

p
c
u
	​

	​

=
k=1
∑
K
	​

a
c
u
	​

,k
	​

b
k
	​

+Δ(a
c
u
	​

	​

)

where:

b
k
	​

 is the learned basis vector of AU k
Δ(⋅) is a correction term for interactions among AUs

This is the compositional heart of the ZSL part.

The correction term is important because expressions are not linear sums of AUs. The model needs a residual interaction term, or it will be too naive.

5. Novel training objective

Use four pressures on the same latent belief space:

L=λ
1
	​

L
AU
	​

+λ
2
	​

L
expr
	​

+λ
3
	​

L
rule
	​

+λ
4
	​

L
cf
	​


Where:

L
AU
	​

: AU supervision
L
expr
	​

: expression supervision
L
rule
	​

: rule violation penalty, gated by uncertainty
L
cf
	​

: counterfactual consistency loss
Counterfactual consistency loss

This is the part that makes it more first-principles.

If you suppress AU k, the expression belief should change in a predictable way.

So the model is trained to satisfy:

remove AU evidence
expression belief should shift accordingly
unrelated AUs should remain stable

This forces the network to learn causal dependence, not just correlation.

6. Why this is genuinely different

Your current draft combines:

dynamic AU graphs
global attention
symbolic clauses
zero-shot prototypes

That is a useful system, but it still reads like a composite of existing ideas.

This framework is different because it is organized around one principle:

expression recognition is inference over uncertain causal AU evidence

Everything else follows from that.

So instead of asking:

how do we combine three papers?

ask:

how does a face generate evidence for latent AUs?
how do AUs generate expressions?
when should knowledge constrain inference?
how do unseen classes arise from AU composition?

That is a cleaner novel methodology.