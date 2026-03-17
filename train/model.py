# ---------------------------------------------------------------------------
# Model definitions for temporal proxemics learning
#
# This file contains:
#   - Video backbones:
#       * R(2+1)D-18
#       * MViTv2-S
#   - A projection module to map branch features to a shared embedding space
#   - Three multimodal / multi-branch architectures:
#       * TemporalBranch            -> single-task classification
#       * MultiTaskCrossAttention   -> multitask with cross-attention fusion
#       * MultiTaskCLStoken         -> multitask with [CLS] token fusion
# Each architecture allows flexible combinations of input branches (RGB individual, RGB pair, Audio) and backbone choices.

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, mvit_v2_s, MViT_V2_S_Weights



# ===========================================================================
# R(2+1)D Backbone
# ===========================================================================
class R2Plus1DBackbone(nn.Module):
    """
    Video backbone based on torchvision's pretrained R(2+1)D-18 model.

    Notes
    -----
    - The final classification layer is replaced with Identity so that the
      backbone returns feature embeddings instead of logits.
    - A configurable number of early blocks can be frozen for fine-tuning.
    - Input is expected as (B, T, C, H, W), and is internally permuted to
      (B, C, T, H, W).
    """
    def __init__(self, nlayersFreeze=0):
        super().__init__()
        weights=R2Plus1D_18_Weights.DEFAULT
        self.model = r2plus1d_18(weights=weights)
        # Feature dimensionality before the original classifier
        feature_dimension = self.model.fc.in_features
        # Replace the final classification layer with identity
        # nn.Identity() is a placeholder that returns its input unchanged, effectively removing the classification layer.
        self.model.fc = nn.Identity()
        # Freeze early blocks if requested
        freeze_blocks = ['stem', 'layer1', 'layer2', 'layer3', 'layer4']
        num_blocks_to_freeze = nlayersFreeze

        frozen = set()
        for name, param in self.model.named_parameters():
            for block in freeze_blocks[:num_blocks_to_freeze]:
                if name.startswith(block):
                    param.requires_grad = False
                    frozen.add(block)
                    break  
        if frozen:
            print(f"[INFO] Frozen R(2+1)D blocks: {sorted(frozen)}")
  
    def forward(self, x):  # (B, 3, T, 112, 112)
        """
        Parameters
        ----------
        x : torch.Tensor
            Input video tensor with shape (B, T, C, H, W)

        Returns
        -------
        torch.Tensor
            Backbone features with shape (B, 512)
        """
        x = x.permute(0, 2, 1, 3, 4) # From (B, T, C, H, W) → ResNet  (B, C, T, H, W)
        x = self.model(x)
        return x


# ===========================================================================
# MViTv2 Small Backbone (k400)
# ===========================================================================
class MViTv2SmallBackbone(nn.Module):
    """
    Video backbone based on torchvision's pretrained MViTv2-S model.

    Notes
    -----
    - The final classification layer is replaced with Identity so that the
      backbone returns feature embeddings instead of logits.
    - The model output dimensionality is 768.
    - A percentage of transformer blocks can be frozen according to
      `nlayersFreeze` in the range [0, 5], mapped to [0%, 100%].
    - Input is expected as (B, T, C, H, W), and is internally permuted to
      (B, C, T, H, W).
    """
    def __init__(self, nlayersFreeze=0):
        super().__init__()
        weights = MViT_V2_S_Weights.KINETICS400_V1
        self.model = mvit_v2_s(weights=weights)

        # Save output dimensionality from the original classification head
        final_linear = self.model.head[1]
        self.dim = final_linear.in_features  # → 768
        # Replace final classifier with identity
        self.model.head[1] = nn.Identity()
        
        # Freeze a percentage of blocks according to nlayersFreeze
        total_blocks = len(self.model.blocks)  # normalmente 16
        freeze_percent = nlayersFreeze * 20  # 0, 20, 40, ..., 100
        num_to_freeze = int((freeze_percent / 100.0) * total_blocks)

        for i, block in enumerate(self.model.blocks):
            if i < num_to_freeze:
                for param in block.parameters():
                    param.requires_grad = False
        
        if num_to_freeze > 0:
            print(f"[INFO] Frozen MViTv2 blocks: "f"{num_to_freeze}/{total_blocks}")

    def forward(self, x):  # x: (B, T, C, H, W)
        """
        Parameters
        ----------
        x : torch.Tensor
            Input video tensor with shape (B, T, C, H, W)

        Returns
        -------
        torch.Tensor
            Backbone features with shape (B, 768)
        """
        x = x.permute(0, 2, 1, 3, 4)  
        return self.model(x)  # → (B, 768)

# ===========================================================================
# Projection Module
# ===========================================================================
class ProjectionModule(nn.Module):
    """
    Small MLP that projects backbone features into a shared 512-D space.

    Parameters
    ----------
    dim : int
        Input feature dimensionality from the backbone.
    """
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Sequential(OrderedDict([
            ("dense1", nn.Linear(dim, 512)),
            ("ln", nn.LayerNorm(512)),
            ("act", nn.Tanh()),
            ("dense2", nn.Linear(512, 512)),
        ]))
    def forward(self, x): return self.proj(x)


# ===========================================================================
# TemporalBranch
# ===========================================================================
class TemporalBranch(nn.Module):
    """
    Single-task temporal model for proxemics classification.

    This model supports multiple branches (RGB individual streams, RGB pair
    stream, and optional audio). Each visual branch uses a backbone to extract
    temporal features, which are then projected into a shared 512-D space.

    Fusion strategy:
    - Branch-specific projected tokens are concatenated
    - Cross-attention is applied using each branch token as key/value
    - A self-attention layer further refines the fused representation
    - Token representations are globally pooled and passed to a classifier

    Parameters
    ----------
    typeImg : str
        Modalities to use. Typically contains 'RGB'.
    onlyPairRGB : bool
        If True, only the pair RGB branch is used.
    onlyPairPose : bool
        Reserved for pose-based branches.
    audio : bool
        Whether to include the audio branch.
    nlayersFreeze : int
        Backbone freezing level.
    backbone_class : nn.Module
        Backbone class to instantiate for visual branches.
    dim : int
        Backbone output dimensionality.
    num_labels : int
        Number of output labels.
    dropout_p : float
        Dropout probability in the classifier.
    num_heads : int
        Number of attention heads.
    """
    def __init__(self,
                 typeImg='RGB',
                 onlyPairRGB=False,
                 onlyPairPose=False,
                 audio=False,
                 nlayersFreeze=0,
                 backbone_class=R2Plus1DBackbone,        
                 dim=512,
                 num_labels=6,
                 dropout_p=0.3,
                 num_heads=4):
        super().__init__()

        self.typeImg = typeImg
        self.onlyPairRGB = onlyPairRGB
        self.onlyPairPose = onlyPairPose
        self.nlayersFreeze = nlayersFreeze
        self.dim = dim

        # Dictionary of shared backbones 
        self.backbones = nn.ModuleDict()
        # Branch-specific projection heads
        self.projections = nn.ModuleDict()
        # Branch processing order must match the order of inputs in forward()
        self.branch_order = []
        # -------------------------------------------------------------------
        # RGB branches
        # -------------------------------------------------------------------
        if 'RGB' in typeImg:
            if not onlyPairRGB:
                # Shared backbone for individual person streams
                self.backbones['rgb_indiv'] = backbone_class(nlayersFreeze)
                
                self.projections['rgb_p0'] = ProjectionModule(dim)
                self.projections['rgb_p1'] = ProjectionModule(dim)
                self.branch_order += ['rgb_p0', 'rgb_p1']
            # Separate backbone for the pair stream
            self.backbones['rgb_pair'] = backbone_class(nlayersFreeze)
            self.projections['rgb_pair'] = ProjectionModule(dim)
            self.branch_order.append('rgb_pair')

        # -------------------------------------------------------------------
        # Audio branch
        # -------------------------------------------------------------------
        if audio:
            self.branch_order.append("audio")
            self.projections["audio"] =nn.Linear(80, 512)

        # === Attention blocks ===
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=num_heads, batch_first=True)

        # === Classification head ===
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(512, num_labels)
        )

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : list[torch.Tensor]
            List of branch inputs in the same order as self.branch_order.

        Returns
        -------
        torch.Tensor
            Raw logits of shape (B, num_labels)
        """
        tokens = []

        for x, branch_name in zip(inputs, self.branch_order):
            # Select backbone based on branch name
            if branch_name in ['rgb_p0', 'rgb_p1']:
                b = self.backbones['rgb_indiv']
            elif branch_name == 'rgb_pair':
                b = self.backbones['rgb_pair']
            elif branch_name == 'audio':
                # # Audio is already an embedding/vector input, so no backbone (B, 80)
                features = x  # (B, D)
                projected = self.projections['audio'](features)
                tokens.append(projected.unsqueeze(1))  # (B, 1, 512)
                continue  
            else:
                raise ValueError(f"Branch '{branch_name}' does not recognised.")

            # Extract features and project them into the shared space
            p = self.projections[branch_name]

            features = b(x)
            projected = p(features)
            tokens.append(projected.unsqueeze(1))  # (B, 1, 512)

        # Concatenate tokens from all branches
        concat = torch.cat(tokens, dim=1)  # (B, N, 512)

        # Cross-attention with each branch token as key/value
        cross_outputs = []
        for t in tokens:
            attn_out, _ = self.cross_attn(query=concat, key=t, value=t)
            cross_outputs.append(attn_out)
        # Aggregate attention outputs across branches
        attn_sum = torch.stack(cross_outputs, dim=0).sum(0)  # (B, N, 512)

        # Self-attention refinement with residual connection
        self_out, _ = self.self_attn(attn_sum, attn_sum, attn_sum)
        caf_tokens = attn_sum + self_out                     # residual
        # Global average pooling across tokens
        pooled = caf_tokens.mean(dim=1)                     # global pool → (B, 512)
        # Final classification logits
        logits = self.classifier(pooled)                    # (B, 6)
        
        return logits  # raw logits 

# ===========================================================================
# MultiTaskCrossAttention
# ===========================================================================
class MultiTaskCrossAttention(nn.Module):
    """
    Multitask temporal model with cross-attention fusion.

    Compared to TemporalBranch, this model produces two outputs:
    - proxemics
    - relationship

    The shared multimodal representation is built in the same way as in
    TemporalBranch, but two task-specific heads are used at the end.
    """
    def __init__(self,
                 typeImg='RGB',
                 onlyPairRGB=False,
                 onlyPairPose=False,
                 audio=False,
                 nlayersFreeze=0,
                 backbone_class=R2Plus1DBackbone,        
                 dim=512,
                 num_labels=6,
                 dropout_p=0.3,
                 num_heads=4):
        super().__init__()

        self.typeImg = typeImg
        self.onlyPairRGB = onlyPairRGB
        self.onlyPairPose = onlyPairPose
        self.nlayersFreeze = nlayersFreeze
        self.dim = dim

        self.backbones = nn.ModuleDict()
        self.projections = nn.ModuleDict()

        self.branch_order = []
        # -------------------------------------------------------------------
        # RGB branches
        # -------------------------------------------------------------------
        if 'RGB' in typeImg:
            if not onlyPairRGB:
                self.backbones['rgb_indiv'] = backbone_class(nlayersFreeze)
                
                self.projections['rgb_p0'] = ProjectionModule(dim)
                self.projections['rgb_p1'] = ProjectionModule(dim)
                self.branch_order += ['rgb_p0', 'rgb_p1']
            
            self.backbones['rgb_pair'] = backbone_class(nlayersFreeze)
            self.projections['rgb_pair'] = ProjectionModule(dim)
            self.branch_order.append('rgb_pair')

        # -------------------------------------------------------------------
        # Audio branch
        # -------------------------------------------------------------------
        if audio:
            self.branch_order.append("audio")
            self.projections["audio"] =nn.Linear(80, 512)

        # === Attention blocks ===
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=num_heads, batch_first=True)


        # === Task-specific heads ===
        self.prox_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)   # logits multi-label
        )
        self.rel_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)   # logits multi-class
        )

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : list[torch.Tensor]
            List of branch inputs in the same order as self.branch_order.

        Returns
        -------
        dict
            Dictionary with:
            - 'proxemics': logits for multi-label classification
            - 'relationship': logits for multi-class classification
        """
        tokens = []

        for x, branch_name in zip(inputs, self.branch_order):
            if branch_name in ['rgb_p0', 'rgb_p1']:
                b = self.backbones['rgb_indiv']
            elif branch_name == 'rgb_pair':
                b = self.backbones['rgb_pair']
            elif branch_name == 'audio':
                features = x  # (B, D)
                projected = self.projections['audio'](features)
                tokens.append(projected.unsqueeze(1))
                continue  
            else:
                raise ValueError(f"Branch '{branch_name}' does not recognised.")

            p = self.projections[branch_name]

            features = b(x)
            projected = p(features)
            tokens.append(projected.unsqueeze(1))  # (B, 1, 512)

        concat = torch.cat(tokens, dim=1)  # (B, N, 512)

        # Cross-attention 
        cross_outputs = []
        for t in tokens:
            attn_out, _ = self.cross_attn(query=concat, key=t, value=t)
            cross_outputs.append(attn_out)

        attn_sum = torch.stack(cross_outputs, dim=0).sum(0)  # (B, N, 512)

        self_out, _ = self.self_attn(attn_sum, attn_sum, attn_sum)
        caf_tokens = attn_sum + self_out                     # residual

        pooled = caf_tokens.mean(dim=1)                     # global pool → (B, 512)
        # === Two-head output ===
        logits_prox = self.prox_head(pooled)  # (B, num_labels_prox)
        logits_rel  = self.rel_head(pooled)   # (B, num_classes_rel)

        return {
            "proxemics": logits_prox,     # use BCEWithLogitsLoss
            "relationship": logits_rel    # use CrossEntropyLoss
        }

# ===========================================================================
# MultiTaskCLStoken
# ===========================================================================
class MultiTaskCLStoken(nn.Module):
    """
    Multitask temporal model with [CLS] token fusion.

    This architecture:
    - extracts a token from each branch,
    - concatenates them,
    - prepends a learnable [CLS] token,
    - processes everything with a Transformer encoder,
    - uses the final [CLS] representation for two task-specific heads.

    Outputs
    -------
    dict
        {
            "proxemics": logits,
            "relationship": logits
        }
    """
    def __init__(self,
                 typeImg='RGB',
                 onlyPairRGB=False,
                 onlyPairPose=False,
                 audio=False,
                 nlayersFreeze=0,
                 backbone_class=R2Plus1DBackbone,        
                 dim=512,
                 num_labels=6,
                 num_heads=4,
                 num_sa_layers=2):
        super().__init__()

        self.typeImg = typeImg
        self.onlyPairRGB = onlyPairRGB
        self.onlyPairPose = onlyPairPose
        self.nlayersFreeze = nlayersFreeze
        self.dim = dim

        
        self.backbones = nn.ModuleDict()
        self.projections = nn.ModuleDict()

        self.branch_order = []
        # -------------------------------------------------------------------
        # RGB branches
        # -------------------------------------------------------------------
        if 'RGB' in typeImg:
            if not onlyPairRGB:
                self.backbones['rgb_indiv'] = backbone_class(nlayersFreeze)
                
                self.projections['rgb_p0'] = ProjectionModule(dim)
                self.projections['rgb_p1'] = ProjectionModule(dim)
                self.branch_order += ['rgb_p0', 'rgb_p1']
            
            self.backbones['rgb_pair'] = backbone_class(nlayersFreeze)
            self.projections['rgb_pair'] = ProjectionModule(dim)
            self.branch_order.append('rgb_pair')
        
        # -------------------------------------------------------------------
        # Audio branch
        # -------------------------------------------------------------------
        if audio:
            self.branch_order.append("audio")
            self.projections["audio"] =nn.Linear(80, 512)

        # === Learnable [CLS] token ===
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.trunc_normal_(self.cls_token, std=.02)

        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_sa_layers)

        # === Task-specific heads ===
        self.proxemics_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_labels)
        )

        self.relation_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_labels)
        )


    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : list[torch.Tensor]
            List of branch inputs in the same order as self.branch_order.

        Returns
        -------
        dict
            Dictionary with:
            - 'proxemics': logits for multi-label classification
            - 'relationship': logits for multi-class classification
        """
        tokens = []

        for x, branch_name in zip(inputs, self.branch_order):
            if branch_name in ['rgb_p0', 'rgb_p1']:
                b = self.backbones['rgb_indiv']
            elif branch_name == 'rgb_pair':
                b = self.backbones['rgb_pair']
            elif branch_name == 'audio':
                features = x  # (B, D)
                projected = self.projections['audio'](features)
                tokens.append(projected.unsqueeze(1))
                continue  # ya hemos hecho todo para esta rama
            else:
                raise ValueError(f"Branch '{branch_name}' does not recognised.")

            p = self.projections[branch_name]

            features = b(x)
            projected = p(features)
            tokens.append(projected.unsqueeze(1))  # (B, 1, 512)

        concat = torch.cat(tokens, dim=1)  # (B, N, 512)

        cls_token = self.cls_token.expand(concat.size(0), 1, 512)  # (B, 1, 512)
        x = torch.cat([cls_token, concat], dim=1)    # (B, N+1, 512)

        encoded = self.encoder(x)                    # (B, N+1, 512)
        cls_out = encoded[:, 0]                      # (B, 512)

        return {
            "proxemics": self.proxemics_head(cls_out),  # multilabel
            "relationship": self.relation_head(cls_out)     # multiclass
        }

