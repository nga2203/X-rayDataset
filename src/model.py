import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel
import math

class ASSISTANCE_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config['architecture']
        
        # 1. Encoders (Set global_pool='' to keep spatial dimensions for Swin/CNN)
        self.img_encoder = timm.create_model(
            cfg['image_encoder'], pretrained=True, num_classes=0, global_pool=''
        )
        self.text_encoder = AutoModel.from_pretrained(cfg['text_encoder'])
        
        img_dim = self.img_encoder.num_features
        txt_dim = self.text_encoder.config.hidden_size
        fusion_dim = cfg['fusion_dim']
        
        # 2. Projections
        self.img_proj = nn.Linear(img_dim, fusion_dim)
        self.text_proj = nn.Linear(txt_dim, fusion_dim)
        self.demo_proj = nn.Sequential(
            nn.Linear(2, fusion_dim // 2), nn.ReLU(), nn.Linear(fusion_dim // 2, fusion_dim)
        )
        
        # [NEW] CLS Token for unified representation and Heatmap generation
        self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
        
        # 3. Fusion (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=8, batch_first=True, dropout=cfg['dropout']
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=cfg['num_layers_fusion'])
        
        # 4. Head 1: Classification (BCE)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg['dropout']),
            nn.Linear(fusion_dim, cfg['num_classes'])
        )

        # 5. Head 2: Generation (Decoder)
        vocab_size = 64000 
        self.tgt_embedding = nn.Embedding(vocab_size, fusion_dim)
        self.pos_encoder = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Dropout(0.1))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=fusion_dim, nhead=8, batch_first=True)
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.lm_head = nn.Linear(fusion_dim, vocab_size)

    def encode_features(self, images, input_ids, attention_mask, demographics):
        B, N, C, H, W = images.shape
        
        # --- IMAGE ENCODING ---
        img_feats = self.img_encoder(images.view(B*N, C, H, W))
        
        # Preserve Spatial Dimensions
        if img_feats.dim() == 4:
            # Check if channel is last (like Swin: B, H, W, C) or second (like CNN: B, C, H, W)
            if img_feats.shape[-1] == self.img_encoder.num_features:
                img_feats = img_feats.reshape(img_feats.size(0), -1, img_feats.size(-1))
            else:
                img_feats = img_feats.flatten(2).transpose(1, 2)
            
        _, seq_len, _ = img_feats.shape
        img_feats = self.img_proj(img_feats).view(B, N * seq_len, -1)
        
        # --- TEXT ENCODING ---
        txt_out = self.text_encoder(input_ids, attention_mask)
        txt_feats = self.text_proj(txt_out.last_hidden_state[:, 0, :]).unsqueeze(1)
        
        # --- DEMO ENCODING ---
        demo_feats = self.demo_proj(demographics).unsqueeze(1)
        
        # --- CLS TOKEN ---
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Fusion (Concat along sequence dim)
        feats = torch.cat([cls_tokens, img_feats, txt_feats, demo_feats], dim=1)
        memory = self.fusion(feats)
        
        return memory, N, seq_len
        
    def forward(self, images, input_ids, attention_mask, demographics, target_ids=None):
        memory, _, _ = self.encode_features(images, input_ids, attention_mask, demographics)
        
        # --- TASK 1: CLASSIFICATION ---
        # Use CLS token output instead of mean pooling
        cls_out = memory[:, 0, :] 
        cls_logits = self.classifier(cls_out)
        
        # --- TASK 2: GENERATION ---
        lm_logits = None
        if target_ids is not None:
            seq_len = target_ids.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(images.device)
            tgt_emb = self.pos_encoder(self.tgt_embedding(target_ids))
            
            decoded = self.text_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            lm_logits = self.lm_head(decoded)
            
        return cls_logits, lm_logits
    
    @torch.no_grad()
    def generate(self, images, input_ids, attention_mask, demographics, tokenizer, max_len=100):
        memory, _, _ = self.encode_features(images, input_ids, attention_mask, demographics)
        
        # Decode
        curr_ids = torch.tensor([[tokenizer.bos_token_id]]).to(images.device)
        
        for _ in range(max_len):
            tgt_emb = self.pos_encoder(self.tgt_embedding(curr_ids))
            decoded = self.text_decoder(tgt=tgt_emb, memory=memory)
            logits = self.lm_head(decoded[:, -1, :])
            
            # Repetition Penalty
            for token_id in curr_ids[0]:
                logits[0, token_id] /= 1.5
            
            probs = torch.softmax(logits / 1.0, dim=-1) 
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.pad_token_id:
                break
            
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
        return tokenizer.decode(curr_ids[0], skip_special_tokens=True)

    # [NEW] HEAD 3: Lesion Localization (Heatmap/CAM)
    def get_heatmap(self, images, input_ids, attention_mask, demographics):
        """
        Generates a Weakly Supervised Heatmap using Grad-CAM on the Image Encoder.
        This extracts pure spatial features BEFORE they are mixed by the Transformer Fusion.
        """
        # Ensure we are in eval mode but gradients are enabled for Grad-CAM
        is_train = self.training
        self.eval()
        
        with torch.enable_grad():
            B = images.shape[0] if getattr(self, 'use_img', True) else demographics.shape[0]
            if not getattr(self, 'use_img', True):
                return None
                
            B_img, N, C, H, W = images.shape
            img_feats_raw = self.img_encoder(images.view(B_img*N, C, H, W))
            
            # Retain gradients on the raw output BEFORE reshaping it
            img_feats_raw.retain_grad()
            
            # Format spatial features for forward pass
            if img_feats_raw.dim() == 4:
                if img_feats_raw.shape[-1] == self.img_encoder.num_features:
                    # Swin: (B, H, W, C)
                    flat_feats = img_feats_raw.reshape(img_feats_raw.size(0), -1, img_feats_raw.size(-1))
                else:
                    # CNN: (B, C, H, W)
                    flat_feats = img_feats_raw.flatten(2).transpose(1, 2)
            else:
                return None # Unhandled shape
                
            _, seq_len, _ = flat_feats.shape
            
            # Continue forward pass
            img_memory = self.img_proj(flat_feats).view(B, N * seq_len, -1)
            
            feats_to_concat = [self.cls_token.expand(B, -1, -1), img_memory]
            
            if getattr(self, 'use_text', True):
                txt_out = self.text_encoder(input_ids, attention_mask)
                txt_feats = self.text_proj(txt_out.last_hidden_state[:, 0, :]).unsqueeze(1)
                feats_to_concat.append(txt_feats)
            
            if getattr(self, 'use_demo', True):
                demo_feats = self.demo_proj(demographics).unsqueeze(1)
                feats_to_concat.append(demo_feats)
                
            feats = torch.cat(feats_to_concat, dim=1)
            memory = self.fusion(feats)
            
            cls_out = memory[:, 0, :]
            cls_logits = self.classifier(cls_out) # (B, num_classes)
            
            # Grad-CAM: backpropagate from the highest confidence class
            self.zero_grad()
            # We take the max logit for each image in the batch and sum them to allow backward
            target = cls_logits.max(dim=-1)[0].sum()
            target.backward()
            
            # Calculate Grad-CAM
            grad_raw = img_feats_raw.grad
            act_raw = img_feats_raw.detach()
            
            if grad_raw.shape[-1] == self.img_encoder.num_features:
                gradients = grad_raw.permute(0, 3, 1, 2) # (B_img*N, C, H, W)
                activations = act_raw.permute(0, 3, 1, 2) # (B_img*N, C, H, W)
            else:
                gradients = grad_raw
                activations = act_raw
                
            # Global Average Pooling of gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # (B_img*N, C, 1, 1)
            
            # Weighted combination of activations
            cam = torch.sum(weights * activations, dim=1) # (B_img*N, H, W)
            cam = F.relu(cam) # Only positive influence
            
            # Reshape back to (B, N, H, W)
            cam = cam.view(B, N, cam.shape[1], cam.shape[2])
            
            # Normalize to [0, 1]
            cam = cam - cam.amin(dim=(2, 3), keepdim=True)
            cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
            
        # Restore training state
        if is_train:
            self.train()
            
        return cam

if __name__ == "__main__":
    print(">>> TESTING ASSISTANCE V3 (SWIN + HEATMAP)...")
    # Mock Config
    config = {
        'architecture': {
            'image_encoder': 'swin_tiny_patch4_window7_224', 'text_encoder': 'vinai/phobert-base-v2',
            'fusion_dim': 768, 'dropout': 0.1, 'num_layers_fusion': 2, 'num_classes': 14
        }
    }
    
    try:
        model = ASSISTANCE_Model(config)
        print("Model initialized.")
        
        # Mock Inputs
        B, N = 2, 3
        # Use 224x224 because of Swin defaults
        images = torch.randn(B, N, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (B, 128))
        att_mask = torch.ones((B, 128))
        demo = torch.randn(B, 2)
        target_ids = torch.randint(0, 1000, (B, 50))
        
        # Forward
        cls_out, lm_out = model(images, input_ids, att_mask, demo, target_ids)
        print("CLS Output shape:", cls_out.shape) # [2, 14]
        print("LM Output shape:", lm_out.shape)   # [2, 50, 64000]
        
        # Heatmap
        heatmaps = model.get_heatmap(images, input_ids, att_mask, demo)
        print("Heatmap shape:", heatmaps.shape) # [2, 3, 7, 7]
        
        if cls_out.shape == (B, 14) and lm_out.shape == (B, 50, 64000) and heatmaps.shape[2:] == (7, 7):
            print("MODEL TEST PASSED!")
        else:
            print("MODEL OUTPUT SHAPE MISMATCH")
    except Exception as e:
        print(f"MODEL TEST FAILED: {e}")