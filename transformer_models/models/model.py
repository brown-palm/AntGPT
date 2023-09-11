import torch
from torch import nn
from torch.distributions import Categorical
import math
from ..parser import parse_args, load_config
from pytorch_lightning import seed_everything
from .slowfast import SlowFast


class DropToken(nn.Module):
    def __init__(self, dim, drop_prob) -> None:
        super().__init__()
        self.dim = dim  # (D, )
        self.pad = torch.nn.parameter.Parameter(torch.randn(dim))  # (D)
        # TODO: figure out initialization
        # self.pad = torch.nn.parameter.Parameter(torch.zeros(dim))  # (D)
        self.drop_prob = drop_prob

    def forward(self, x):
        # x: (..., D)
        if self.training:
            input_shape = x.shape
            x = torch.reshape(x, (-1, input_shape[-1]))
            rand_tensor = torch.rand(x.shape[0], device=x.device)
            x[rand_tensor < self.drop_prob, :] = self.pad
            x = torch.reshape(x, input_shape)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer("pe", pe)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[: x.size(0), :, :]
        return self.dropout(x)    

class MMPositionalEncoding(nn.Module):

    def __init__(self, cfg, d_model, num_input_clips=1, dropout=0.1, max_len=1000,
                use_text=False, use_img=True):
        super(MMPositionalEncoding, self).__init__()
        self.cfg = cfg
        self.num_input_clips = num_input_clips
        self.d_model = d_model
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer("pe", pe)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

        modality_tokens_init = torch.randn((6, d_model))  # (6, D)
        self.modality_tokens = torch.nn.parameter.Parameter(modality_tokens_init)
        rel_pe_init = torch.zeros((cfg.data.image.num_images_per_segment, 1, d_model))  # (L, 1, D)
        if use_img:
            self.rel_pe = torch.nn.parameter.Parameter(rel_pe_init)
        else:
            self.rel_pe = torch.nn.parameter.Parameter(rel_pe_init, requires_grad=False)

    def forward(self, text_feat=None, img_feat=None, pred_text_feat=None, query_feat=None):

        num_input_clips = self.num_input_clips
        num_img_per_clip = self.cfg.data.image.num_images_per_segment

        abs_pe = torch.flip(self.pe, dims=[0])

        if query_feat is not None:
            len_query_feat = query_feat.shape[0]
            abs_pe_pos = self.pe[:len_query_feat, :, :]
            # query_feat = query_feat + abs_pe_pos + self.modality_tokens[5]
            query_feat = query_feat + self.modality_tokens[5]
            query_feat = self.layernorm(query_feat)
            query_feat = self.dropout(query_feat)
            abs_pe = abs_pe[len_query_feat:]
        if text_feat is not None:
            pass
            #num_segments = len(text_feat)
            #text_feat = text_feat + abs_pe[:num_segments] + self.modality_tokens[1]
            #text_feat = self.layernorm(text_feat)
            #text_feat = self.dropout(text_feat)
        if img_feat is not None:
            # img_feat: (num_input_clips * num_img_per_clip, B, D)
            # abs_pe: (num_input_clips, 1, D)
            # rel_pe: (num_img_per_clip, 1, D)
            batch_size, D = img_feat.shape[1], img_feat.shape[-1]
            # (num_input_clips, num_img_per_clip, B, D)
            img_feat = torch.reshape(img_feat, (-1, num_img_per_clip, batch_size, D))
            num_input_clips_images = len(img_feat)
            # (num_input_clips, num_img_per_clip, B, D)
            img_feat = img_feat + torch.unsqueeze(abs_pe[:num_input_clips], 1) + self.rel_pe + self.modality_tokens[2]
            img_feat = torch.reshape(img_feat, (num_input_clips_images * num_img_per_clip, batch_size, -1))
            img_feat = self.layernorm(img_feat)
            img_feat = self.dropout(img_feat)
        if pred_text_feat is not None:
            len_pred_text_feat = pred_text_feat.shape[0]
            abs_pe_pos = self.pe[num_input_clips:num_input_clips + len_pred_text_feat, :, :]
            pred_text_feat = pred_text_feat + abs_pe_pos + self.modality_tokens[4]
            # pred_text_feat = pred_text_feat + self.modality_tokens[3]
            pred_text_feat = self.layernorm(pred_text_feat)
            pred_text_feat = self.dropout(pred_text_feat)

        return text_feat, img_feat, pred_text_feat, query_feat

# --------------------------------------------------------------------#

class PredictiveTransformerEncoder(nn.Module):
    def __init__(self, cfg, num_queries):
        super().__init__()
        #seed_everything(cfg.seed)
        self.cfg = cfg
        dim_in = cfg.model.base_feat_size
        num_heads = cfg.model.pte.num_heads
        num_layers = cfg.model.pte.num_layers
        self.num_queries = num_queries

        # TODO: check initialization
        self.queries = torch.nn.parameter.Parameter(torch.randn((cfg.model.num_actions_to_predict, dim_in)))  # (Z, D)
        # self.queries = torch.nn.parameter.Parameter(torch.zeros((num_queries, dim_in)))  # (Z, D)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_in, num_heads, dropout=cfg.model.pte.enc_dropout),
            num_layers,
            norm=nn.LayerNorm(dim_in),
        )
        self.pos_encoder = MMPositionalEncoding(
            cfg, dim_in, num_input_clips=self.cfg.data.input_segments[1]-self.cfg.data.input_segments[0], 
            dropout=cfg.model.pte.pos_dropout, use_text=self.cfg.data.use_gt_text, use_img=cfg.model.img_feat_size > 0)

    def forward(self, text_features, image_features, pred_text_features, mask_text, mask_image, mask_pred_text, train=True):
        # clip_features, image_features, object_features: (B, L, D)  L could be different
        if text_features is not None:
            batch_size, num_inputs, _ = text_features.shape
            text_features = torch.transpose(text_features, 0, 1)   # (num_inputs, B, D)
        if image_features is not None:
            batch_size, num_inputs, _ = image_features.shape
            image_features = torch.transpose(image_features, 0, 1)   # (num_inputs, B, D)
        if pred_text_features is not None:
            batch_size, num_inputs, _ = pred_text_features.shape
            pred_text_features = torch.transpose(pred_text_features, 0, 1)   # (Z, B, D)

        queries = self.queries.expand(batch_size, -1, -1).permute(1, 0, 2)  # (Z, B, D)

        text_features, image_features, pred_text_features, queries = self.pos_encoder(text_features, image_features, pred_text_features, queries)
        x = torch.cat([feat for feat in [text_features, pred_text_features, image_features, queries] if feat is not None], dim=0)
        mask_query = torch.zeros((batch_size, self.num_queries), dtype=torch.bool, device=queries.device)  # (B, Z)
        mask = torch.cat([feat for feat in [mask_text, mask_pred_text, mask_image, mask_query] if feat is not None], dim=-1)
        x = self.encoder(x, src_key_padding_mask=mask, is_causal=False)  # (num_inputs + Z, B, D)
        if self.cfg.model.autoregressive:
            if self.cfg.model.teacherforcing and train:
                x = x[-self.cfg.model.num_actions_to_predict:, ...]  # (Z, B, D)
            else:
                seq = x
                for _ in range(self.cfg.model.total_actions_to_predict):
                    next = self.encoder(x, src_key_padding_mask=mask, is_causal=True)[-1:, :, :]  # (1, B, D)
                    x = torch.cat([x, next], dim=0)[1:,:,:]
                    seq = torch.cat([seq, next], dim=0)
                x = seq[-self.cfg.model.total_actions_to_predict:, ...]  # (Z, B, D)
        else:
            x = x[-self.cfg.model.total_actions_to_predict:, ...]  # (Z, B, D)
        return torch.transpose(x, 0, 1)  # (B, Z, D)
    
# --------------------------------------------------------------------#


class MLPDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.head = nn.Linear(cfg.model.base_feat_size, sum(cfg.model.num_classes))
    
    def forward(self, x):
        # x: (B, Z, D)
        logits = self.head(x)  # (B, Z, #verbs + #nouns)
        logits = torch.split(logits, self.cfg.model.num_classes, dim=-1)  # [(B, Z, #verbs), (B, Z, #nouns)]
        return logits  
    
# --------------------------------------------------------------------#


class ClassificationModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #seed_everything(cfg.seed)
        self.cfg = cfg
        self.backbone = SlowFast(cfg, with_head=True, num_classes=cfg.model.base_feat_size, head_dropout_rate=0)
        if cfg.data.use_gt_text:
            self.build_text_encoder()
        if cfg.data.use_goal:
            self.build_text_feat_proj()
        if cfg.model.img_feat_size > 0:
            self.build_img_feat_proj()
        self.build_aggregator()
        self.build_decoder()

    def build_text_encoder(self):
        emb_size = self.cfg.model.text_feat_size
        self.embedding_layer_verb = nn.Embedding(self.cfg.model.num_classes[0] + 1, emb_size)
        self.embedding_layer_noun = nn.Embedding(self.cfg.model.num_classes[1] + 1, emb_size)
        self.text_proj = nn.Linear(emb_size+emb_size, self.cfg.model.base_feat_size)
        if self.cfg.model.drop_text > 0:
            self.drop_text = DropToken(self.cfg.model.text_feat_size * len(self.cfg.model.num_classes), self.cfg.model.drop_text)

    def build_img_feat_proj(self):
        proj_size = self.cfg.model.base_feat_size
        self.img_feat_proj = nn.Linear(self.cfg.model.img_feat_size, proj_size)
        if self.cfg.model.drop_img > 0:
            self.drop_img = DropToken(self.cfg.model.img_feat_size, self.cfg.model.drop_img)

    def build_text_feat_proj(self):
        proj_size = self.cfg.model.base_feat_size
        self.text_feat_proj = nn.Linear(self.cfg.model.text_feat_size, proj_size)
        
    def build_aggregator(self):
        cfg = self.cfg
        aggregator = None
        
        if cfg.model.aggregator == 'pte':
            aggregator = PredictiveTransformerEncoder(cfg, num_queries=cfg.model.num_actions_to_predict)
        elif cfg.model.aggregator == 'trf':
            aggregator = PredictiveTransformerEncoder(cfg, num_queries=1)
        else:
            raise NotImplementedError(f"aggregator {cfg.model.aggregator} not supported")
        self.aggregator = aggregator
    
    def build_decoder(self):
        cfg = self.cfg
        decoder = None
        if cfg.model.decoder == 'mlp':
            decoder = MLPDecoder(cfg)
        else:
            raise NotImplementedError(f"decoder {cfg.model.decoder} not supported")
        self.decoder = decoder

    def encode_text(self, texts):
        verb_features = self.embedding_layer_verb(texts[..., 0])  # (B, num_segments, text_feat_size) or (B, num_segments, num_seqs, text_feat_size)
        if len(verb_features.shape) == 4:
            verb_features = verb_features.sum(dim=-2)  # (B, num_segments, text_feat_size)
        noun_features = self.embedding_layer_noun(texts[..., 1])  # (B, num_segments, text_feat_size) or (B, num_segments, num_seqs, text_feat_size)
        if len(noun_features.shape) == 4:
            noun_features = noun_features.sum(dim=-2)  # (B, num_segments, text_feat_size)
        text_features = torch.cat([verb_features, noun_features], dim=-1)   # (B, num_segments, text_feat_size * 2)
        if self.cfg.model.drop_text > 0:
            text_features = self.drop_text(text_features)
        text_features = self.text_proj(text_features)  # (B, num_segments, base_feat_size)
        return text_features
    
    def encode_text_feature(self, texts):
        text_features = self.text_feat_proj(texts.to(dtype=torch.float32))
        return text_features
    
    def encode_image_features(self, x):
        if self.cfg.model.drop_img > 0:
            x = self.drop_img(x)
        out = self.img_feat_proj(x)  # (B, N, D)
        return out

    def aggregate(self, text_features, image_features, pred_text_features, mask_text, mask_image, mask_pred_text):
        return self.aggregator(text_features, image_features, pred_text_features, mask_text, mask_image, mask_pred_text)

    def decode(self, features):
        return self.decoder(features)

    def forward(self, texts, image_features, pred_text, mask_text, mask_image, mask_pred_text):
        text_features = self.encode_text(texts) if texts is not None else None
        #text_features = self.encode_text_feature(texts) if texts is not None else None
        if image_features is not None:
            image_features = self.encode_image_features(image_features)
        pred_text_features = self.encode_text_feature(pred_text) if pred_text is not None else None
        features = self.aggregate(text_features, image_features, pred_text_features, mask_text, mask_image, mask_pred_text)  # (B, ?, D) Might want to change the order of features to be totally consistent
        x = self.decode(features)  # [(B, Z, #verbs), (B, Z, #nouns)]
        return x
    

    def generate(self, logits, k=1):
        x = logits # [(B, Z, C), (B, Z, C)]

        def match(v, n):
            if f'{v}_{n}' in self.vocab:
                return True
            return False

        results_all = {}  # sampling_method --> results

        for sampling_method in self.cfg.model.sampleing_method:
            results = []
            if sampling_method == 'naive':
                for head_x in x:
                    if k > 1:
                        preds_dist = Categorical(
                            logits=torch.clamp(head_x, min=0))
                        preds = [preds_dist.sample()
                                 for _ in range(k)]  # [(B, Z)] * K
                    elif k == 1:
                        preds = [head_x.argmax(2)]
                    head_x = torch.stack(preds, dim=1)  # (B, K, Z)
                    results.append(head_x)
            elif sampling_method == 'action_sample':
                num_tries = 20
                [head_verb, head_noun] = x
                batch_size, Z = head_verb.shape[0], head_verb.shape[1]
                preds_verb_dist = Categorical(
                    logits=torch.clamp(head_verb, min=0))
                preds_noun_dist = Categorical(
                    logits=torch.clamp(head_noun, min=0))
                verb_sampled = [preds_verb_dist.sample()
                                for _ in range(num_tries)]  # (B, Z) * num_tries
                noun_sampled = [preds_noun_dist.sample()
                                for _ in range(num_tries)]  # (B, Z) * num_tries
                verb_sampled = torch.stack(
                    verb_sampled, dim=0)  # (num_tries, B, Z)
                noun_sampled = torch.stack(
                    noun_sampled, dim=0)  # (num_tries, B, Z)
                verb_sampled_k = verb_sampled[:k, ...]  # (K, B, Z)
                noun_sampled_k = noun_sampled[:k, ...]  # (K, B, Z)
                matched_num = torch.zeros_like(
                    verb_sampled_k[0, ...])  # (B, Z)
                for i in range(num_tries):
                    for b in range(batch_size):
                        for z in range(Z):
                            # Do not use k after this loop
                            verb_idx, noun_idx = verb_sampled[i,
                                                              b, z], noun_sampled[i, b, z]
                            if not match(verb_idx, noun_idx):
                                continue
                            if matched_num[b, z] >= k:
                                continue
                            verb_sampled_k[matched_num[b, z], b, z] = verb_idx
                            noun_sampled_k[matched_num[b, z], b, z] = noun_idx
                            matched_num[b, z] += 1
                results = [verb_sampled_k.permute(1, 0, 2), noun_sampled_k.permute(
                    1, 0, 2)]  # (B, K, Z), (B, K, Z)
                # print(torch.mean(matched_num.to(torch.float)))
            elif sampling_method == 'action_max':
                num_tries = 5
                [head_verb, head_noun] = x  # (B, Z, C), (B, Z, C)
                batch_size, Z, _ = head_verb.shape
                pred_verb_score, pred_verb_idx = torch.sort(
                    head_verb, descending=True)  # (B, Z, num_verbs)
                pred_noun_score, pred_noun_idx = torch.sort(
                    head_noun, descending=True)  # (B, Z, num_nouns)
                # (B, Z, num_tries)
                pred_verb_score, pred_verb_idx = pred_verb_score[...,
                                                                 :num_tries], pred_verb_idx[..., :num_tries]
                # (B, Z, num_tries)
                pred_noun_score, pred_noun_idx = pred_noun_score[...,
                                                                 :num_tries], pred_noun_idx[..., :num_tries]
                verb_sampled, noun_sampled = torch.zeros((batch_size, Z, k), device=head_verb.device), torch.zeros(
                    (batch_size, Z, k), device=head_verb.device)  # (B, Z, K)
                for b in range(batch_size):
                    for z in range(Z):
                        info = []
                        for t1 in range(num_tries):
                            verb_idx = pred_verb_idx[b, z, t1]
                            verb_score = pred_verb_score[b, z, t1]
                            for t2 in range(num_tries):
                                noun_idx = pred_noun_idx[b, z, t2]
                                noun_score = pred_noun_score[b, z, t2]
                                if match(verb_idx, noun_idx):
                                    info.append(
                                        [verb_idx, noun_idx, 1e4 + verb_score*noun_score])
                                else:
                                    info.append(
                                        [verb_idx, noun_idx, verb_score*noun_score])
                        info.sort(key=lambda x: x[-1], reverse=True)
                        for i in range(k):
                            verb_sampled[b, z, i] = info[i][0]
                            noun_sampled[b, z, i] = info[i][1]
                results = [verb_sampled.permute(0, 2, 1), noun_sampled.permute(
                    0, 2, 1)]  # (B, K, Z), (B, K, Z)
            else:
                raise NotImplementedError(
                    f'sampling method {self.cfg.model.sampling_method} not implemented')
            results_all[sampling_method] = results
        return results_all, x

# --------------------------------------------------------------------#

def sanity_check():
    args = parse_args()
    cfg = load_config(args)
    cm = ClassificationModule(cfg)
    
if __name__ == '__main__':
    sanity_check()
