import copy
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gan.network.loss import get_losses
from gan.utils.builder import Builders
from llm_labeling.train_expression_regressor import ExpressionTransformer, MASKED_LOGIT_SENTINEL
from mamba.mamba import Mamba, MambaConfig

class NetG_Mamba(nn.Module):
    def __init__(
        self,
        n_chars,
        latent_size,
        seq_len,
        d_model,
        n_layers,
        expand_factor=1,
        d_state=8,
        dt_rank='auto',
        d_conv=2,
        pscan=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.latent2seq = nn.Linear(latent_size, seq_len * d_model)
        config = MambaConfig(
            d_model=d_model,
            n_layers=n_layers,
            expand_factor=expand_factor,
            d_state=d_state,
            dt_rank=dt_rank,
            d_conv=d_conv,
            pscan=pscan,
        )
        self.mamba = Mamba(config)
        self.out_proj = nn.Linear(d_model, n_chars)
    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    def forward(self, z):
        # z: (batch, latent_size)
        x = self.latent2seq(z).view(-1, self.seq_len, self.mamba.config.d_model)
        x = self.mamba(x)  # (batch, seq_len, d_model)
        x = self.out_proj(x)  # (batch, seq_len, n_chars)
        return x

class NetG_DCGAN(nn.Module):
    def __init__(
            self, 
            n_chars:int,
            latent_size: int, 
            seq_len: int,
            hidden: int,

        ):
        super().__init__()
        assert seq_len == 20
        use_bias=True
        self.linear=nn.Linear(latent_size,6*384)
        self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(384,256,(6,1),(1,1),bias=use_bias),#[5, 256, 47, 1]
                    nn.BatchNorm2d(256),nn.ReLU(),
                    nn.ConvTranspose2d(256,192,(5,1),(1,1),bias=use_bias),#[5, 192, 100, 1]
                    nn.BatchNorm2d(192),nn.ReLU(),
                    nn.ConvTranspose2d(192,128,(6,1),(1,1),bias=use_bias),#[5, 128, 205, 1]
                    nn.BatchNorm2d(128),nn.ReLU(),
                )
        self.conv=nn.Sequential(
                    nn.ZeroPad2d((0,0,4,3)),#[5, 128, 212, 1]
                    nn.Conv2d(128,128,(8,1),(1,1),0,bias=use_bias),#[5, 128, 205, 1]
                    nn.BatchNorm2d(128),nn.ReLU(),
                    nn.ZeroPad2d((0,0,4,3)),#[5, 128, 212, 1]
                    nn.Conv2d(128,64,(8,1),(1,1),0,bias=use_bias),#[5, 64, 205, 1]
                    nn.BatchNorm2d(64),nn.ReLU(),
                    nn.ZeroPad2d((0,0,4,3)),#[5, 64, 212, 1]
                    nn.Conv2d(64,n_chars,(8,1),(1,1),0,bias=use_bias),#[5, 4, 205, 1]
        #             nn.BatchNorm2d(4)
        )
        
    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0],384,6,1)
        x = self.deconv(x)
        x = self.conv(x)
        x = x.view(x.shape[0],x.shape[2],x.shape[1])
        return x#(bs, seq_len, 48)
    
class NetG_Lstm(nn.Module):
    def __init__(
        self,
        n_chars:int,
        n_layers: int,
        d_model: int,
        dropout: float,
        seq_len: int,
        potential_size: int,
    ):
        super().__init__()
        self.n_chars = n_chars
        self.max_len = seq_len
        self.n_layers = n_layers
        self.d_model = d_model

        self.fc_h = nn.Sequential(
            nn.Linear(potential_size,n_layers*d_model),nn.ReLU()
        )
        self.fc_c = nn.Sequential(
            nn.Linear(potential_size,n_layers*d_model),nn.ReLU()
        )
        self.emb = nn.Embedding(n_chars + 1, d_model, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size = d_model,
            hidden_size = d_model,
            num_layers = n_layers,
            batch_first = True,
            dropout = dropout
        )
        self.fc = nn.Linear(d_model,n_chars)
        self.start_token_idx = n_chars

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self,z):
        # z: (batch_size, potential_size)
        bs = z.shape[0]
        h = self.fc_h(z).view(bs, self.n_layers, self.d_model).permute(1, 0, 2).contiguous()
        c = self.fc_c(z).view(bs, self.n_layers, self.d_model).permute(1, 0, 2).contiguous()

        input_step = torch.full(
            (bs,),
            fill_value=self.start_token_idx,
            dtype=torch.long,
            device=z.device,
        )
        logits = []
        for _ in range(self.max_len):
            embedded = self.emb(input_step).unsqueeze(1)
            output, (h, c) = self.rnn(embedded, (h, c))
            step_logits = self.fc(output.squeeze(1))
            logits.append(step_logits)

            # Autoregressive teacher-free decoding.
            input_step = step_logits.detach().argmax(dim=1)

        return torch.stack(logits, dim=1)

class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)

class NetG_CNN(nn.Module):
    def __init__(self, n_chars, latent_size,seq_len , hidden):
        super( ).__init__()
        self.fc1 = nn.Linear(latent_size, hidden*seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, n_chars, 1)
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.hidden = hidden

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, noise):
        batch_size = noise.size(0)
        output = self.fc1(noise)
        output = output.view(-1, self.hidden, self.seq_len) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(batch_size*self.seq_len, -1)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))

class ExpressionScorePredictor:
    """Wrapper that loads the trained regressor and exposes a predict method."""

    def __init__(self, checkpoint_path: Union[Path, str], device: Union[str, torch.device]):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_cfg = checkpoint["config"]
        self.seq_len = int(model_cfg["seq_len"])
        self.input_dim = int(model_cfg["input_dim"])
        self.model = ExpressionTransformer(
            seq_len=self.seq_len,
            input_dim=self.input_dim,
            d_model=int(model_cfg["d_model"]),
            n_heads=int(model_cfg["n_heads"]),
            num_layers=int(model_cfg["num_layers"]),
            ff_multiplier=int(model_cfg["ff_multiplier"]),
            dropout=float(model_cfg["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def _build_features(self, masked_logits: torch.Tensor, onehot_tensor: torch.Tensor) -> torch.Tensor:
        sanitized = torch.where(
            masked_logits <= MASKED_LOGIT_SENTINEL,
            torch.zeros_like(masked_logits),
            masked_logits,
        )
        sanitized = sanitized.to(dtype=torch.float32)
        onehot_tensor = onehot_tensor.to(dtype=torch.float32)
        features = torch.cat([sanitized, onehot_tensor], dim=-1)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if features.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {features.size(1)}")
        if features.size(2) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {features.size(2)}")
        return features

    def predict(self, masked_logits: torch.Tensor, onehot_tensor: torch.Tensor) -> torch.Tensor:
        features = self._build_features(masked_logits, onehot_tensor)
        with torch.no_grad():
            preds = self.model(features.to(self.device))
        return preds.detach()

def train_network_generator(netG, netM, netP, score_predictor: ExpressionScorePredictor, cfg, data, target, current_round, random_method, metric, lr, n_actions):
    opt = torch.optim.Adam(netG.parameters(),lr=lr)
    best_weights = None
    best_score = -float('inf')
    patience_counter = 0
    z1 = torch.zeros([cfg.batch_size,cfg.potential_size]).to(cfg.device)
    z2 = torch.zeros([cfg.batch_size,cfg.potential_size]).to(cfg.device)

    netM.eval()
    netP.eval()
    
    empty_blds = None
    best_str_to_print = ''

    for epoch in range(cfg.num_epochs_g):
        netG.train()
        opt.zero_grad()
        z1 = random_method(z1)
        z2 = random_method(z2)
        logit_raw_1 =netG(z1)#（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
        logit_raw_2 =netG(z2)

        masked_x_1,masks_1,blds_1= netM(logit_raw_1)
        masked_x_2,masks_2,blds_2= netM(logit_raw_2)

        valid_mask_1 = torch.tensor([builder.is_valid() for builder in blds_1.builders], device=cfg.device, dtype=torch.bool)
        valid_mask_2 = torch.tensor([builder.is_valid() for builder in blds_2.builders], device=cfg.device, dtype=torch.bool)

        onehot_tensor_1 = F.gumbel_softmax(masked_x_1,hard=True)
        pred_1,latent_1 = netP(onehot_tensor_1,latent=True)

        onehot_tensor_2 = F.gumbel_softmax(masked_x_2,hard=True)
        pred_2,latent_2 = netP(onehot_tensor_2,latent=True)
        with torch.no_grad():
            score_pred_1 = score_predictor.predict(masked_x_1, onehot_tensor_1).to(cfg.device).float()
            score_pred_2 = score_predictor.predict(masked_x_2, onehot_tensor_2).to(cfg.device).float()

        complexity_pred_parts = []
        if valid_mask_1.any():
            complexity_pred_parts.append(score_pred_1[valid_mask_1])
        if valid_mask_2.any():
            complexity_pred_parts.append(score_pred_2[valid_mask_2])

        if complexity_pred_parts:
            complexity_predicted = torch.cat(complexity_pred_parts)
            complexity_mean = complexity_predicted.mean()
        else:
            complexity_predicted = torch.zeros(0, device=cfg.device, dtype=torch.float32)
            complexity_mean = torch.zeros((), device=cfg.device, dtype=torch.float32)

        loss_inputs = {
            'logit_raw_1':logit_raw_1,
            'logit_raw_2':logit_raw_2,
            'masked_x_1':masked_x_1,
            'masked_x_2':masked_x_2,
            'masks_1':masks_1,
            'masks_2':masks_2,
            'blds_1':blds_1,
            'blds_2':blds_2,
            'z1':z1,
            'z2':z2,
            'onehot_tensor_1':onehot_tensor_1,
            'onehot_tensor_2':onehot_tensor_2,
            'pred_1':pred_1,
            'pred_2':pred_2,
            'latent_1':latent_1,
            'latent_2':latent_2,
            'complexity_scores':complexity_predicted,
            'complexity_mean':complexity_mean,
        }
        loss = get_losses(loss_inputs,cfg)

        blds:Builders = blds_1+blds_2
        blds.drop_invalid()
        blds.evaluate(data,target,metric)
        n_valid_train = blds.batch_size

        str_to_print = f"##{epoch}/{cfg.num_epochs_g} : n_valid_train:{n_valid_train}, n_valid:{len(blds.scores)}, loss:{loss:.4f}"
        mean_score = np.mean(blds.scores)
        max_score = np.max(blds.scores) if len(blds.scores)>0 else 0
        std_score = np.std(blds.scores) if len(blds.scores)>0 else 0
        str_to_print += f", max_score:{max_score:.4f},   mean_score:{mean_score:.4f}, std_score:{std_score:.4f}"
        blds.drop_duplicated()
        str_to_print += f",unique:{blds.batch_size}"
        complexity_val_model = float(complexity_mean.detach().item()) if complexity_predicted.numel() else 0.0
        str_to_print += f", financial_logic:{complexity_val_model:.3f}"
        print(str_to_print)
        if max_score>0:
            exprs = blds.exprs_str[np.argmax(blds.scores)]
            print(f"Max score {max_score} expr: {exprs}")
        # save_blds(blds,f"out/{cfg.name}/train/{current_round}",epoch)

        if empty_blds is None:
            empty_blds = blds
        else:
            empty_blds = empty_blds + blds

        
        if cfg.g_es_score == 'mean':
            es_score = mean_score
        elif cfg.g_es_score == 'max':
            es_score = max_score
        elif cfg.g_es_score == 'combined':
            es_score = max_score + 2. *  std_score
        else:
            raise NotImplementedError
        
        if es_score > best_score:
            best_score = es_score
            best_weights = copy.deepcopy(netG.state_dict())
            best_str_to_print = str_to_print
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > cfg.g_es:
                print(f'Early stopping triggered at epoch {epoch}, {best_score} !')
                
                break
        
        if epoch>0:
            loss.backward()
            opt.step()

    if best_weights is not None:
        print('load_best_weights')
        netG.load_state_dict(best_weights)
        print(best_str_to_print)

    empty_blds.drop_duplicated()
    return empty_blds