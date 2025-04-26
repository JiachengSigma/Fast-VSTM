import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np



class LoRALayer(nn.Module):
    """LoRA层实现（支持动态rank和自适应）"""
    def __init__(self, in_dim, out_dim, rank=8, adaptive=False):
        super().__init__()
        self.adaptive = adaptive
        
        # 原始权重（冻结）
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.weight.requires_grad = False
        
        # LoRA参数
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        
        # 自适应参数
        if adaptive:
            self.importance = nn.Parameter(torch.ones(rank))
            self.threshold = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        base = self.linear(x)
        lora = (x @ self.lora_A)
        
        if self.adaptive:
            mask = (self.importance > self.threshold).float()
            lora = lora * mask
        
        lora = lora @ self.lora_B
        return base + lora

class Adapter(nn.Module):
    """适配器模块（支持共享）"""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.down = nn.Linear(dim, dim//reduction)
        self.up = nn.Linear(dim//reduction, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


   

# --------------------------
# SVD 微调核心模块
# --------------------------

class SVDLinear(nn.Module):
    """基于SVD分解的低秩线性层"""
    def __init__(self, in_features, out_features, rank_ratio=0.2):
        super().__init__()
        self.rank = int(min(in_features, out_features) * rank_ratio)
        
        # 原始权重矩阵的SVD分解参数
        self.U = nn.Parameter(torch.Tensor(in_features, self.rank))
        self.S = nn.Parameter(torch.Tensor(self.rank))
        self.V = nn.Parameter(torch.Tensor(self.rank, out_features))
        
        # 残差连接系数
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.uniform_(self.S, 0.9, 1.1)  # 初始奇异值接近1
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

    def forward(self, x):
        # 重建权重矩阵
        W = self.U @ torch.diag(self.S) @ self.V
        return F.linear(x, W) + self.alpha * x  # 残差连接

def apply_svd_to_model(model, rank_ratio=0.2, layer_pattern="encoder.layer.[0-9]+.attention"):
    """
    将指定层的线性层替换为SVDLinear
    :param layer_pattern: 正则表达式匹配目标层
    """
    pattern = re.compile(layer_pattern)
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and pattern.match(name):
            # 替换线性层
            svd_layer = SVDLinear(
                module.in_features,
                module.out_features,
                rank_ratio=rank_ratio
            )
            
            # 从原始权重初始化
            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(module.weight.data)
                svd_layer.U.data = U[:, :svd_layer.rank]
                svd_layer.S.data = S[:svd_layer.rank]
                svd_layer.V.data = Vh[:svd_layer.rank, :].t()
            
            setattr(model, name, svd_layer)
        else:
            # 递归处理子模块
            apply_svd_to_model(module, rank_ratio, layer_pattern)
    

# --------------------------
# 修改微调应用函数
# --------------------------

def apply_finetune_method(model, method="full", **kwargs):
    """应用微调策略到模型"""
    if method == "full":
        # 全参数微调
        for param in model.parameters():
            param.requires_grad = True

    elif method.startswith("svd"):
        # SVD微调
        rank_ratio = float(method.split("_")[1]) if "_" in method else 0.2
        apply_svd_to_model(model, rank_ratio=rank_ratio)
        
        # 冻结非SVD参数
        for name, param in model.named_parameters():
            if "U" not in name and "S" not in name and "V" not in name:
                param.requires_grad = False

    # 原有其他方法保持不变...
    elif method == "bitfit":
        # 仅微调偏置项
        for name, param in model.named_parameters():
            if "bias" not in name:
                param.requires_grad = False

    # 其他方法...

# --------------------------
# 训练器增强
# --------------------------

class SVDAwareTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, config):
        super().__init__(model, train_loader, val_loader, config)
        
        # SVD特定优化器配置
        if config.get("svd_optimizer", "adamw") == "sgd":
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config["lr"],
                momentum=0.9,
                weight_decay=config["weight_decay"]
            )
            
        # 奇异值正则化
        self.svd_reg = config.get("svd_reg", 0.01)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            inputs = {k: v.to(self.config["device"]) for k, v in batch.items()}
            labels = inputs.pop("labels")

            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            
            # 基础损失
            loss = self.criterion(outputs.logits, labels)
            
            # SVD正则化项
            reg_loss = 0
            for name, param in self.model.named_parameters():
                if "S" in name:  # 仅正则化奇异值
                    reg_loss += torch.norm(param, p=2)
            
            total_loss += (loss + self.svd_reg * reg_loss).item()
            
            (loss + self.svd_reg * reg_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
            self.optimizer.step()

        self.scheduler.step()
        return total_loss / len(self.train_loader)

# --------------------------
# 配置示例
# --------------------------
"""
# config/svd_config.yaml
svd:
  rank_ratio: 0.3
  target_layers: "encoder.layer.[0-9]+.attention"
  optimizer: "adamw"
  svd_reg: 0.01

training:
  lr: 1e-5
  epochs: 50
  batch_size: 32
"""

# --------------------------
# 使用示例
# --------------------------
if __name__ == "__main__":
    # SVD微调示例
    model = AutoModelForSequenceClassification.from_pretrained("Rostlab/prot_bert")
    model = apply_finetune_method(model, method="svd_0.3")
    
    # 检查可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params/1e6:.2f}M")  # 约3.2M参数

    # 特殊训练器
    trainer = SVDAwareTrainer(model, train_loader, val_loader, config)
    trainer.train()

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"]
        )
        
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            inputs = {k: v.to(self.config["device"]) for k, v in batch.items()}
            labels = inputs.pop("labels")
            
            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = self.criterion(outputs.logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["grad_clip"]
            )
            
            self.optimizer.step()
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = {k: v.to(self.config["device"]) for k, v in batch.items()}
                labels = inputs.pop("labels")
                
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        auroc = roc_auc_score(all_labels, all_preds)
        aupr = average_precision_score(all_labels, all_preds)
        return {"auroc": auroc, "aupr": aupr}

    def train(self):
        best_aupr = 0
        for epoch in range(self.config["epochs"]):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate()
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val AUROC: {val_metrics['auroc']:.4f} | Val AUPR: {val_metrics['aupr']:.4f}")
            
            if val_metrics["aupr"] > best_aupr:
                best_aupr = val_metrics["aupr"]
                torch.save(self.model.state_dict(), f"best_{self.config['method']}.pt")

# --------------------------
# 实验配置和运行
# --------------------------

if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据加载（需实现具体数据集）
    train_dataset = ProteinDataset(split="train")
    val_dataset = ProteinDataset(split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # 实验配置
    methods = [
        "full", "bitfit", "lora_8", "lora_16",
        "lora_32", "adalora", "padapter", "hadapter"
    ]
    
    results = []
    
    for method in methods:
        print(f"\n===== Training {method} =====")
        
        # 初始化基础模型
        model = AutoModelForSequenceClassification.from_pretrained("Rostlab/prot_bert")
        model = apply_finetune_method(model, method).to(device)
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_params/1e6:.2f}M")
        
        # 训练配置
        config = {
            "device": device,
            "epochs": 10,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "method": method
        }
        
        # 训练和评估
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
        
        # 记录最终结果
        final_metrics = trainer.evaluate()
        results.append({
            "Method": method,
            "Params (M)": f"{trainable_params/1e6:.2f}",
            **final_metrics
        })
    
    # 生成结果表格
    print("\nFinal Results:")
    print("{:<15} {:<12} {:<8} {:<8}".format("Method", "Params(M)", "AUROC", "AUPR"))
    for res in results:
        print("{:<15} {:<12} {:<8.4f} {:<8.4f}".format(
            res["Method"], res["Params (M)"], res["auroc"], res["aupr"]
        ))