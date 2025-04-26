import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

class RelationDistiller(nn.Module):
 
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        migrate_components: List[str] = ["Q", "K", "V"],  
        temperature: float = 2.0,
        alpha: float = 0.7
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.migrate_components = migrate_components
        self.temperature = temperature
        self.alpha = alpha
        

        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_relation_loss(self, s_attns, t_attns):
  
        relation_loss = 0.0
        for s_attn, t_attn in zip(s_attns, t_attns):
            s_q = s_attn["query_relation"] if "Q" in self.migrate_components else None
            s_k = s_attn["key_relation"] if "K" in self.migrate_components else None
            s_v = s_attn["value_relation"] if "V" in self.migrate_components else None
            
            t_q = t_attn["query_relation"].detach()
            t_k = t_attn["key_relation"].detach()
            t_v = t_attn["value_relation"].detach()
            
       
            if s_q is not None:
                relation_loss += F.mse_loss(s_q, t_q)
            if s_k is not None:
                relation_loss += F.mse_loss(s_k, t_k)
            if s_v is not None:
                relation_loss += F.mse_loss(s_v, t_v)
            
        return relation_loss / len(s_attns)

    def forward(self, inputs: Dict):
        # 教师模型推理
        with torch.no_grad():
            t_outputs = self.teacher(inputs)
            t_logits = t_outputs["logits"]
            t_attns = t_outputs["attention_relations"]
        
        # 学生模型推理
        s_outputs = self.student(inputs)
        s_logits = s_outputs["logits"]
        s_attns = s_outputs["attention_relations"]
        
        # 计算任务损失
        task_loss = F.cross_entropy(s_logits, inputs["labels"])
        
        # 计算知识蒸馏损失
        kd_loss = F.kl_div(
            F.log_softmax(s_logits / self.temperature, dim=-1),
            F.softmax(t_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        
        # 计算关系迁移损失
        relation_loss = self.compute_relation_loss(s_attns, t_attns)
        
        # 总损失
        total_loss = (
            (1 - self.alpha) * task_loss +
            self.alpha * 0.5 * kd_loss + 
            self.alpha * 0.5 * relation_loss
        )
        
        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "kd_loss": kd_loss,
            "relation_loss": relation_loss,
            "logits": s_logits
        }

class MultiHeadRelation(nn.Module):
    """带关系矩阵提取的多头注意力层"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.n_heads = n_heads
        
    def forward(self, query, key, value):
        attn_output, attn_weights = self.attn(query, key, value)
        
        # 计算关系矩阵（示例实现）
        q = query.reshape(-1, self.n_heads, query.size(-1)//self.n_heads)
        k = key.reshape(-1, self.n_heads, key.size(-1)//self.n_heads)
        v = value.reshape(-1, self.n_heads, value.size(-1)//self.n_heads)
        
        relations = {
            "query_relation": q @ q.transpose(-2, -1),  # Q-Q关系
            "key_relation": k @ k.transpose(-2, -1),    # K-K关系
            "value_relation": v @ v.transpose(-2, -1)   # V-V关系
        }
        
        return attn_output, relations

# 使用示例
if __name__ == "__main__":
    # 初始化模型
    teacher = ProtBertWithRelations()  # 需要实现attention_relations输出
    student = StudentModelWithRelations()  # 需要实现attention_relations输出
    
    # 配置不同蒸馏策略（对应表格中的不同方法）
    configs = {
        "Q-Q+K-K+V-V": ["Q", "K", "V"],
        "w.o.Q-Q": ["K", "V"],
        "w.o.K-K": ["Q", "V"],
        "w.o.V-V": ["Q", "K"],
        "Q-K+V-V": ["Q", "K", "V"]  # 需要特殊处理
    }
    
    # 训练不同配置
    for method, components in configs.items():
        print(f"Training {method}...")
        distiller = RelationDistiller(
            student=student,
            teacher=teacher,
            migrate_components=components,
            temperature=3.0,
            alpha=0.7
        )
        
        # 训练循环（需要实现train_loader和optimizer）
        for epoch in range(10):
            for batch in train_loader:
                outputs = distiller(batch)
                outputs["total_loss"].backward()
                optimizer.step()
                optimizer.zero_grad()
                
        # 评估并保存结果
        evaluate(distiller, test_loader, method)