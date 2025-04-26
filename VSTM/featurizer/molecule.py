from pathlib import Path
import typing as T

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.nn import ModuleList
from torch.nn.functional import one_hot

from ..utils import canonicalize, get_logger
from .base import Featurizer

# from mol2vec.features import (
#     mol2alt_sentence,
#     mol2sentence,
#     MolSentence,
#     sentences2vec,
# )
# from gensim.models import word2vec
# import dgl
# from dgl.dataloading import GraphDataLoader
# from dgl.nn import GraphConv, GATConv, SAGEConv, SGConv, TAGConv
# from dgl.nn.pytorch.glob import SumPooling


logg = get_logger()

MODEL_CACHE_DIR = Path("/afs/csail.mit.edu/u/s/samsl/Work/Adapting_PLM_DTI/models")


# class Mol2VecFeaturizer(Featurizer):
#     def __init__(self, radius: int = 1, save_dir: Path = Path().absolute()):
#         super().__init__("Mol2Vec", 300)

#         self._radius = radius
#         self._model = word2vec.Word2Vec.load(
#             f"{MODEL_CACHE_DIR}/mol2vec_saved/model_300dim.pkl"
#         )

#     def _transform(self, smile: str) -> torch.Tensor:

#         molecule = Chem.MolFromSmiles(smile)
#         try:
#             sentence = MolSentence(mol2alt_sentence(molecule, self._radius))
#             wide_vector = sentences2vec(sentence, self._model, unseen="UNK")
#             feats = wide_vector.mean(axis=0)
#         except Exception:
#             feats = np.zeros(self.shape)

#         feats = torch.from_numpy(feats).squeeze().float()
#         return feats


# class MorganFeaturizer(Featurizer):
#     def __init__(
#         self,
#         shape: int = 2048,
#         radius: int = 2,
#         save_dir: Path = Path().absolute(),
#     ):
#         super().__init__("Morgan", shape, save_dir)

#         self._radius = radius

#     def smiles_to_morgan(self, smile: str):
#         """
#         Convert smiles into Morgan Fingerprint.
#         :param smile: SMILES string
#         :type smile: str
#         :return: Morgan fingerprint
#         :rtype: np.ndarray
#         """
#         try:
#             smile = canonicalize(smile)
#             mol = Chem.MolFromSmiles(smile)
#             features_vec = AllChem.GetMorganFingerprintAsBitVect(
#                 mol, self._radius, nBits=self.shape
#             )
#             features = np.zeros((1,))
#             DataStructs.ConvertToNumpyArray(features_vec, features)
#         except Exception as e:
#             logg.error(
#                 f"rdkit not found this smiles for morgan: {smile} convert to all 0 features"
#             )
#             logg.error(e)
#             features = np.zeros((self.shape,))
#         return features

#     def _transform(self, smile: str) -> torch.Tensor:
#         # feats = torch.from_numpy(self._featurizer(smile)).squeeze().float()
#         feats = torch.from_numpy(self.smiles_to_morgan(smile)).squeeze().float()
#         if feats.shape[0] != self.shape:
#             logg.warning("Failed to featurize: appending zero vector")
#             feats = torch.zeros(self.shape)
#         return feats

class UniMolFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute(), use_3d: bool = True):
        """
        Uni-Mol分子特征提取器
        :param use_3d: 是否使用3D结构信息（需要提供或生成3D坐标）
        """
        super().__init__("Uni-Mol", 512, save_dir)  # 假设输出维度为512
        self.use_3d = use_3d
        
    
        from unimol import UniMolModel
        self.model = UniMolModel.from_pretrained("dptech-corp/unimol-base")
        self.model.eval()

        self._register_cuda("model", self.model)

        self.confgen_params = AllChem.ETKDGv3()
        self.confgen_params.randomSeed = 42

    def _smiles_to_3d(self, smiles: str) -> T.Tuple[T.List, T.List]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
            
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, self.confgen_params)
        AllChem.MMFFOptimizeMolecule(mol)
        
        coords = mol.GetConformer().GetPositions()
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        return atoms, coords.tolist()

    def _transform(self, 
                 smiles: str, 
                 coordinates: T.Optional[T.List] = None) -> torch.Tensor:
        """
        核心特征转换方法
        :param smiles: 分子SMILES字符串
        :param coordinates: 可选预计算坐标（形状为[N,3]的列表）
        """
        # 输入预处理
        if self.use_3d:
            if coordinates is None:
                _, coordinates = self._smiles_to_3d(smiles)
            else:
                if len(coordinates) < 1:
                    raise ValueError("Invalid coordinates provided")
                    
        # 将数据移动到对应设备
        device = next(self.model.parameters()).device
        
        # 模型前向传播
        with torch.no_grad():
            output = self.model(
                smiles=[smiles], 
                coordinates=[coordinates] if self.use_3d else None
            )
            
        # 特征聚合（取原子表征的平均）
        atom_features = output["atom_feat"]  
        return torch.mean(atom_features, dim=0)

    @classmethod
    def from_pretrained(cls, 
                       model_path: T.Union[str, Path],
                       **kwargs) -> "UniMolFeaturizer":
        return cls(**kwargs)


class DistilledUniMolFeaturizer(Featurizer):
    def __init__(self, 
                save_dir: Path = Path().absolute(), 
                use_3d: bool = True,
                student_model_path: str = "../../best_models/BindingDB_best_model.pt"):
        """
        蒸馏版UniMol特征提取器
        :param student_model_path: 蒸馏后学生模型的保存路径
        :param use_3d: 是否继承教师模型的3D结构感知能力
        """
        super().__init__("DistilledUni-Mol", 256, save_dir)  # 假设压缩后维度为256
        self.use_3d = use_3d
        
        # 加载蒸馏后的学生模型
        from unimol import UniMolModel
        self.student_model = UniMolModel(
            num_encoder_layers=6,  # 减少层数(原版12层)
            embedding_dim=256,     # 压缩隐藏维度
            ffn_embedding_dim=512  # 保持前馈层维度不变
        )
        self.student_model.load_state_dict(torch.load(student_model_path))
        self.student_model.eval()

        self._register_cuda("student_model", self.student_model)

        # 共享教师模型的构象生成参数
        self.confgen_params = AllChem.ETKDGv3()
        self.confgen_params.randomSeed = 42

    def _smiles_to_3d(self, smiles: str) -> T.Tuple[T.List, T.List]:
        """继承教师模型的3D坐标生成方法"""
        # ...（与原始UniMolFeaturizer相同代码）...
        
    def _transform(self, 
                 smiles: str, 
                 coordinates: T.Optional[T.List] = None) -> torch.Tensor:
        """
        蒸馏特征生成流程：
        1. 复用教师模型的3D坐标生成策略
        2. 使用轻量级学生模型进行特征提取
        3. 采用注意力池化代替平均池化[7](@ref)
        """
        # 坐标生成逻辑与教师模型保持一致
        if self.use_3d:
            if coordinates is None:
                _, coordinates = self._smiles_to_3d(smiles)
            else:
                if len(coordinates) < 1:
                    raise ValueError("Invalid coordinates provided")
        
        # 学生模型推理
        device = next(self.student_model.parameters()).device
        with torch.no_grad():
            output = self.student_model(
                smiles=[smiles],
                coordinates=[coordinates] if self.use_3d else None
            )
        
        # 注意力池化机制（参考MolKD的蒸馏策略[7](@ref)）
        atom_features = output["atom_feat"]
        attention_weights = torch.softmax(
            self.student_model.attention(atom_features), dim=0
        )
        return torch.sum(attention_weights * atom_features, dim=0)

# class GNN(torch.nn.Module):
#     def __init__(self, gnn, n_layer, feature_len, dim):
#         super(GNN, self).__init__()
#         self.gnn = gnn
#         self.n_layer = n_layer
#         self.feature_len = feature_len
#         self.dim = dim
#         self.gnn_layers = ModuleList([])
#         if gnn in ["gcn", "gat", "sage", "tag"]:
#             for i in range(n_layer):
#                 if gnn == "gcn":
#                     self.gnn_layers.append(
#                         GraphConv(
#                             in_feats=feature_len if i == 0 else dim,
#                             out_feats=dim,
#                             activation=None
#                             if i == n_layer - 1
#                             else torch.relu,
#                         )
#                     )
#                 elif gnn == "gat":
#                     num_heads = (
#                         16  # make sure that dim is dividable by num_heads
#                     )
#                     self.gnn_layers.append(
#                         GATConv(
#                             in_feats=feature_len if i == 0 else dim,
#                             out_feats=dim // num_heads,
#                             activation=None
#                             if i == n_layer - 1
#                             else torch.relu,
#                             num_heads=num_heads,
#                         )
#                     )
#                 elif gnn == "sage":
#                     agg = "pool"
#                     self.gnn_layers.append(
#                         SAGEConv(
#                             in_feats=feature_len if i == 0 else dim,
#                             out_feats=dim,
#                             activation=None
#                             if i == n_layer - 1
#                             else torch.relu,
#                             aggregator_type=agg,
#                         )
#                     )
#                 elif gnn == "tag":
#                     hops = 2
#                     self.gnn_layers.append(
#                         TAGConv(
#                             in_feats=feature_len if i == 0 else dim,
#                             out_feats=dim,
#                             activation=None
#                             if i == n_layer - 1
#                             else torch.relu,
#                             k=hops,
#                         )
#                     )
#         elif gnn == "sgc":
#             self.gnn_layers.append(
#                 SGConv(in_feats=feature_len, out_feats=dim, k=n_layer)
#             )
#         else:
#             raise ValueError("unknown GNN model")
#         self.pooling_layer = SumPooling()
#         self.factor = None

#     def forward(self, graph):
#         feature = graph.ndata["feature"]
#         h = one_hot(feature, num_classes=self.feature_len)
#         h = torch.sum(h, dim=1, dtype=torch.float)
#         for layer in self.gnn_layers:
#             h = layer(graph, h)
#             if self.gnn == "gat":
#                 h = torch.reshape(h, [h.size()[0], -1])
#         if self.factor is None:
#             self.factor = math.sqrt(self.dim) / float(
#                 torch.mean(torch.linalg.norm(h, dim=1))
#             )
#         h *= self.factor
#         graph_embedding = self.pooling_layer(graph, h)
#         return graph_embedding


# class GraphDataset(dgl.data.DGLDataset):
#     def __init__(self, path_to_model, smiles_list, gpu):
#         self.path = path_to_model
#         self.smiles_list = smiles_list
#         self.gpu = gpu
#         self.parsed = []
#         self.graphs = []
#         super().__init__(name="graph_dataset")

#     def process(self):
#         with open(self.path + "/feature_enc.pkl", "rb") as f:
#             feature_encoder = pickle.load(f)
#         for i, smiles in enumerate(self.smiles_list):
#             try:
#                 raw_graph = pysmiles.read_smiles(
#                     smiles, zero_order_bonds=False
#                 )
#                 dgl_graph = networkx_to_dgl(raw_graph, feature_encoder)
#                 self.graphs.append(dgl_graph)
#                 self.parsed.append(i)
#             except Exception as e:
#                 logg.error(e)
#                 # print('ERROR: No. %d smiles is not parsed successfully' % i)
#         # print('the number of smiles successfully parsed: %d' % len(self.parsed))
#         # print('the number of smiles failed to be parsed: %d' % (len(self.smiles_list) - len(self.parsed)))
#         if torch.cuda.is_available() and self.gpu is not None:
#             self.graphs = [
#                 graph.to("cuda:" + str(self.gpu)) for graph in self.graphs
#             ]

#     def __getitem__(self, i):
#         return self.graphs[i]

#     def __len__(self):
#         return len(self.graphs)


# def networkx_to_dgl(raw_graph, feature_encoder):
#     attribute_names = ["element", "charge", "aromatic", "hcount"]
#     # add edges
#     src = [s for (s, _) in raw_graph.edges]
#     dst = [t for (_, t) in raw_graph.edges]
#     graph = dgl.graph((src, dst), num_nodes=len(raw_graph.nodes))
#     # add node features
#     node_features = []
#     for i in range(len(raw_graph.nodes)):
#         raw_feature = raw_graph.nodes[i]
#         numerical_feature = []
#         for j in attribute_names:
#             if raw_feature[j] in feature_encoder[j]:
#                 numerical_feature.append(feature_encoder[j][raw_feature[j]])
#             else:
#                 numerical_feature.append(feature_encoder[j]["unknown"])
#         node_features.append(numerical_feature)
#     node_features = torch.tensor(node_features)
#     graph.ndata["feature"] = node_features
#     # transform to bi-directed graph with self-loops
#     graph = dgl.to_bidirected(graph, copy_ndata=True)
#     graph = dgl.add_self_loop(graph)
#     return graph


# class MolEFeaturizer(object):
#     def __init__(self, path_to_model, gpu=0):
#         self.path_to_model = path_to_model
#         self.gpu = gpu
#         with open(path_to_model + "/hparams.pkl", "rb") as f:
#             hparams = pickle.load(f)
#         self.mole = GNN(
#             hparams["gnn"],
#             hparams["layer"],
#             hparams["feature_len"],
#             hparams["dim"],
#         )
#         self.dim = hparams["dim"]
#         if torch.cuda.is_available() and gpu is not None:
#             self.mole.load_state_dict(torch.load(path_to_model + "/model.pt"))
#             self.mole = self.mole.cuda(gpu)
#         else:
#             self.mole.load_state_dict(
#                 torch.load(
#                     path_to_model + "/model.pt",
#                     map_location=torch.device("cpu"),
#                 )
#             )

#     def transform(self, smiles_list, batch_size=None, data=None):
#         if data is None:
#             data = GraphDataset(self.path_to_model, smiles_list, self.gpu)
#         dataloader = GraphDataLoader(
#             data,
#             batch_size=batch_size
#             if batch_size is not None
#             else len(smiles_list),
#         )
#         all_embeddings = np.zeros((len(smiles_list), self.dim), dtype=float)
#         flags = np.zeros(len(smiles_list), dtype=bool)
#         res = []
#         with torch.no_grad():
#             self.mole.eval()
#             for graphs in dataloader:
#                 graph_embeddings = self.mole(graphs)
#                 res.append(graph_embeddings)
#             res = torch.cat(res, dim=0).cpu().numpy()
#         all_embeddings[data.parsed, :] = res
#         flags[data.parsed] = True
#         return all_embeddings, flags


# class MolRFeaturizer(Featurizer):
#     def __init__(
#         self,
#         shape: int = 1024,
#         save_dir: Path = Path().absolute(),
#     ):
#         super().__init__("MolR", shape, save_dir)

#         self.path_to_model = f"{MODEL_CACHE_DIR}/molr_saved/gcn_1024"
#         self._molE_featurizer = MolEFeaturizer(
#             path_to_model=self.path_to_model
#         )

#     def _transform(self, smile: str) -> torch.Tensor:
#         smile = canonicalize(smile)
#         try:
#             embeddings, _ = self._molE_featurizer.transform([smile])
#         except NotImplementedError:
#             embeddings = np.zeros(self.shape)
#         tens = torch.from_numpy(embeddings).squeeze().float()
#         return tens
