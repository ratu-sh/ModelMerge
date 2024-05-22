import os
from rich.console import Console
from rich.markdown import Markdown
import json
import requests

def query_ollama(prompt, model):
    # 设置请求的URL和数据
    url = 'http://localhost:11434/api/generate'
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    response = requests.Session().post(
        url,
        json=data,
        stream=True,
    )
    full_response: str = ""
    for line in response.iter_lines():
        if not line or line.decode("utf-8")[:6] == "event:" or line.decode("utf-8") == "data: {}":
            continue
        line = line.decode("utf-8")
        # print(line)
        resp: dict = json.loads(line)
        content = resp.get("response")
        if not content:
            continue
        full_response += content
        yield content

if __name__ == "__main__":
    console = Console()
    # model = 'llama2'
    # model = 'mistral'
    # model = 'llama3:8b'
    model = 'phi3:medium'
    # model = 'qwen:14b'
    # model = 'wizardlm2:7b'
    # model = 'codeqwen:7b-chat'
    # model = 'phi'

    # 查询答案
    prompt = r'''

class MotionAGFormer(nn.Module):
    """
    MotionAGFormer, the main class of our model.
    """

    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False, graph_only=False,
                 neighbour_num=4, n_frames=243, dir_num=4, dir_act='tanh', edge_num=2, edge_act='tanh', rel0=True, h36=True):
        """
        :param n_layers: Number of layers.
        :param dim_in: Input dimension.
        :param dim_feat: Feature dimension.
        :param dim_rep: Motion representation dimension
        :param dim_out: output dimension. For 3D pose lifting it is set to 3
        :param mlp_ratio: MLP ratio.
        :param act_layer: Activation layer.
        :param drop: Dropout rate.
        :param drop_path: Stochastic drop probability.
        :param use_layer_scale: Whether to use layer scaling or not.
        :param layer_scale_init_value: Layer scale init value in case of using layer scaling.
        :param use_adaptive_fusion: Whether to use adaptive fusion or not.
        :param num_heads: Number of attention heads in attention branch
        :param qkv_bias: Whether to include bias in the linear layers that create query, key, and value or not.
        :param qkv_scale: scale factor to multiply after outer product of query and key. If None, it's set to
                          1 / sqrt(dim_feature // num_heads)
        :param hierarchical: Whether to use hierarchical structure or not.
        :param num_joints: Number of joints.
        :param use_temporal_similarity: If true, for temporal GCN uses top-k similarity between nodes
        :param temporal_connection_len: Connects joint to itself within next `temporal_connection_len` frames
        :param use_tcn: If true, uses MS-TCN for temporal part of the graph branch.
        :param graph_only: Uses GCN instead of GraphFormer in the graph branch.
        :param neighbour_num: Number of neighbors for temporal GCN similarity.
        :param n_frames: Number of frames. Default is 243
        """
        super().__init__()
        self.h36 = h36
        if h36:
            self.root_id = 0
            self.limbs_id = [[0, 1], [1, 2], [2, 3],
                    [0, 4], [4, 5], [5, 6],
                    [0, 7], [7, 8], [8, 9], [9, 10],
                    [8, 11], [11, 12], [12, 13],
                    [8, 14], [14, 15], [15, 16]
                    ]
        else:
            self.root_id = 14
            self.limbs_id = [[14, 8], [8, 9], [9, 10],
                             [14, 11], [11, 12], [12, 13],
                             [14, 15], [15, 1], [1, 16], [16, 0],
                             [1, 5], [5, 6], [6, 7],
                             [1, 2], [2, 3], [3, 4]]

        self.joints_embed = nn.Linear(dim_in, dim_feat)
        # self.inp_embed = nn.Linear(dim_feat * 2, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        # self.pos_embed2 = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.norm = nn.LayerNorm(dim_feat)

        self.layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,
                                    num_heads=num_heads,
                                    use_layer_scale=use_layer_scale,
                                    qkv_bias=qkv_bias,
                                    qkv_scale=qkv_scale,
                                    layer_scale_init_value=layer_scale_init_value,
                                    use_adaptive_fusion=use_adaptive_fusion,
                                    hierarchical=hierarchical,
                                    use_temporal_similarity=use_temporal_similarity,
                                    temporal_connection_len=temporal_connection_len,
                                    use_tcn=use_tcn,
                                    graph_only=graph_only,
                                    neighbour_num=neighbour_num,
                                    n_frames=n_frames)
        # self.layers_edge = create_layers(dim=dim_feat,
        #                             n_layers=n_layers//2,
        #                             mlp_ratio=mlp_ratio,
        #                             act_layer=act_layer,
        #                             attn_drop=attn_drop,
        #                             drop_rate=drop,
        #                             drop_path_rate=drop_path,
        #                             num_heads=num_heads,
        #                             use_layer_scale=use_layer_scale,
        #                             qkv_bias=qkv_bias,
        #                             qkv_scale=qkv_scale,
        #                             layer_scale_init_value=layer_scale_init_value,
        #                             use_adaptive_fusion=use_adaptive_fusion,
        #                             hierarchical=hierarchical,
        #                             use_temporal_similarity=use_temporal_similarity,
        #                             temporal_connection_len=temporal_connection_len,
        #                             use_tcn=use_tcn,
        #                             graph_only=graph_only,
        #                             neighbour_num=neighbour_num,
        #                             n_frames=n_frames)

        # self.rep_logit_dir = nn.Sequential(OrderedDict([
        #     ('fc', nn.Linear(dim_feat, dim_rep)),
        #     ('act', nn.Tanh())
        # ]))
        # self.rep_logit_edge = nn.Sequential(OrderedDict([
        #     ('fc', nn.Linear(dim_feat, dim_rep)),
        #     ('act', nn.Tanh())
        # ]))

        # self.head = nn.Linear(dim_rep, dim_out)
        self.rel0 = rel0
        self.dir_act = self.get_act(dir_act)
        self.edge_act = self.get_act(edge_act)
        if self.rel0:
            self.dir_head = MLP_zsr(dim_feat, dim_rep, 3, dir_num, drop_out=0.0, act=self.dir_act)
            self.edge_head = MLP_zsr(dim_feat, dim_rep, 1, edge_num, drop_out=0.0, act=self.edge_act)
        else:
            self.pose_head = PredHead(dim_feat, dim_rep, 3, 0, dir_num, 0, act=self.dir_act)
            self.dir_head = PredHead(dim_feat, dim_rep, 3, 0, dir_num, 0, act=self.dir_act)
            self.edge_head = PredHead(dim_feat, dim_rep, 3, 0, edge_num, 0, act=self.edge_act)

    def get_act(self, act):
        if act == 'tanh':
            return nn.Tanh()
        elif act == 'relu':
            return nn.ReLU()

    def get_limbs(self, x):
        B, T, J, C = x.shape
        limbs = x[:, :, self.limbs_id, :]
        limbs = limbs[:, :, :, 1, :] - limbs[:, :, :, 0, :]  # (B, T, J-1, C)
        res = torch.zeros((B, T, J, C), device=x.device)
        res[:,:,1:,:] = limbs[:,:,:,:]
        return res
    def get_dir(self, x):
        limbs = x[:, :, self.limbs_id, :]
        limbs = limbs[:, :, :, 1, :] - limbs[:, :, :, 0, :]     # (b, f, 16, 3)
        limbs = F.normalize(limbs, dim=-1)
        return limbs

    def get_edge(self, x):
        limbs = x[:, :, self.limbs_id, :]
        limbs = limbs[:, :, :, 1, :] - limbs[:, :, :, 0, :]     # (b, f, 16, 3)
        limbs_len = torch.norm(limbs, dim=-1, keepdim=True)
        return limbs_len

    def dir_forward(self, x):
        x = self.dir_head(x)        # (bs, f, 17, 3)
        x = self.get_dir(x)         # (bs, f, 16, 3)
        return x

    def edge_forward(self, x):
        x = self.edge_head(x)       # (b, f, 17, 3)
        x = self.get_edge(x)        # (b, f, 16, 1)
        return x

    def combine_dir_edge(self, x_dir, x_edge):
        shape = list(x_dir.shape)
        shape[-2] = 17
        pose = torch.zeros(shape, device=x_dir.device)
        for i in range(16):
            k0, k1 = self.limbs_id[i]
            pose[:,:,k1] = pose[:,:,k0] + x_dir[:,:, i] * (x_edge[:,:, i].clamp(min=1e-3))
        return pose
    # def combine_dir_edge_detach(self, x_dir, x_edge):
    #     assert x_dir.shape[-1] == 3 and x_edge.shape[-1] == 1
    #     # detach
    #     dir_length = torch.norm(x_dir, dim=-1, keepdim=True).clamp(min=1e-9).detach()
    #     x_dir = x_dir / dir_length
    #     pose = x_dir * x_edge
    #     pose[:,:,0] = 0
    #     return pose, x_dir, x_edge

    def forward(self, x, return_rep=False):
        """
        :param x: tensor with shape [B, T, J, C] (T=243, J=17, C=3)
        :param return_rep: Returns motion representation feature volume (In case of using this as backbone)
        """
        eps = 1e-7
        x = self.joints_embed(x)
        x = x + self.pos_embed
        # x_l = self.get_limbs(x)
        # x = self.inp_embed(torch.cat([x, x_l], dim=-1)) + self.pos_embed2     # 再添加一个pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if self.rel0:
            x_dir = self.dir_head(x)
            x_edge = self.edge_head(x)
            x_dir = F.normalize(x_dir, dim=-1)
            x_dir[...,self.root_id,:] = 0
            pose = x_dir * (x_edge.clamp(min=1e-3))
            return pose, x_dir, x_edge
        else:
            # pose_fusion = self.pose_fusion(x).softmax(dim=-1)   # (B, T, J, 2)
            pose_0 = self.pose_head(x)
            x_dir_0 = self.get_dir(pose_0)
            x_edge_0 = self.get_edge(pose_0)
            # pose = pose_0
            # x_dir = x_dir_0
            # x_edge = x_edge_0
            x_dir_1 = self.dir_forward(x)     # (bs, f, 16, 3)
            x_edge_1 = self.edge_forward(x)   # (bs, f, 16, 1)
            pose_1 = self.combine_dir_edge(x_dir_1, x_edge_1)
            # pose = pose_1
            # x_dir = x_dir_1
            # x_edge = x_edge_1
            x_dir = (x_dir_0 + x_dir_1) / 2
            x_edge = (x_edge_0 + x_edge_1) / 2
            pose = (pose_0 + pose_1)/2
            return pose, x_dir, x_edge

        # for layer in self.layers:
        #     x = layer(x)

        # x = self.norm(x)
        # x_dir = self.rep_logit_dir(x)   # (bs, f, 17, 512)
        # if return_rep:
        #     return x

        # x = self.head(x)
        # x_dir = self.dir_proj(x.transpose(-1, -2)).transpose(-1,-2)
        # x_dir = self.dir_head(x_dir)    # (bs, f, 16, 3)
        # x_scale = self.scale_proj(x.transpose(-1, -2)).transpose(-1, -2)
        # x_scale = self.scale_head(x_scale)  # (bs, f, 16, 1)
        # pose, uni_dir = combine(x_dir, x_scale)
        # return x
        # return uni_dir, x_scale, pose
分析一下这个代码

'''
    answer = ""
    for result in query_ollama(prompt, model):
        os.system("clear")
        answer += result
        md = Markdown(answer)
        console.print(md, no_wrap=False)
