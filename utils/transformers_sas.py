import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F
from .stochastic_depth import DropPath
import torch.nn as nn

class Quad2(nn.Module): # quad2 w/o mask
    def __init__(self, c=5, dim=-1):
        super(Quad2, self).__init__()
        self.dim=dim
        self.c=c



    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')
        x=x+self.c
        x=torch.square(x)

        #x=x*attention_mask_quad

        s=torch.sum(x, dim=self.dim, keepdim=True)

        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #s = s.expand(-1, -1, -1, x.shape[-1])
        #print (f'attention score sum: {s.shape} {s[0]}')

        output=x/ s

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))

        return output

class trainableQuad2(nn.Module): # trainableQuad2 w/o mask
    def __init__(self, attn, input_N_dim=None, S=None, NS=None, num_heads=None, dim=-1, use_a=False, use_d=False, use_e=False, init=False):
        super(trainableQuad2, self).__init__()
        self.dim=dim
        #self.c=c

        self.use_a=use_a
        self.use_d = use_d
        self.usc_e= use_e
        if self.use_a:
            if attn=='self':
                self.a = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.a = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn=='mpcext':
                self.a = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if self.use_d:
            if attn=='self':
                self.d = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.d = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn=='mpcext':
                self.d = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if self.usc_e:
            if attn == 'self':

                self.e = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn == 'ext':
                self.e = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn == 'mpcext':
                self.e = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)


        if attn=='self':
            self.c = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
        elif attn=='ext':
            self.c = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
        elif attn == 'mpcext':
            self.c = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if init:
            with torch.no_grad():
                self.c.fill_(2)
                self.d.fill_(2)
                print ("init c and d")
                #print (self.d)
            self.c.requires_grad = True
            self.d.requires_grad = True




    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')
        # torch.save(x, "inputs_of_softmax")
        # breakpoint()
        if self.use_d:
            x = x * self.d

        x=x+self.c
        x=torch.square(x)
        # x=torch.pow(x,2)

        #x=x*attention_mask_quad
        if self.use_a:
            x = torch.square(self.a) * x

        if self.usc_e:
            x= x+self.e

        s=torch.sum(x, dim=self.dim, keepdim=True)

        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #s = s.expand(-1, -1, -1, x.shape[-1])
        #print (f'attention score sum: {s.shape} {s[0]}')

        output=x/ s

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))

        return output




class ReluSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(ReluSoftmax, self).__init__()
        self.dim=dim


    def forward(self, x):
        #print (f'softmax input shape:{x.shape}')
        #s = torch.sum(torch.relu(x), dim=-1) + 1e-10

        #print (f'sum shape:{s.shape}')
        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #print (f'sum reshape:{s.shape}')
        #print (s)

        #s = s.expand(-1,-1,-1, x.shape[-1])
        #print (s.shape)


        #output = torch.relu(x) / s
        #print (torch.sum(torch.relu(output), dim=-1))
        output = torch.relu(x)/(torch.sum(torch.relu(x), dim=self.dim,  keepdim=True) + 1e-10)

        return output


class ScaleAttn(nn.Module):
    def __init__(self, scale, dim=-1):
        super(ScaleAttn, self).__init__()
        self.scale=scale
        self.dim=dim


    def forward(self, x):
        # print (f'softmax input shape:{x.shape}')
        # assert x.shape[-1]==x.shape[-2]

        n=x.shape[-1] # n is sequence length

        x = x / self.scale # recover q * k^T
        output= x / n
        return output


class ExternalAttention(Module):
    """
    Obtained from https://github.com/MenghaoGuo/EANet/blob/main/EAMLP/models/token_performer.py
    https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/ExternalAttention.py
    """

    def __init__(self, dim, num_heads=8, externalattention_dim=64, attention_dropout=0.1, projection_dropout=0.1, softmax_approx='original', externalattention_divhead=False, allargs=None):
        super().__init__()
        print (f"External Attention, div head: {externalattention_divhead}")
        print (f"attention softmax:{softmax_approx}")
        self.externalattention_divhead=externalattention_divhead

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.S=externalattention_dim

        #self.qkv = Linear(dim, dim * 3, bias=False)
        self.q = Linear(dim, dim, bias=False)

        if not self.externalattention_divhead:
            self.Mk = Linear(dim, self.S, bias=False)
        else:
            self.Mk = Linear(head_dim, self.S, bias=False)

        if not self.externalattention_divhead:
            self.Mv = Linear(self.S, dim,  bias=False)
        else:
            self.Mv = Linear(self.S, head_dim, bias=False)

        self.attn_drop = Dropout(attention_dropout)
        if not self.externalattention_divhead:
            self.proj = Linear(dim, dim)
        else:
            self.proj = Linear(head_dim, head_dim)
        self.proj_drop = Dropout(projection_dropout)

        if softmax_approx=='original':
            self.softmaxlayer=nn.Softmax(dim=-2)

        elif softmax_approx=='relusoftmax':
            self.softmaxlayer=ReluSoftmax(dim=-2)

        elif softmax_approx=='quad2':
            self.softmaxlayer=Quad2(c=5, dim=-2)

        elif softmax_approx=='scaleattn':
            self.softmaxlayer=ScaleAttn(self.scale, dim=-1)


        elif softmax_approx=='biquad4':
            self.softmaxlayer=biquad4(c=5, dim=-2)

        elif softmax_approx=='trainablequad2cd':
             self.softmaxlayer=trainableQuad2(attn='ext',
                                                 input_N_dim=allargs.input_N_dim,
                                                 S=self.S,
                                                 dim=-2,
                                                 use_d=True,
                                                 )


    def forward(self, x):
        B, N, C = x.shape # N is # of patches

        if not self.externalattention_divhead:
            x=self.q(x) # x: B, N, C
        else:
            x = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # x: B, Head, N, C//Head

        idn = x[:]
        #x = x.view(B, N, -1)
        #print (f"after view: {x.shape}")

        attn=self.Mk(x) # attn: B, N, S | divhead: B, H, N, S


        #print (f"after Mk: {x.shape}")
        attn = self.softmaxlayer(attn) # B, N, S | divhead: B, H, N, S
        #print (f"sum dim: {attn.sum(dim=-1, keepdim=True).shape}") # B, N, 1
        #print (attn.sum(dim=-1, keepdim=True)[0][0])

        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        #print (f"after softmax: {attn.shape}")
        #print (f"check after softmax: {attn.sum(dim=-1)}")


        attn = self.attn_drop(attn)

        x = self.Mv(attn) # B, N, C | divhead: B, H, N, C//H



        x = self.proj(x)
        x = self.proj_drop(x)

        # residual connection
        x=idn+x

        if self.externalattention_divhead:
            x=x.permute(0, 2, 1, 3).view(B, N, -1)

        return x

class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1, softmax_approx='original', allargs=None):
        super().__init__()
        print (f"attention softmax:{softmax_approx}")

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

        if softmax_approx=='original':
            self.softmaxlayer=nn.Softmax(dim=-1)

        elif softmax_approx=='relusoftmax':
            self.softmaxlayer=ReluSoftmax()

        elif softmax_approx=='quad2':
            self.softmaxlayer=Quad2(c=5)

        elif softmax_approx=='scaleattn':
            self.softmaxlayer=ScaleAttn(self.scale)

        elif softmax_approx=='linearsoftmax':
            self.softmaxlayer=linearsoftmax()


        elif softmax_approx=='biquad4':
            self.softmaxlayer=biquad4(c=5)

        
        elif softmax_approx=='trainablequad2cd':
            self.softmaxlayer=trainableQuad2(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               init=allargs.initcd,
                                               )


    def forward(self, x):
        B, N, C = x.shape # N is # of patches
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # B, num_head, N_path, N_path
        #print (f"before softmax: {attn.shape}")
        attn=self.softmaxlayer(attn)

        #attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionHeadCore(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1, softmax_approx='original', allargs=None):
        super().__init__()
        print (f"attention softmax:{softmax_approx}")

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        # self.proj = Linear(dim, dim)
        self.proj = Linear(head_dim, head_dim)
        self.proj_drop = Dropout(projection_dropout)

        if softmax_approx=='original':
            self.softmaxlayer=nn.Softmax(dim=-1)

        elif softmax_approx=='relusoftmax':
            self.softmaxlayer=ReluSoftmax()

        elif softmax_approx=='quad2':
            self.softmaxlayer=Quad2(c=5)

        elif softmax_approx=='scaleattn':
            self.softmaxlayer=ScaleAttn(self.scale)

        elif softmax_approx=='linearsoftmax':
            self.softmaxlayer=linearsoftmax()


        elif softmax_approx=='biquad4':
            self.softmaxlayer=biquad4(c=5)

        
        elif softmax_approx=='trainablequad2cd':
            self.softmaxlayer=trainableQuad2(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               init=allargs.initcd,
                                               )


    def forward(self, x):
        B, N, C = x.shape # N is # of patches
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # B, num_head, N_path, N_path
        #print (f"before softmax: {attn.shape}")
        attn=self.softmaxlayer(attn)

        #attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ExternalAttentionHeadCore(Module):
    """
    Obtained from https://github.com/MenghaoGuo/EANet/blob/main/EAMLP/models/token_performer.py
    https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/ExternalAttention.py
    """

    def __init__(self, dim, num_heads=8, externalattention_dim=64, attention_dropout=0.1, projection_dropout=0.1, softmax_approx='original', externalattention_divhead=False, allargs=None):
        super().__init__()
        print (f"External Attention, div head: {externalattention_divhead}")
        print (f"attention softmax:{softmax_approx}")
        self.externalattention_divhead=externalattention_divhead

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.S=externalattention_dim

        #self.qkv = Linear(dim, dim * 3, bias=False)
        self.q = Linear(dim, dim, bias=False)

        if not self.externalattention_divhead:
            self.Mk = Linear(dim, self.S, bias=False)
        else:
            self.Mk = Linear(head_dim, self.S, bias=False)

        if not self.externalattention_divhead:
            self.Mv = Linear(self.S, dim,  bias=False)
        else:
            self.Mv = Linear(self.S, head_dim, bias=False)

        self.attn_drop = Dropout(attention_dropout)
        if not self.externalattention_divhead:
            self.proj = Linear(dim, dim)
        else:
            self.proj = Linear(head_dim, head_dim)
        self.proj_drop = Dropout(projection_dropout)

        if softmax_approx=='original':
            self.softmaxlayer=nn.Softmax(dim=-2)

        elif softmax_approx=='relusoftmax':
            self.softmaxlayer=ReluSoftmax(dim=-2)

        elif softmax_approx=='quad2':
            self.softmaxlayer=Quad2(c=5, dim=-2)

        elif softmax_approx=='scaleattn':
            self.softmaxlayer=ScaleAttn(self.scale, dim=-2)


        elif softmax_approx=='biquad4':
            self.softmaxlayer=biquad4(c=5, dim=-2)

        elif softmax_approx=='trainablequad2cd':
            self.softmaxlayer=trainableQuad2(attn='ext',
                                                input_N_dim=allargs.input_N_dim,
                                                S=self.S,
                                                dim=-2,
                                                use_d=True,
                                                )
            # self.softmaxlayer=biquad4(c=5, dim=-2)

    def forward(self, x):
        B, N, C = x.shape # N is # of patches

        if not self.externalattention_divhead:
            x=self.q(x) # x: B, N, C
        else:
            x = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # x: B, Head, N, C//Head

        idn = x[:]
        attn=self.Mk(x) # attn: B, N, S | divhead: B, H, N, S


        #print (f"after Mk: {x.shape}")
        attn = self.softmaxlayer(attn) # B, N, S | divhead: B, H, N, S

        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))

        attn = self.attn_drop(attn)

        x = self.Mv(attn) # B, N, C | divhead: B, H, N, C//H



        x = self.proj(x)
        x = self.proj_drop(x)

        # residual connection
        x=idn+x

        x=x.permute(0, 2, 1, 3)

        return x    

class HybridAttentionHeadWise(Module):
    def __init__(self, dim, num_heads=8, externalattention_dim=64, attention_dropout=0.1, projection_dropout=0.1, softmax_approx='original', externalattention_divhead=False, allargs=None):
        super(HybridAttentionHeadWise, self).__init__()
        self.alphas = Parameter(0.1*torch.ones(1, 1, num_heads, 1), requires_grad=True)
        self.self_attn = AttentionHeadCore(dim=dim, num_heads=num_heads,
                                    attention_dropout=attention_dropout, projection_dropout= projection_dropout, softmax_approx=softmax_approx, allargs=allargs)
        self.external_attn = ExternalAttentionHeadCore(dim=dim,  num_heads=num_heads, externalattention_dim=externalattention_dim, externalattention_divhead=externalattention_divhead,
                                       attention_dropout=attention_dropout, projection_dropout=projection_dropout, softmax_approx=softmax_approx, allargs=allargs)

        self.num_heads = num_heads

    def forward(self, x):
        B, N, C = x.shape
        x1 = self.self_attn(x)
        x2 = self.external_attn(x)

        a,b,c,d=x1.shape
        # print('HybridAttn Check!!!')
        # print('Heads: ', self.num_heads)

        # print('After attn: ', x1.shape, x2.shape)


        alpha = self.alphas
        alpha = alpha.expand(a, b, c, d)
        # print('alpha:', alpha.shape)
        out = alpha*x1 + (1-alpha)*x2

        out = out.reshape(B,N,C)

        return out


class HybridAttention(Module):
    def __init__(self, dim, num_heads=8, externalattention_dim=64, attention_dropout=0.1, projection_dropout=0.1, softmax_approx='original', externalattention_divhead=False, allargs=None):
        super(HybridAttention, self).__init__()
        self.alphas = Parameter(0.1*torch.ones(1, 1, 1), requires_grad=True)
        self.self_attn = Attention(dim=dim, num_heads=num_heads,
                                    attention_dropout=attention_dropout, projection_dropout= projection_dropout, softmax_approx=softmax_approx, allargs=allargs)
        self.external_attn = ExternalAttention(dim=dim,  num_heads=num_heads, externalattention_dim=externalattention_dim, externalattention_divhead=externalattention_divhead,
                                       attention_dropout=attention_dropout, projection_dropout=projection_dropout, softmax_approx=softmax_approx, allargs=allargs)


    def forward(self, x):
        x1 = self.self_attn(x)
        x2 = self.external_attn(x)

        # alpha = sigmoid(alpha)
        alpha = self.alphas

        # alpha = alpha.expand_as(x1)
        # out = self.alphas*x1 + (1-self.alphas)*x2
        out = alpha*x1 + (1-alpha)*x2

        return out



class trainableBiquad4(Module):

    def __init__(self, dim):
        super(trainableBiquad4, self).__init__()
        self.a = nn.Parameter(torch.ones(1,1,1), requires_grad = True)
        self.c = nn.Parameter(torch.zeros(1,1,1), requires_grad= True)
        self.active = biquad4(self.c, dim=dim)

    def forward(self, x, attention_mask_quad=None):
        a = self.a
        c = self.c
        output = a* self.active(x) + c

        return output


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1, softmax_approx='original', attention_mechanism='original', externalattention_dim=64, externalattention_divhead=False, allargs=None):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)

        if attention_mechanism=='original':
            self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                       attention_dropout=attention_dropout, projection_dropout=dropout, softmax_approx=softmax_approx, allargs=allargs)
        elif attention_mechanism=='externalattention':
            self.self_attn = ExternalAttention(dim=d_model,  num_heads=nhead, externalattention_dim=externalattention_dim, externalattention_divhead=externalattention_divhead,
                                       attention_dropout=attention_dropout, projection_dropout=dropout, softmax_approx=softmax_approx, allargs=allargs)
        elif attention_mechanism=='hybrid':
            self.self_attn = HybridAttention(dim=d_model,  num_heads=nhead, externalattention_dim=externalattention_dim, externalattention_divhead=externalattention_divhead,
                                       attention_dropout=attention_dropout, projection_dropout=dropout, softmax_approx=softmax_approx, allargs=allargs)

        elif attention_mechanism=='hybridHeadWise':
            self.self_attn = HybridAttentionHeadWise(dim=d_model,  num_heads=nhead, externalattention_dim=externalattention_dim, externalattention_divhead=externalattention_divhead,
                                       attention_dropout=attention_dropout, projection_dropout=dropout, softmax_approx=softmax_approx, allargs=allargs)


        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src



class TransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None,
                 softmax_approx='original',
                 attention_mechanism='original',
                 externalattention_dim=64,
                 externalattention_divhead=False,
                allargs=None):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i], softmax_approx=softmax_approx,
                                    attention_mechanism=attention_mechanism, externalattention_dim=externalattention_dim, externalattention_divhead=externalattention_divhead, allargs=allargs)
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        last_vit_feature=x

        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x, last_vit_feature

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

