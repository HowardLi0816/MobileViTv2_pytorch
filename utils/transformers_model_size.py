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
    def __init__(self, attn, input_N_dim=None, S=None, NS=None, num_heads=None, dim=-1, usc_c=True, use_a=False,  use_d=False, use_e=False, init=False):
        super(trainableQuad2, self).__init__()
        self.dim=dim
        #self.c=c

        self.use_a=use_a
        self.use_d = use_d
        self.usc_e= use_e
        self.usc_c=usc_c
        if self.use_a:
            if attn=='self' or attn=='linvar':
                self.a = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.a = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn=='mpcext':
                self.a = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if self.use_d:
            if attn=='self' or attn=='linvar':
                self.d = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.d = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn=='mpcext':
                self.d = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if self.usc_e:
            if attn == 'self' or attn=='linvar':

                self.e = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn == 'ext':
                self.e = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn == 'mpcext':
                self.e = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if self.usc_c:
            if attn=='self' or attn=='linvar':
                self.c = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.c = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn == 'mpcext':
                self.c = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if init:
            with torch.no_grad():
                self.c.fill_(1)
                self.d.fill_(1)
                print ("init c and d")
                #print (self.d)
            self.c.requires_grad = True
            self.d.requires_grad = True




    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')

        if self.use_d:
            x = x * self.d

        if self.usc_c:
            x=x+self.c

        x=torch.square(x)

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

class fixQuad2(nn.Module): # trainableQuad2 w/o mask
    def __init__(self, dim=-1, c=1, d=1):
        super(fixQuad2, self).__init__()
        self.dim=dim
        self.c=c
        self.d=d


    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')

        x = x + self.c
        x = x * self.d


        x=torch.square(x)


        s=torch.sum(x, dim=self.dim, keepdim=True)

        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #s = s.expand(-1, -1, -1, x.shape[-1])
        #print (f'attention score sum: {s.shape} {s[0]}')

        output=x/ s

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))

        return output

class trainableQuad2headlevel(nn.Module): # trainableQuad2 w/o mask
    def __init__(self, attn, input_N_dim=None, S=None, NS=None, num_heads=None, dim=-1, use_a=False, use_d=False, use_e=False, init=False):
        super(trainableQuad2headlevel, self).__init__()
        self.dim=dim
        #self.c=c

        self.use_d = use_d



        if self.use_d:
            if attn=='self' or attn=='linvar':
                self.d = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)



        if attn=='self' or attn=='linvar':
            self.c = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)

        if init:
            with torch.no_grad():
                self.c.fill_(1)
                self.d.fill_(1)
                print ("init c and d")
                #print (self.d)
            self.c.requires_grad = True
            self.d.requires_grad = True




    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')

        if self.use_d:
            x = x * self.d

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

class trainableQuad2tokenlevel(nn.Module): # trainableQuad2 w/o mask
    def __init__(self, attn, input_N_dim=None, S=None, NS=None, num_heads=None, dim=-1, use_a=False, use_d=False, use_e=False, init=False):
        super(trainableQuad2tokenlevel, self).__init__()
        self.dim=dim
        #self.c=c

        self.use_d = use_d



        if self.use_d:
            if attn=='self' or attn=='linvar':
                self.d = nn.Parameter(torch.ones(1, num_heads, input_N_dim, 1), requires_grad=True)



        if attn=='self' or attn=='linvar':
            self.c = nn.Parameter(torch.ones(1, num_heads, input_N_dim, 1), requires_grad=True)

        if init:
            with torch.no_grad():
                self.c.fill_(1)
                self.d.fill_(1)
                print ("init c and d")
                #print (self.d)
            self.c.requires_grad = True
            self.d.requires_grad = True




    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')

        if self.use_d:
            x = x * self.d

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

class trainableQuad2layerlevel(nn.Module): # trainableQuad2 w/o mask
    def __init__(self, dim=-1, c=None, d=None):
        super(trainableQuad2layerlevel, self).__init__()
        self.dim=dim
        self.c=c
        self.d=d




    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')

        #if self.use_d:
        x = x * self.d

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


class softmaxreciprocalapprox(nn.Module): # quad2 w/o mask
    def __init__(self, c=5, dim=-1):
        super(softmaxreciprocalapprox, self).__init__()
        self.dim=dim
        self.c=c



    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')
        #x=x+self.c
        x=torch.exp(x)

        #x=x*attention_mask_quad

        #s=torch.sum(x, dim=self.dim, keepdim=True)
        s = torch.sum(x, dim=self.dim, keepdim=True)

        s=3*torch.exp(0.5-s)+0.003
        #s = 2.9183 * torch.exp(0.4191 - 1.4337*s) + -0.1143

        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #s = s.expand(-1, -1, -1, x.shape[-1])
        #print (f'attention score sum: {s.shape} {s[0]}')

        #output=x/ s
        output=x*s

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))

        return output

class trainablesoftmaxreciprocalapprox(nn.Module): # quad2 w/o mask
    def __init__(self, c=5, dim=-1):
        super(trainablesoftmaxreciprocalapprox, self).__init__()
        self.dim=dim
        self.c=c

        self.parameter=nn.Parameter(torch.zeros(4), requires_grad=True)

        with torch.no_grad():
            self.parameter[0].fill_(1)
            self.parameter[1].fill_(0.5)
            self.parameter[2].fill_(3)
            self.parameter[3].fill_(0.003)



    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')
        #x=x+self.c
        x=torch.exp(x)

        #x=x*attention_mask_quad

        #s=torch.sum(x, dim=self.dim, keepdim=True)
        s = torch.sum(x, dim=self.dim, keepdim=True)

        #s=3*torch.exp(0.5-s)+0.003
        s = self.parameter[2] * torch.exp(self.parameter[1] - self.parameter[0]*s) + self.parameter[3]

        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #s = s.expand(-1, -1, -1, x.shape[-1])
        #print (f'attention score sum: {s.shape} {s[0]}')

        #output=x/ s
        output=x*s

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))

        return output


class softmaxreciprocalfree1(nn.Module): # quad2 w/o mask
    def __init__(self, c=5, dim=-1):
        super(softmaxreciprocalfree1, self).__init__()
        self.dim=dim
        self.c=c



    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')
        x=x+self.c
        x=torch.square(x)

        #x=x*attention_mask_quad

        #s=torch.sum(x, dim=self.dim, keepdim=True)
        #curmax=torch.max(x)
        curmax=torch.amax(x, dim=self.dim, keepdim=True)

        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #s = s.expand(-1, -1, -1, x.shape[-1])
        #print (f'attention score sum: {s.shape} {s[0]}')

        #output=x/ s
        output=x/curmax

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))

        return output

class softmaxreciprocalfree2(nn.Module): # quad2 w/o mask
    def __init__(self, c=5, dim=-1):
        super(softmaxreciprocalfree2, self).__init__()
        self.dim=dim
        self.c=c



    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')
        # curmax=torch.max(x)
        curmax = torch.amax(x, dim=self.dim, keepdim=True)
        x=x-curmax

        #x=x+self.c
        output=torch.square(x)

        #x=x*attention_mask_quad

        #s=torch.sum(x, dim=self.dim, keepdim=True)


        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #s = s.expand(-1, -1, -1, x.shape[-1])
        #print (f'attention score sum: {s.shape} {s[0]}')

        #output=x/ s


        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))

        return output

class softmaxreciprocalfree3(nn.Module): # quad2 w/o mask
    def __init__(self, c=5, dim=-1):
        super(softmaxreciprocalfree3, self).__init__()
        self.dim=dim
        self.c=c



    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')
        x=x+self.c
        x=torch.square(x)

        #x=x*attention_mask_quad

        #s=torch.sum(x, dim=self.dim, keepdim=True)
        # curmax=torch.max(x)
        curmax=torch.amax(x, dim=self.dim, keepdim=True)
        curmin=torch.amin(x, dim=self.dim, keepdim=True)

        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #s = s.expand(-1, -1, -1, x.shape[-1])
        #print (f'attention score sum: {s.shape} {s[0]}')

        #output=x/ s
        output=(x-curmin)/(curmax-curmin)

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))

        return output

class trainablelinear(nn.Module): # trainableQuad2 w/o mask
    def __init__(self, attn, input_N_dim=None, S=None, NS=None, num_heads=None, dim=-1, use_a=True, use_b=True, init=True):
        super(trainablelinear, self).__init__()
        self.dim=dim
        #self.c=c

        self.use_a=use_a
        self.use_b = use_b


        if self.use_a:
            if attn=='self':
                self.a = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.a = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn=='mpcext':
                self.a = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if self.use_b:
            if attn=='self':
                self.b = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.b = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn == 'mpcext':
                self.b = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if init:
            with torch.no_grad():
                self.a.fill_(0.1)
                self.b.fill_(10)
                print ("init a and b")
                #print (self.d)
            self.a.requires_grad = True
            self.b.requires_grad = True




    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')

        # a* (x+b)
        if self.use_b:
            x = x + self.b

        if self.use_a:
            x= x * self.a

        s=torch.sum(x, dim=self.dim, keepdim=True)

        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #s = s.expand(-1, -1, -1, x.shape[-1])
        #print (f'attention score sum: {s.shape} {s[0]}')

        output=x/ s

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))

        return output



class Cube3(nn.Module): # cube3 w/o mask
    def __init__(self, c=5, dim=-1):
        super(Cube3, self).__init__()
        self.dim=dim
        self.c=c



    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')
        x=x+self.c
        #x=torch.square(x)
        x= torch.pow(x, 3)

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

class biquad4(nn.Module): # quad2 w/o mask
    def __init__(self, c=5, dim=-1):
        super(biquad4, self).__init__()
        self.dim=dim
        self.c=c



    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')
        x=x+self.c

        x=torch.pow(x, 4)

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

class biquad4woc(nn.Module): # quad2 w/o mask
    def __init__(self, dim=-1):
        super(biquad4woc, self).__init__()
        self.dim=dim
        #self.c=c



    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')
        #x=x+self.c

        x=torch.pow(x, 4)

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

class trainableBiquad4(nn.Module): # quad2 w/o mask
    def __init__(self, attn, input_N_dim=None, S=None, NS=None, num_heads=None, dim=-1, use_a=False, use_d=False):
        super(trainableBiquad4, self).__init__()
        self.dim=dim
        #self.c=c
        self.use_a=use_a
        self.use_d = use_d
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

        if attn=='self':
            self.c = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
        elif attn=='ext':
            self.c = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
        elif attn == 'mpcext':
            self.c = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)




    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')
        #x=x+self.c

        # quad(a)* biquad4(d * x+c)
        #print (sum(torch.flatten(x) >= 0) / len(torch.flatten(x)))
        #t=torch.flatten((x > -1) & (x < 1)).int()
        #print (f"non-negative rate:", torch.sum(t)/t.shape[0])
        if self.use_d:
            x= x*self.d


        x = x + self.c
        x = torch.pow(x, 4)

        #print (f"x.shape: {x.shape}")
        #print (f"c.shape: {self.c.shape}")
        if self.use_a:
            #self.a=torch.square(self.a) # cannot assign to a parameter
            x = torch.square(self.a)*x
            #print (f"a.shape: {self.a.shape}")


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

class mpcpolynomialsoftmax_sigmoid_1(nn.Module): # quad2 w/o mask
    def __init__(self, dim=-1):
        super(mpcpolynomialsoftmax_sigmoid_1, self).__init__()
        self.dim=dim



    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')
        #x=x+self.c

        x=-0.0003*torch.pow(x, 3)+0.1798*x+10.5

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

class linearsoftmax(nn.Module): # quad2 w/o mask
    def __init__(self, dim=-1):
        super(linearsoftmax, self).__init__()
        self.dim=dim
        #self.c=c



    def forward(self, x, attention_mask_quad=None):

        s=torch.sum(x, dim=self.dim, keepdim=True)

        output=x/ s

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))


        return output

class linearabssoftmax(nn.Module): # linearabssoftmax
    def __init__(self, dim=-1):
        super(linearabssoftmax, self).__init__()
        self.dim=dim
        #self.c=c



    def forward(self, x, attention_mask_quad=None):
        x=torch.abs(x)

        s=torch.sum(x, dim=self.dim, keepdim=True)

        output=x/ s

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))


        return output

class linearabsplussoftmax(nn.Module): # Linearabsplussoftmax
    def __init__(self, c=5, dim=-1):
        super(linearabsplussoftmax, self).__init__()
        self.dim=dim
        self.c=c



    def forward(self, x, attention_mask_quad=None):
        x = x + self.c

        x=torch.abs(x)

        s=torch.sum(x, dim=self.dim, keepdim=True)

        output=x/ s

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))


        return output

class trainablelinearabssoftmax(nn.Module): # trainableQuad2 w/o mask
    def __init__(self, attn, input_N_dim=None, S=None, NS=None, num_heads=None, dim=-1, usc_c=True, use_a=False,  use_d=False, use_e=False, init=False):
        super(trainablelinearabssoftmax, self).__init__()
        self.dim=dim
        #self.c=c

        self.use_a=use_a
        self.use_d = use_d
        self.usc_e= use_e
        self.usc_c=usc_c
        if self.use_a:
            if attn=='self' or attn=='linvar':
                self.a = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.a = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn=='mpcext':
                self.a = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if self.use_d:
            if attn=='self' or attn=='linvar':
                self.d = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.d = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn=='mpcext':
                self.d = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)


        if self.usc_c:
            if attn=='self' or attn=='linvar':
                self.c = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.c = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn == 'mpcext':
                self.c = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if init:
            with torch.no_grad():
                self.c.fill_(1)
                self.d.fill_(1)
                print ("init c and d")
                #print (self.d)
            self.c.requires_grad = True
            self.d.requires_grad = True




    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')

        if self.use_d:
            x = x * self.d

        if self.usc_c:
            x=x+self.c

        x=torch.abs(x)

        #x=x*attention_mask_quad
        if self.use_a:
            x = torch.square(self.a) * x

        s=torch.sum(x, dim=self.dim, keepdim=True)

        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #s = s.expand(-1, -1, -1, x.shape[-1])
        #print (f'attention score sum: {s.shape} {s[0]}')

        output=x/ s

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))

        return output

class trainableoriginalsoftmax(nn.Module): # trainableQuad2 w/o mask
    def __init__(self, attn, input_N_dim=None, S=None, NS=None, num_heads=None, dim=-1, usc_c=True, use_a=False,  use_d=False, use_e=False, init=False):
        super(trainableoriginalsoftmax, self).__init__()
        self.dim=dim
        #self.c=c

        self.use_a=use_a
        self.use_d = use_d
        self.usc_e= use_e
        self.usc_c=usc_c
        if self.use_a:
            if attn=='self' or attn=='linvar':
                self.a = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.a = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn=='mpcext':
                self.a = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if self.use_d:
            if attn=='self' or attn=='linvar':
                self.d = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.d = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn=='mpcext':
                self.d = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)


        if self.usc_c:
            if attn=='self' or attn=='linvar':
                self.c = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.c = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn == 'mpcext':
                self.c = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if init:
            with torch.no_grad():
                self.c.fill_(1)
                self.d.fill_(1)
                print ("init c and d")
                #print (self.d)
            self.c.requires_grad = True
            self.d.requires_grad = True




    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')

        if self.use_d:
            x = x * self.d

        if self.usc_c:
            x=x+self.c

        x=torch.exp(x)

        #x=x*attention_mask_quad
        if self.use_a:
            x = torch.square(self.a) * x

        s=torch.sum(x, dim=self.dim, keepdim=True)

        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #s = s.expand(-1, -1, -1, x.shape[-1])
        #print (f'attention score sum: {s.shape} {s[0]}')

        output=x/ s

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))

        return output


class sigmoidsoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(sigmoidsoftmax, self).__init__()
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
        output = torch.sigmoid(x)/(torch.sum(torch.sigmoid(x), dim=self.dim,  keepdim=True) + 1e-10)

        return output

class tanhsoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(tanhsoftmax, self).__init__()
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
        x=torch.tanh(x)
        s=torch.sum(x, dim=self.dim,  keepdim=True) + 1e-10
        output = x/s
        #print (output)

        #output = torch.relu(x) / s
        #print (torch.sum(torch.relu(output), dim=-1))
        #output = torch.tanh(x)/(torch.sum(torch.tanh(x), dim=self.dim,  keepdim=True) + 1e-10)


        return output

class relu6softmax(nn.Module):
    def __init__(self, dim=-1):
        super(relu6softmax, self).__init__()
        self.dim=dim
        self.layer=nn.ReLU6()


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
        #output = torch.relu(x)/(torch.sum(torch.relu(x), dim=self.dim,  keepdim=True) + 1e-10)
        x=self.layer(x)

        s=torch.sum(x, dim=self.dim,  keepdim=True) + 1e-10

        output=x/s

        return output

class gelusoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(gelusoftmax, self).__init__()
        self.dim=dim
        self.layer=nn.GELU()


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
        #output = torch.relu(x)/(torch.sum(torch.relu(x), dim=self.dim,  keepdim=True) + 1e-10)
        x=self.layer(x)

        s=torch.sum(x, dim=self.dim,  keepdim=True) + 1e-10

        output=x/s

        return output

class silusoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(silusoftmax, self).__init__()
        self.dim=dim
        self.layer=nn.SiLU()


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
        #output = torch.relu(x)/(torch.sum(torch.relu(x), dim=self.dim,  keepdim=True) + 1e-10)
        x=self.layer(x)

        s=torch.sum(x, dim=self.dim,  keepdim=True) + 1e-10

        output=x/s

        return output

class softplussoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(softplussoftmax, self).__init__()
        self.dim=dim
        self.layer=nn.Softplus()


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
        #output = torch.relu(x)/(torch.sum(torch.relu(x), dim=self.dim,  keepdim=True) + 1e-10)
        x=self.layer(x)

        s=torch.sum(x, dim=self.dim,  keepdim=True) + 1e-10

        output=x/s

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

class trainableReluSoftmax(nn.Module): # trainableQuad2 w/o mask
    def __init__(self, attn, input_N_dim=None, S=None, NS=None, num_heads=None, dim=-1, use_a=True, use_b=True, init=True):
        super(trainableReluSoftmax, self).__init__()
        self.dim=dim
        #self.c=c

        self.use_a=use_a
        self.use_b = use_b
        if self.use_a:
            if attn=='self':
                self.a = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.a = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn=='mpcext':
                self.a = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)

        if self.use_b:
            if attn=='self':
                self.b = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
            elif attn=='ext':
                self.b = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
            elif attn=='mpcext':
                self.b = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)




        if attn=='self':
            self.c = nn.Parameter(torch.ones(1, num_heads, input_N_dim, input_N_dim), requires_grad=True)
        elif attn=='ext':
            self.c = nn.Parameter(torch.ones(1, input_N_dim, S), requires_grad=True)
        elif attn == 'mpcext':
            self.c = nn.Parameter(torch.ones(1, S, NS), requires_grad=True)


        if init:
            with torch.no_grad():
                self.a.fill_(1)
                self.b.fill_(1)
                self.c.fill_(1)
                print ("init a=1, b=1 and c=1")
                #print (self.d)
            self.a.requires_grad = True
            self.b.requires_grad = True
            self.c.requires_grad = True

    def forward(self, x, attention_mask_quad=None):
        #print (f'softmax input shape:{x.shape}')

        if self.use_b:
            x=x+self.b

        if self.use_a:
            x = x * self.a

        x=torch.maximum(x, self.c)

        s=torch.sum(x, dim=self.dim, keepdim=True)

        #resizeshape = tuple(list(map(int, x.shape[:-1])) + [1])
        #s = torch.reshape(s, resizeshape)

        #s = s.expand(-1, -1, -1, x.shape[-1])
        #print (f'attention score sum: {s.shape} {s[0]}')

        output=x/ s

        #print (f'output: {output.shape} {output[0][0][0]}')
        #print (torch.sum(output, dim=-1))

        return output


class leakyrelusoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(leakyrelusoftmax, self).__init__()
        self.dim=dim
        #self.leakyrelu=nn.LeakyReLU(1e-2)


    def forward(self, x):
        #x=self.leakyrelu(x)
        x=torch.relu(x)

        s=torch.sum(x, dim=self.dim,  keepdim=True) + 1e-10

        output = x/s

        return output

class ScaleAttn(nn.Module):
    def __init__(self, scale, dim=-1):
        super(ScaleAttn, self).__init__()
        self.scale=scale
        self.dim=dim


    def forward(self, x):
        #print (f'softmax input shape:{x.shape}')
        assert x.shape[-1]==x.shape[-2]

        n=x.shape[-1] # n is sequence length

        x = x / self.scale # recover q * k^T
        output= x / n

        #print (output)
        return output

class singlerelusoftmax(nn.Module):
    def __init__(self, scale,dim=-1):
        super(singlerelusoftmax, self).__init__()
        self.scale = scale
        self.dim=dim


    def forward(self, x):
        assert x.shape[-1]==x.shape[-2]

        n=x.shape[-1] # n is sequence length

        x = torch.relu(x) / self.scale # recover q * k^T
        output= x / n
        return output

class squaresoftmax(nn.Module):
    def __init__(self, scale, dim=-1):
        super(squaresoftmax, self).__init__()
        self.scale = scale
        self.dim=dim


    def forward(self, x):
        assert x.shape[-1]==x.shape[-2]

        n=x.shape[-1] # n is sequence length

        x = torch.square(x) / self.scale # recover q * k^T
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

        elif softmax_approx=='trainablerelusoftmax':
            self.softmaxlayer=trainableReluSoftmax(attn='ext',
                                               input_N_dim=allargs.input_N_dim,
                                               S=self.S,
                                               dim=-2,
                                               )

        elif softmax_approx=='quad2':
            self.softmaxlayer=Quad2(c=5, dim=-2)

        elif softmax_approx=='fixquad2':
            self.softmaxlayer=fixQuad2(c=allargs.fixquad2c, dim=-2,
                                       )

        elif softmax_approx=='trainablequad2c':
            self.softmaxlayer=trainableQuad2(attn='ext',
                                               input_N_dim=allargs.input_N_dim,
                                               S=self.S,
                                               dim=-2,
                                               use_d=False,
                                               )

        elif softmax_approx=='trainablequad2cd':
            self.softmaxlayer=trainableQuad2(attn='ext',
                                               input_N_dim=allargs.input_N_dim,
                                               S=self.S,
                                               dim=-2,
                                               use_d=True,
                                               )

        elif softmax_approx=='trainablequad2cda':
            self.softmaxlayer=trainableQuad2(attn='ext',
                                               input_N_dim=allargs.input_N_dim,
                                               S=self.S,
                                               dim=-2,
                                               use_d=True,
                                               use_a=True,
                                               )

        elif softmax_approx=='trainablequad2cde':
            self.softmaxlayer=trainableQuad2(attn='ext',
                                               input_N_dim=allargs.input_N_dim,
                                               S=self.S,
                                               dim=-2,
                                               use_d=True,
                                               use_e=True,
                                               )
        elif softmax_approx=='trainablequad2cdae':
            self.softmaxlayer=trainableQuad2(attn='ext',
                                               input_N_dim=allargs.input_N_dim,
                                               S=self.S,
                                               dim=-2,
                                               use_d=True,
                                               use_e=True,
                                               use_a=True,
                                               )

        elif softmax_approx=='scaleattn':
            self.softmaxlayer=ScaleAttn(self.scale, dim=-2)

        elif softmax_approx=='relu6softmax':
            self.softmaxlayer=relu6softmax(dim=-2)

        elif softmax_approx=='sigmoidsoftmax':
            self.softmaxlayer=sigmoidsoftmax(dim=-2)

        elif softmax_approx=='biquad4':
            self.softmaxlayer=biquad4(c=5, dim=-2)

        elif softmax_approx=='biquad4woc':
            self.softmaxlayer=biquad4woc(dim=-2)

        elif softmax_approx=='trainablebiquad4c':
            self.softmaxlayer=trainableBiquad4(attn='ext',
                                               input_N_dim=allargs.input_N_dim,
                                               S=self.S,
                                               dim=-2,
                                               )

        elif softmax_approx=='trainablebiquad4ca':
            self.softmaxlayer=trainableBiquad4(attn='ext',
                                               input_N_dim=allargs.input_N_dim,
                                               S=self.S,
                                               dim=-2,
                                               use_a=True,
                                               )

        elif softmax_approx=='trainablebiquad4cd':
            self.softmaxlayer=trainableBiquad4(attn='ext',
                                               input_N_dim=allargs.input_N_dim,
                                               S=self.S,
                                               dim=-2,
                                               use_d=True,
                                               )

        elif softmax_approx=='softplussoftmax':
            self.softmaxlayer=softplussoftmax(dim=-2)

        elif softmax_approx=='linearabssoftmax':
            self.softmaxlayer=linearabssoftmax(dim=-2)

        elif softmax_approx=='linearabsplussoftmax':
            self.softmaxlayer=linearabsplussoftmax(c=5, dim=-2)

        elif softmax_approx=='trainablelinear':
            self.softmaxlayer=trainablelinear(attn='ext',
                                               input_N_dim=allargs.input_N_dim,
                                               S=self.S,
                                               dim=-2,
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

class MPC_ExternalAttention(Module):
    """
    Obtained from https://github.com/MenghaoGuo/EANet/blob/main/EAMLP/models/token_performer.py
    https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/ExternalAttention.py
    """

    def __init__(self, dim, num_heads=8, MPC_externalattention_dim=128, externalattention_dim=64, input_N_dim=256, attention_dropout=0.1, projection_dropout=0.1, softmax_approx='original', externalattention_divhead=False, allargs=None):
        super().__init__()
        print (f"MPC_ExternalAttention Attention, div head: {externalattention_divhead}")
        print (f"attention softmax:{softmax_approx}")
        self.externalattention_divhead=externalattention_divhead

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.CS=externalattention_dim
        self.NS=MPC_externalattention_dim

        self.input_N_dim=input_N_dim

        #self.qkv = Linear(dim, dim * 3, bias=False)
        self.q = Linear(dim, dim, bias=False)

        if not self.externalattention_divhead:
            self.Mk = Linear(dim, self.CS, bias=False)
            self.NMk = Linear(input_N_dim, self.NS, bias=False)
        #else:
            #self.Mk = Linear(head_dim, self.S, bias=False)

        if not self.externalattention_divhead:
            self.Mv = Linear(self.CS, dim,  bias=False)
            self.NMv = Linear(self.NS, input_N_dim, bias=False)
        #else:
            #self.Mv = Linear(self.S, head_dim, bias=False)

        self.attn_drop = Dropout(attention_dropout)
        if not self.externalattention_divhead:
            self.proj = Linear(dim, dim)
        #else:
            #self.proj = Linear(head_dim, head_dim)
        self.proj_drop = Dropout(projection_dropout)

        if softmax_approx=='original':
            self.softmaxlayer=nn.Softmax(dim=-1)

        elif softmax_approx=='relusoftmax':
            self.softmaxlayer=ReluSoftmax(dim=-1)

        elif softmax_approx=='trainablerelusoftmax':
            self.softmaxlayer=trainableReluSoftmax(attn='mpcext',
                                               S=self.CS,
                                               NS=self.NS,
                                               dim=-1,
                                               )

        elif softmax_approx=='quad2':
            self.softmaxlayer=Quad2(c=5, dim=-1)

        elif softmax_approx=='fixquad2':
            self.softmaxlayer=fixQuad2(c=allargs.fixquad2c, dim=-1,
                                       )

        elif softmax_approx=='trainablequad2c':
            self.softmaxlayer=trainableQuad2(attn='mpcext',
                                               S=self.CS,
                                               NS=self.NS,
                                               dim=-1,
                                               use_d=False,
                                               )

        elif softmax_approx=='trainablequad2cd':
            self.softmaxlayer=trainableQuad2(attn='mpcext',
                                               S=self.CS,
                                               NS=self.NS,
                                               dim=-1,
                                               use_d=True,
                                               )

        elif softmax_approx=='trainablequad2cda':
            self.softmaxlayer=trainableQuad2(attn='mpcext',
                                               S=self.CS,
                                               NS=self.NS,
                                               dim=-1,
                                               use_d=True,
                                               use_a=True,
                                               )

        elif softmax_approx=='trainablequad2cde':
            self.softmaxlayer=trainableQuad2(attn='mpcext',
                                               S=self.CS,
                                               NS=self.NS,
                                               dim=-1,
                                               use_d=True,
                                               use_e=True,
                                               )
        elif softmax_approx=='trainablequad2cdae':
            self.softmaxlayer=trainableQuad2(attn='mpcext',
                                               S=self.CS,
                                               NS=self.NS,
                                               dim=-1,
                                               use_d=True,
                                               use_e=True,
                                               use_a=True,
                                               )

        elif softmax_approx=='scaleattn':
            self.softmaxlayer=ScaleAttn(self.scale, dim=-1)

        elif softmax_approx=='relu6softmax':
            self.softmaxlayer=relu6softmax(dim=-1)

        elif softmax_approx=='sigmoidsoftmax':
            self.softmaxlayer=sigmoidsoftmax(dim=-1)

        elif softmax_approx=='biquad4':
            self.softmaxlayer=biquad4(c=5, dim=-1)

        elif softmax_approx=='biquad4woc':
            self.softmaxlayer=biquad4woc(dim=-1)

        elif softmax_approx == 'trainablebiquad4c':
            self.softmaxlayer = trainableBiquad4(attn='mpcext',
                                                 S=self.CS,
                                                 NS=self.NS,
                                                 dim=-1,
                                                 )

        elif softmax_approx == 'trainablebiquad4ca':
            self.softmaxlayer = trainableBiquad4(attn='mpcext',
                                                 S=self.CS,
                                                 NS=self.NS,
                                                 dim=-1,
                                                 use_a=True,
                                                 )

        elif softmax_approx == 'trainablebiquad4cd':
            self.softmaxlayer = trainableBiquad4(attn='mpcext',
                                                 S=self.CS,
                                                 NS=self.NS,
                                                 dim=-1,
                                                 use_d=True,
                                                 )

        elif softmax_approx=='softplussoftmax':
            self.softmaxlayer=softplussoftmax(dim=-1)

        elif softmax_approx=='linearabssoftmax':
            self.softmaxlayer=linearabssoftmax(dim=-1)

        elif softmax_approx=='linearabsplussoftmax':
            self.softmaxlayer=linearabsplussoftmax(c=5, dim=-1)

        elif softmax_approx=='trainablelinear':
            self.softmaxlayer=trainablelinear(attn='mpcext',
                                               S=self.CS,
                                               NS=self.NS,
                                               dim=-1,
                                               )





    def forward(self, x):
        B, N, C = x.shape # N is # of patches

        #print (f"B, N, C: {x.shape}")
        assert self.input_N_dim==N

        if not self.externalattention_divhead:
            x=self.q(x) # x: B, N, C
        #else:
            #x = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # x: B, Head, N, C//Head

        idn = x[:]
        #x = x.view(B, N, -1)
        #print (f"after view: {x.shape}")

        #print (f"before Mk: {x.shape}")
        x=self.Mk(x) # attn: B, N, CS | divhead: B, H, N, S
        #print (f"after Mk: {x.shape}")
        x=x.permute(0, 2, 1) # B, CS, N
        #print (f"after permute: {x.shape}")

        attn=self.NMk(x) # B, CS, NS
        #print (f"after NMk: {attn.shape}")



        #print (f"after Mk: {x.shape}")
        attn = self.softmaxlayer(attn) # B, CS, NS | divhead: B, H, N, S
        #print (f"sum dim: {attn.sum(dim=-1, keepdim=True).shape}") # B, NS, 1
        #print (attn.sum(dim=-1, keepdim=True)[0][0])

        attn = attn / (1e-9 + attn.sum(dim=-2, keepdim=True))
        #print (f"after softmax: {attn.shape}") # B, CS, NS
        #print (f"check after softmax: {attn.sum(dim=-1)}")


        attn = self.attn_drop(attn)

        x=self.NMv(attn) # B, CS, N

        x = x.permute(0, 2, 1)  # B, N, CS
        #print (f"after NMv: {x.shape}")
        x = self.Mv(x) # B, N, C | divhead: B, H, N, C//H
        #print (f"after Mv: {x.shape}")



        x = self.proj(x)
        x = self.proj_drop(x)

        # residual connection
        x=idn+x

        #if self.externalattention_divhead:
            #x=x.permute(0, 2, 1, 3).view(B, N, -1)

        return x

class LinVarAttention(Module):

    def __init__(self, dim, num_heads=8, linvar_dim=128, input_N_dim=256, attention_dropout=0.1, projection_dropout=0.1, softmax_approx='original', allargs=None,):
        super().__init__()
        print (f"LinVarAttention, div head: {True}")
        print (f"attention softmax:{softmax_approx}")

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.NS=linvar_dim

        self.input_N_dim = input_N_dim

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.Pq = Linear(input_N_dim, self.NS, bias=False)
        self.Pk = Linear(input_N_dim, self.NS, bias=False)
        self.Rq = Linear(self.NS, input_N_dim, bias=False)
        self.Rk = Linear(self.NS, input_N_dim, bias=False)



        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

        if softmax_approx=='original':
            self.softmaxlayer=nn.Softmax(dim=-1)

        elif softmax_approx=='relusoftmax':
            self.softmaxlayer=ReluSoftmax()

        elif softmax_approx=='trainablerelusoftmax':
            self.softmaxlayer=trainableReluSoftmax(attn='linvar',
                                               input_N_dim=linvar_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               )


        elif softmax_approx=='quad2':
            self.softmaxlayer=Quad2(c=5)

        elif softmax_approx=='fixquad2':
            self.softmaxlayer=fixQuad2(c=allargs.fixquad2c,
                                       )

        elif softmax_approx=='trainablequad2c':
            self.softmaxlayer=trainableQuad2(attn='linvar',
                                               input_N_dim=linvar_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=False,
                                               init=allargs.initcd,
                                               )

        elif softmax_approx=='trainablequad2cd':
            self.softmaxlayer=trainableQuad2(attn='linvar',
                                               input_N_dim=linvar_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               init=allargs.initcd,
                                               )

        elif softmax_approx=='trainablequad2cda':
            self.softmaxlayer=trainableQuad2(attn='linvar',
                                               input_N_dim=linvar_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               use_a=True,
                                               )

        elif softmax_approx=='trainablequad2cde':
            self.softmaxlayer=trainableQuad2(attn='linvar',
                                               input_N_dim=linvar_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               use_e=True,
                                               )
        elif softmax_approx=='trainablequad2cdae':
            self.softmaxlayer=trainableQuad2(attn='linvar',
                                               input_N_dim=linvar_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               use_e=True,
                                               use_a=True,
                                               )

        elif softmax_approx=='scaleattn':
            self.softmaxlayer=ScaleAttn(self.scale)

        elif softmax_approx=='linearsoftmax':
            self.softmaxlayer=linearsoftmax()

        elif softmax_approx=='leakyrelusoftmax':
            self.softmaxlayer=leakyrelusoftmax()

        elif softmax_approx=='cube3':
            self.softmaxlayer=Cube3(c=5)

        elif softmax_approx=='tanhsoftmax':
            self.softmaxlayer=tanhsoftmax()

        elif softmax_approx=='sigmoidsoftmax':
            self.softmaxlayer=sigmoidsoftmax()

        elif softmax_approx=='biquad4':
            self.softmaxlayer=biquad4(c=5)

        elif softmax_approx=='biquad4woc':
            self.softmaxlayer=biquad4woc()

        elif softmax_approx == 'trainablebiquad4c':
            self.softmaxlayer = trainableBiquad4(attn='linvar',
                                                 input_N_dim=linvar_dim,
                                                 num_heads=self.num_heads,
                                                 dim=-1,
                                                 )

        elif softmax_approx == 'trainablebiquad4ca':
            self.softmaxlayer = trainableBiquad4(attn='linvar',
                                                 input_N_dim=linvar_dim,
                                                 num_heads=self.num_heads,
                                                 dim=-1,
                                                 use_a=True,
                                                 )

        elif softmax_approx == 'trainablebiquad4cd':
            self.softmaxlayer = trainableBiquad4(attn='linvar',
                                                 input_N_dim=linvar_dim,
                                                 num_heads=self.num_heads,
                                                 dim=-1,
                                                 use_d=True,
                                                 )

        elif softmax_approx=='linearabssoftmax':
            self.softmaxlayer=linearabssoftmax()

        elif softmax_approx=='linearabsplussoftmax':
            self.softmaxlayer=linearabsplussoftmax(c=5)

        elif softmax_approx=='singlerelusoftmax':
            self.softmaxlayer=singlerelusoftmax(self.scale)

        elif softmax_approx=='squaresoftmax':
            self.softmaxlayer=squaresoftmax(self.scale)

        elif softmax_approx=='relu6softmax':
            self.softmaxlayer=relu6softmax()

        elif softmax_approx=='gelusoftmax':
            self.softmaxlayer=gelusoftmax()

        elif softmax_approx=='silusoftmax':
            self.softmaxlayer=silusoftmax()

        elif softmax_approx=='softplussoftmax':
            self.softmaxlayer=softplussoftmax()

        elif softmax_approx=='mpcpolynomialsoftmax_sigmoid_1':
            self.softmaxlayer=mpcpolynomialsoftmax_sigmoid_1()

        elif softmax_approx=='trainablelinear':
            self.softmaxlayer=trainablelinear(attn='linvar',
                                               input_N_dim=linvar_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               )


    def forward(self, x):
        B, N, C = x.shape # N is # of patches

        assert self.input_N_dim == N

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, num_head, N, C//num_head

        #print (f"before proj: q:{q.shape}, k: {v.shape}")

        q=self.Pq(q.transpose(-2, -1)) # B, num_head, C//num_head, NS
        k=self.Pk(k.transpose(-2, -1)) # B, num_head, C//num_head, NS

        #print (f"after proj: q:{q.shape}, k: {v.shape}")




        #attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (q.transpose(-2, -1) @ k) * self.scale
        #print (f"attn: {attn.shape}")

        # B, num_head, NS, NS
        #print (f"before softmax: {attn.shape}")
        attn=self.softmaxlayer(attn)

        attn=self.Rq(attn)
        attn=self.Rk(attn.transpose(-2, -1))

        #print (f"after recovery: attn: {attn.shape}")

        #attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1, softmax_approx='original', learnable_c=None,learnable_d=None, allargs=None,):
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

        elif softmax_approx=='trainableoriginalsoftmax':
            self.softmaxlayer=trainableoriginalsoftmax(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=False,
                                               )

        elif softmax_approx=='trainableoriginalsoftmaxc':
            self.softmaxlayer=trainableoriginalsoftmax(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=False,
                                               )

        elif softmax_approx=='trainableoriginalsoftmaxd':
            self.softmaxlayer=trainableoriginalsoftmax(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               usc_c=False,
                                               use_d=True,
                                               )

        elif softmax_approx=='trainableoriginalsoftmaxcd':
            self.softmaxlayer=trainableoriginalsoftmax(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               init=allargs.initcd,
                                               )

        elif softmax_approx=='trainableoriginalsoftmaxcda':
            self.softmaxlayer=trainableoriginalsoftmax(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               use_a=True,
                                               )

        elif softmax_approx=='relusoftmax':
            self.softmaxlayer=ReluSoftmax()

        elif softmax_approx=='trainablerelusoftmax':
            self.softmaxlayer=trainableReluSoftmax(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               )

        elif softmax_approx=='quad2':
            self.softmaxlayer=Quad2(c=5)

        elif softmax_approx=='fixquad2':
            self.softmaxlayer=fixQuad2(c=allargs.fixquad2c,
                                       )

        elif softmax_approx=='softmaxreciprocalapprox':
            self.softmaxlayer=softmaxreciprocalapprox(c=5)

        elif softmax_approx=='trainablesoftmaxreciprocalapprox':
            self.softmaxlayer=trainablesoftmaxreciprocalapprox(c=5)


        elif softmax_approx=='softmaxreciprocalfree1':
            self.softmaxlayer=softmaxreciprocalfree1(c=5)

        elif softmax_approx=='softmaxreciprocalfree2':
            self.softmaxlayer=softmaxreciprocalfree2(c=5)

        elif softmax_approx=='softmaxreciprocalfree3':
            self.softmaxlayer=softmaxreciprocalfree3(c=5)

        elif softmax_approx=='trainablequad2c':
            self.softmaxlayer=trainableQuad2(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=False,
                                               )

        elif softmax_approx=='trainablequad2d':
            self.softmaxlayer=trainableQuad2(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                                usc_c=False,
                                               use_d=True,

                                               )

        elif softmax_approx=='trainablequad2cd':
            self.softmaxlayer=trainableQuad2(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               init=allargs.initcd,
                                               )

        elif softmax_approx=='trainablequad2cdheadlevel':
            self.softmaxlayer=trainableQuad2headlevel(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               )

        elif softmax_approx=='trainablequad2cdtokenlevel':
            self.softmaxlayer=trainableQuad2tokenlevel(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               )

        elif softmax_approx=='trainablequad2cdlayerlevel':
            self.softmaxlayer=trainableQuad2layerlevel(
                                               dim=-1,
                                               c=learnable_c,
                                               d=learnable_d,
                                               )

        elif softmax_approx=='trainablequad2cdallone':
            self.softmaxlayer=trainableQuad2layerlevel(
                                               dim=-1,
                                               c=learnable_c,
                                               d=learnable_d,
                                               )

        elif softmax_approx=='trainablequad2cda':
            self.softmaxlayer=trainableQuad2(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               use_a=True,
                                               )

        elif softmax_approx=='trainablequad2cde':
            self.softmaxlayer=trainableQuad2(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               use_e=True,
                                               )
        elif softmax_approx=='trainablequad2cdae':
            self.softmaxlayer=trainableQuad2(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               use_e=True,
                                               use_a=True,
                                               )

        elif softmax_approx=='scaleattn':
            self.softmaxlayer=ScaleAttn(self.scale)

        elif softmax_approx=='linearsoftmax':
            self.softmaxlayer=linearsoftmax()

        elif softmax_approx=='leakyrelusoftmax':
            self.softmaxlayer=leakyrelusoftmax()

        elif softmax_approx=='cube3':
            self.softmaxlayer=Cube3(c=5)

        elif softmax_approx=='tanhsoftmax':
            self.softmaxlayer=tanhsoftmax()

        elif softmax_approx=='sigmoidsoftmax':
            self.softmaxlayer=sigmoidsoftmax()

        elif softmax_approx=='biquad4':
            self.softmaxlayer=biquad4(c=5)

        elif softmax_approx=='biquad4woc':
            self.softmaxlayer=biquad4woc()

        elif softmax_approx == 'trainablebiquad4c':
            self.softmaxlayer = trainableBiquad4(attn='self',
                                                 input_N_dim=allargs.input_N_dim,
                                                 num_heads=self.num_heads,
                                                 dim=-1,
                                                 )

        elif softmax_approx == 'trainablebiquad4ca':
            self.softmaxlayer = trainableBiquad4(attn='self',
                                                 input_N_dim=allargs.input_N_dim,
                                                 num_heads=self.num_heads,
                                                 dim=-1,
                                                 use_a=True,
                                                 )

        elif softmax_approx == 'trainablebiquad4cd':
            self.softmaxlayer = trainableBiquad4(attn='self',
                                                 input_N_dim=allargs.input_N_dim,
                                                 num_heads=self.num_heads,
                                                 dim=-1,
                                                 use_d=True,
                                                 )

        elif softmax_approx == 'trainablebiquad4cda':
            self.softmaxlayer = trainableBiquad4(attn='self',
                                                 input_N_dim=allargs.input_N_dim,
                                                 num_heads=self.num_heads,
                                                 dim=-1,
                                                 use_a=True,
                                                 use_d=True,
                                                 )

        elif softmax_approx=='linearabssoftmax':
            self.softmaxlayer=linearabssoftmax()

        elif softmax_approx=='trainablelinearabssoftmaxc':
            self.softmaxlayer=trainablelinearabssoftmax(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=False,
                                               )

        elif softmax_approx=='trainablelinearabssoftmaxd':
            self.softmaxlayer=trainablelinearabssoftmax(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               usc_c=False,
                                               use_d=True,
                                               )

        elif softmax_approx=='trainablelinearabssoftmaxcd':
            self.softmaxlayer=trainablelinearabssoftmax(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               init=allargs.initcd,
                                               )

        elif softmax_approx=='trainablelinearabssoftmaxcda':
            self.softmaxlayer=trainablelinearabssoftmax(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               use_d=True,
                                               use_a=True,
                                               )

        elif softmax_approx=='linearabsplussoftmax':
            self.softmaxlayer=linearabsplussoftmax(c=5)

        elif softmax_approx=='singlerelusoftmax':
            self.softmaxlayer=singlerelusoftmax(self.scale)

        elif softmax_approx=='squaresoftmax':
            self.softmaxlayer=squaresoftmax(self.scale)

        elif softmax_approx=='relu6softmax':
            self.softmaxlayer=relu6softmax()

        elif softmax_approx=='gelusoftmax':
            self.softmaxlayer=gelusoftmax()

        elif softmax_approx=='silusoftmax':
            self.softmaxlayer=silusoftmax()

        elif softmax_approx=='softplussoftmax':
            self.softmaxlayer=softplussoftmax()

        elif softmax_approx=='mpcpolynomialsoftmax_sigmoid_1':
            self.softmaxlayer=mpcpolynomialsoftmax_sigmoid_1()

        elif softmax_approx=='trainablelinear':
            self.softmaxlayer=trainablelinear(attn='self',
                                               input_N_dim=allargs.input_N_dim,
                                               num_heads=self.num_heads,
                                               dim=-1,
                                               )


    def forward(self, x):
        B, N, C = x.shape # N is # of patches
        #print (f"B, N, C: {B, N, C}")

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, num_head, N, C//num_head


        attn = (q @ k.transpose(-2, -1)) * self.scale

        # B, num_head, N_path, N_path
        #print (f"before softmax, B, num_head, N_path, N_path: {attn.shape}")
        attn=self.softmaxlayer(attn)

        #attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MaskedAttention(Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            assert mask.shape[-1] == attn.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn.masked_fill_(~mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1, softmax_approx='original',
                 attention_mechanism='original', externalattention_dim=64, input_N_dim=256,
                 externalattention_divhead=False, learnable_c=None,
                 learnable_d=None, allargs=None, ):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)

        if attention_mechanism=='original':
            if softmax_approx=='trainablequad2cdlayerlevel':
                learnable_c=nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
                learnable_d = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
                self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                        attention_dropout=attention_dropout, projection_dropout=dropout, softmax_approx=softmax_approx,
                                           learnable_c=learnable_c, learnable_d=learnable_d, allargs=allargs)
            elif softmax_approx=='trainablequad2cdallone':
                self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                        attention_dropout=attention_dropout, projection_dropout=dropout, softmax_approx=softmax_approx,
                                           learnable_c=learnable_c, learnable_d=learnable_d, allargs=allargs)
            else:
                self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                        attention_dropout=attention_dropout, projection_dropout=dropout, softmax_approx=softmax_approx, allargs=allargs)
        elif attention_mechanism=='externalattention':
            self.self_attn = ExternalAttention(dim=d_model,  num_heads=4, externalattention_dim=externalattention_dim, externalattention_divhead=externalattention_divhead,
                                       attention_dropout=attention_dropout, projection_dropout=dropout, softmax_approx=softmax_approx, allargs=allargs)

        elif attention_mechanism=='mpcexternalAttention':
            self.self_attn = MPC_ExternalAttention(dim=d_model,  num_heads=nhead, externalattention_dim=externalattention_dim, input_N_dim=input_N_dim, externalattention_divhead=externalattention_divhead,
                                       attention_dropout=attention_dropout, projection_dropout=dropout, softmax_approx=softmax_approx, allargs=allargs)

        elif attention_mechanism=='linvarattention':
            self.self_attn = LinVarAttention(dim=d_model, num_heads=allargs.num_heads, input_N_dim=allargs.input_N_dim,
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


class MaskedTransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(MaskedTransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = MaskedAttention(dim=d_model, num_heads=nhead,
                                         attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, mask=None, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src), mask))
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
                 input_N_dim=256,
                 externalattention_divhead=False,
                 allargs=None,
                 ):
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

        if softmax_approx == 'trainablequad2cdallone':
            learnable_c = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
            learnable_d = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)


        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]

        if softmax_approx == 'trainablequad2cdallone':
            learnable_c = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
            learnable_d = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True)
            self.blocks = ModuleList([
                TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                        dim_feedforward=dim_feedforward, dropout=dropout,
                                        attention_dropout=attention_dropout, drop_path_rate=dpr[i], softmax_approx=softmax_approx,
                                        attention_mechanism=attention_mechanism, externalattention_dim=externalattention_dim,
                                        input_N_dim=input_N_dim, externalattention_divhead=externalattention_divhead,
                                        allargs=allargs,
                                        learnable_c=learnable_c,
                                        learnable_d=learnable_d
                                        )
                for i in range(num_layers)])

        else:
            self.blocks = ModuleList([
                TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                        dim_feedforward=dim_feedforward, dropout=dropout,
                                        attention_dropout=attention_dropout, drop_path_rate=dpr[i], softmax_approx=softmax_approx,
                                        attention_mechanism=attention_mechanism, externalattention_dim=externalattention_dim,
                                        input_N_dim=input_N_dim, externalattention_divhead=externalattention_divhead,
                                        allargs=allargs,
                                        )
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


class MaskedTransformerClassifier(Module):
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
                 positional_embedding='sine',
                 seq_len=None,
                 *args, **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert seq_len is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            seq_len += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                seq_len += 1  # padding idx
                self.positional_emb = Parameter(torch.zeros(1, seq_len, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(seq_len,
                                                                          embedding_dim,
                                                                          padding_idx=True),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            MaskedTransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                          dim_feedforward=dim_feedforward, dropout=dropout,
                                          attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x, mask=None):
        if self.positional_emb is None and x.size(1) < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            if mask is not None:
                mask = torch.cat([torch.ones(size=(mask.shape[0], 1), device=mask.device), mask.float()], dim=1)
                mask = (mask > 0)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

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
    def sinusoidal_embedding(n_channels, dim, padding_idx=False):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(0)
        if padding_idx:
            return torch.cat([torch.zeros((1, 1, dim)), pe], dim=1)
        return pe
