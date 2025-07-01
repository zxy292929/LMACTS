import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain.schema import HumanMessage, SystemMessage
from OperatorsSelect import OperatorsSelect
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["ZHIPUAI_API_KEY"] = "abac97da27af5add9e167776e9004986.7UIrxBHels5LncbL"
class OperatorImplementation:
    def OperatorImplementation(unseen_dataset_desc):
        OperatorsList = OperatorsSelect.OperatorsSelect(unseen_dataset_desc)
        print("OperatorsList:",OperatorsList)
        OperatorsText = '\n'.join([f"{idx+1}.{layer}" for idx, layer in enumerate(OperatorsList)])
        print("OperatorsText:",OperatorsText)
        langchain_query = ChatZhipuAI(temperature=1, model='glm-4-plus')
        RoleSetting='''You're a programmer. Your task is to refer to the following example, according to the given operator name to implement the specific code. Note that some operators require several functions to implement.\n
        The following are the examples of code implementations based on operator names:\n
        1.Operator name: skip_connect\n
        Function name: Identity()\n
        Specific code implementation:\n
        class Identity(nn.Module):\n

            def __init__(self):\n
                super(Identity, self).__init__()\n\n

            def forward(self, x):\n
                return x\n

        2.Operator name: dcc_1\n
        Function name: DCCLayer(C, C, (1, 2), dilation=1)\n
        Specific code implementation:\n
        class CausalConv2d(nn.Conv2d):\n

            def __init__(self, in_channels,\n
                     out_channels,\n
                     kernel_size,\n
                     stride=1,\n
                     dilation=1,\n
                     groups=1,\n
                     bias=True):\n
                self._padding = (kernel_size[-1] - 1) * dilation\n
                super(CausalConv2d, self).__init__(in_channels,\n
                                           out_channels,\n
                                           kernel_size=kernel_size,\n
                                           stride=stride,\n
                                           padding=(0, self._padding),\n
                                           dilation=dilation,\n
                                           groups=groups,\n
                                           bias=bias)\n

            def forward(self, input):\n
                result = super(CausalConv2d, self).forward(input)\n
                if self._padding != 0:\n
                    return result[:, :, :, :-self._padding]\n
                return result\n


        class DCCLayer(nn.Module):\n
        """\n
        dilated causal convolution layer with GLU function\n
        """\n

            def __init__(self, c_in, c_out, kernel_size, dilation=1):\n
                super(DCCLayer, self).__init__()\n
                self.filter_conv = CausalConv2d(c_in, c_out, kernel_size, dilation=dilation)\n
                self.gate_conv = CausalConv2d(c_in, c_out, kernel_size, dilation=dilation)\n

            def forward(self, x):\n
            """\n
            :param x: [batch_size, f_in, N, T]\n
            :return:\n
            """\n
                filter = self.filter_conv(x)\n
                gate = torch.sigmoid(self.gate_conv(x))\n
                output = filter * gate\n

            return output\n
        '''
#         3.Operator name: diff_gcn\n
#         Function name: DiffusionConvLayer(2, supports, nodevec1, nodevec2, C, C, dropout)\n
#         Specific code implementation:\n
#         class linear(nn.Module):\n
#         """\n
#         Linear for 2d feature map\n
#         """\n
#             def __init__(self, c_in, c_out):\n
#                 super(linear, self).__init__()\n
#                 self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))  # bias=True\n
#             def forward(self, x):\n
#             """\n
#             :param x: [batch_size, f_in, N, T]\n
#             :return:\n
#             """\n
#                 return self.mlp(x)\n

#         class nconv(nn.Module):\n

#             def __init__(self):\n
#                 super(nconv, self).__init__()\n

#             def forward(self, x, A):\n
#                 x = torch.einsum('ncvl, vw->ncwl', (x, A))\n
#                 return x.contiguous()\n
        
#         class DiffusionConvLayer(nn.Module):\n
#         """\n
#         K-order diffusion convolution layer with self-adaptive adjacency matrix (N, N)\n
#         """\n

#             def __init__(self, K, supports, nodevec1, nodevec2, c_in, c_out, dropout=False):\n
#                 super(DiffusionConvLayer, self).__init__()\n
#                 c_in = (K * (len(supports) + 1) + 1) * c_in\n
#                 self.nodevec1 = nodevec1\n
#                 self.nodevec2 = nodevec2\n
#                 self.mlp = linear(c_in, c_out).to(DEVICE)  # 7 * 32 * 32\n
#                 self.c_out = c_out\n
#                 self.K = K\n
#                 self.supports = supports\n
#                 self.nconv = nconv()\n
#                 self.dropout = dropout\n

#             def forward(self, x):\n
#                 adp = F.relu(torch.mm(self.nodevec1, self.nodevec2)) \n
#                 mask = torch.zeros_like(adp) - 10 ** 10\n
#                 adp = torch.where(adp == 0, mask, adp)\n
#                 adp = F.softmax(adp, dim=1)\n
#                 new_supports = self.supports + [adp]\n

#                 out = [x]\n
#                 for a in new_supports:\n
#                     x1 = self.nconv(x, a)\n
#                     out.append(x1)\n
#                     for k in range(2, self.K + 1):\n
#                         x2 = self.nconv(x1, a)\n
#                         out.append(x2)\n
#                         x1 = x2\n

#                 h = torch.cat(out, dim=1)\n
#                 h = self.mlp(h)\n
#                 if self.dropout:\n
#                     h = F.dropout(h, 0.3, training=self.training)\n

#                 return h\n    

        # 4.Operator name: trans\n
        # Function name: InformerLayer(C)\n
        # Specific code implementation:\n
        # class AttentionLayer(nn.Module):\n
        #     def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):\n
        #         super(AttentionLayer, self).__init__()\n

        #         d_keys = d_keys or (d_model // n_heads)\n
        #         d_values = d_values or (d_model // n_heads)\n

        #         self.inner_attention = attention\n
        #         self.query_projection = nn.Linear(d_model, d_keys * n_heads)\n
        #         self.key_projection = nn.Linear(d_model, d_keys * n_heads)\n
        #         self.value_projection = nn.Linear(d_model, d_values * n_heads)\n
        #         self.out_projection = nn.Linear(d_values * n_heads, d_model)\n
        #         self.n_heads = n_heads\n
        #         self.mix = mix\n

        #     def forward(self, queries, keys, values, attn_mask):\n
        #         B, L, _ = queries.shape\n
        #         _, S, _ = keys.shape\n
        #         H = self.n_heads\n

        #         queries = self.query_projection(queries).view(B, L, H, -1)\n
        #         keys = self.key_projection(keys).view(B, S, H, -1)\n
        #         values = self.value_projection(values).view(B, S, H, -1)\n

        #         out, attn = self.inner_attention(\n
        #             queries,\n
        #             keys,\n
        #             values,\n
        #             attn_mask\n
        #         )\n
        #         if self.mix:\n
        #             out = out.transpose(2, 1).contiguous()\n
        #         out = out.view(B, L, -1)\n

        #         return self.out_projection(out), attn\n
            
        # class TriangularCausalMask():\n
        #     def __init__(self, B, L):\n
        #         mask_shape = [B, 1, L, L]\n
        #         with torch.no_grad():\n
        #             self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(DEVICE)\n

        #     @property\n
        #     def mask(self):\n
        #         return self._mask\n
        
        # class FullAttention(nn.Module):\n
        #     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):\n
        #         super(FullAttention, self).__init__()\n
        #         self.scale = scale\n
        #         self.mask_flag = mask_flag\n
        #         self.output_attention = output_attention\n
        #         self.dropout = nn.Dropout(attention_dropout)\n

        #     def forward(self, queries, keys, values, attn_mask):\n
        #         B, L, H, E = queries.shape\n
        #         _, S, _, D = values.shape\n
        #         scale = self.scale or 1. / math.sqrt(E)\n

        #         scores = torch.einsum("blhe,bshe->bhls", queries, keys)\n
        #         if self.mask_flag:\n
        #             if attn_mask is None:\n
        #                 attn_mask = TriangularCausalMask(B, L)\n

        #             scores.masked_fill_(attn_mask.mask, -np.inf)\n

        #         A = self.dropout(torch.softmax(scale * scores, dim=-1))\n
        #         V = torch.einsum("bhls,bshd->blhd", A, values)\n

        #         if self.output_attention:\n
        #             return (V.contiguous(), A)\n
        #         else:\n
        #             return (V.contiguous(), None)\n

        # class InformerLayer(nn.Module):\n
        #     def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):\n
        #         super(InformerLayer, self).__init__()\n
        #         self.attention = AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)\n
        #         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)\n
        #         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)\n
        #         self.norm1 = nn.LayerNorm(d_model)\n
        #         self.norm2 = nn.LayerNorm(d_model)\n
        #         self.dropout = nn.Dropout(dropout)\n
        #         self.activation = F.relu if activation == "relu" else F.gelu\n
        #         self.d_model = d_model\n

        #     def forward(self, x, attn_mask=None):\n
        #         b, C, N, T = x.shape\n
        #         x = x.permute(0, 2, 3, 1)\n
        #         x = x.reshape(-1, T, C)\n

        #         new_x, attn = self.attention(\n
        #         x, x, x,\n
        #         attn_mask=attn_mask\n
        #         )\n
        #         x = x + self.dropout(new_x)\n

        #         y = x = self.norm1(x)\n
        #         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))\n
        #         y = self.dropout(self.conv2(y).transpose(-1, 1))\n
        #         output = self.norm2(x+y)\n

        #         output = output.reshape(b, -1, T, C)\n
        #         output = output.permute(0, 3, 1, 2)\n

        #         return output\n

            
        #     '''
        input=f'''Now, your task is to implement the operator names given below using the examples above.\n 
        Take a deep breath and solve this problem step by step:\n
        The code you design will go through this step:hidden_state2 = Op(hidden_state1)\n
        All you have to do is implement the "Op".\n
        The dimension of both Hidden_state2 and hidden_state1 must be [64,32,7,12].\n
        If there are parameters such as in_channels, out_channels, kernel_size, please assign them to yourself.\n
        Operator name:\n
        {OperatorsText}\n
        Your answer should be in the following format: \n
        1.Operator name:[operator_name_1]\n
        Function name: [function_name_1]\n
        Specific code implementation:[specific_code_implementation_1]\n
        2.Operator name:[operator_name_2]\n
        Function name: [function_name_2]\n
        Specific code implementation:[specific_code_implementation_2]\n
        3.Operator name:[operator_name_3]\n
        Function name: [function_name_3]\n
        Specific code implementation:[specific_code_implementation_3]\n
        4.Operator name:[operator_name_4]\n
        Function name: [function_name_4]\n
        Specific code implementation:[specific_code_implementation_4]\n
        5.Operator name:[operator_name_5]\n
        Function name: [function_name_5]\n
        Specific code implementation:[specific_code_implementation_5]\n
        6.Operator name:[operator_name_6]\n
        Function name: [function_name_6]\n
        Specific code implementation:[specific_code_implementation_6]  \n      
        7.Operator name:[operator_name_7]\n
        Function name: [function_name_7]\n
        Specific code implementation:[specific_code_implementation_7]\n
        8.Operator name:[operator_name_8]\n
        Function name: [function_name_8]\n
        Specific code implementation:[specific_code_implementation_8]\n
        9.Operator name:[operator_name_9]\n
        Function name: [function_name_9]\n
        Specific code implementation:[specific_code_implementation_9]      \n  
        10.Operator name:[operator_name_10]\n
        Function name: [function_name_10]\n
        Specific code implementation:[specific_code_implementation_10]\n  
        '''
        messages = [
            SystemMessage(content=RoleSetting),
            HumanMessage(content=input)
        ]
        response = langchain_query.invoke(messages)   
        print("messages:", messages)
        return response.content
def main():
    unseen_dataset_desc = '''unseen_dataset_desc: Dataset Type: Unseen\n
Temporal Features:\n
- Temporal Granularity: 1\n
- Time Span: 9862\n
- Cyclic Patterns: -\n
- Cycle Length: -\n\n

Statistical Features:\n
- Sum Values: 5298.189\n
- Mean Change: -0.000003371029\n
- Mean Second Derivative Central: 0.00000004662725\n
- Median: 0.6952419\n
- Mean: 0.6982326\n
- Length: 7588.0\n
- Standard Deviation: 0.08045370\n
- Variation Coefficient: 0.1152248\n
- Variance: 0.006472797\n
- Skewness: -0.1517160\n
- Kurtosis: -0.6799092\n
- Root Mean Square: 0.7028524\n
- Absolute Sum Of Changes: 16.68794\n
- Longest Strike Below Mean: 2762.0\n'''
    result = OperatorImplementation.OperatorImplementation(unseen_dataset_desc)
    print("Result:", result)

# Run the main function
if __name__ == "__main__":
    main()