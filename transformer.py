# 在学习Transformer完理论后，对其代码如何实现完全不知。
# 所以利用一个简单的例子来完成Transformer的Pytorch的实现
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# S: 表示decoding输入的开始符号
# E: 表示decoding输出结束符号
# P: 如果当前批次数据大小小于时间步长，将填充空白序列的符号
sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
# 手动构建词向量 注意Decoder和Encoder的单词不能放在一起构建词向量
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(src_vocab) # 表示Encoder词向量的个数

tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}   # Decoder的词向量
idx2word = {i: w for i, w in enumerate(tgt_vocab)}  # 构建字典 根据词向量映射单词
tgt_vocab_size = len(tgt_vocab) # Decoder词向量的个数

src_len = 5  # 表示Encoder输入的一个句子中最多的单词个数
tgt_len = 6  # 表示Decoder输入（输出）的一个句子中最多的单词个数

# Transformer Parameters Transformer参数 重要参数
d_model = 512  # 字嵌入&位置嵌入的维度，这俩值是相同的，因此用一个变量就行了 表示词向量的维度以及位置向量的维度
# FeedForward dimension 表示Feed Forward隐藏层神经元的个数
d_ff = 2048
# Q、K、V向量的维度，其中Q与K的维度必须相等，
# V的维度没有限制，不过为了方便起见，都设为64
d_k = d_v = 64  # dimension of K(=Q), V
# Encoder和Decoder的个数
n_layers = 6
# 多头注意力中head的数量
n_heads = 8

"""
make_data(sentences)函数
参数：sentences为Encoder要输入的数据（包括Encoder输入 Decoder输入 Decoder输出）
函数作用：将每个部分分词 根据构建的单词对应字典 将单词转换成对应的词向量。
注意将Encoder Deconder 不同的部分拆开来赋值
return：将得到的矩阵转化成Tensor 使用torch.LongTensor(xxx)进行转化
"""
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

# 得到各个部分对应的词向量
"""
enc_inputs:Encoder输入部分
dec_inputs:Decoder输入部分
dec_outputs:Decoder输出部分
"""
enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

"""
创建dataset对象 继承Data.Dataset 
必须包含__init__,__len__和__getitem__函数

"""
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

# 创建dataloader对象
# 这里batch_size=2
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

# Positional Encoding
# 参数：n_position字库的大小 d_model位置编码的维度
def get_sinusoid_encodingg_table(n_position,d_model):
        def cal_angle(position,hid_idx):
                return position/np.power(10000,2*(hid_idx//2)/d_model)
        def get_posi_angle_vec(position):
                return [cal_angle(position,hid_j) for hid_j in range(d_model)]
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2]) # dim 2i
        sinusoid_table[:,1::2] = np.cos(sinusoid_table[:,1::2]) # dim 2i+1
        return torch.FloatTensor(sinusoid_table)

# 实现padding mask
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    由于在Encoder和Decoder中都需要进行mask操作，
    因此就无法确定这个函数的参数中seq_len的值，
    如果是在Encoder中调用的，seq_len就等于src_len；
    如果是在Decoder中调用的，seq_len就有可能等于src_len，
    也有可能等于tgt_len（因为Decoder有两次mask）
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


# 实现sequence_mask
def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 实现上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# 封装点积运算
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        batch_size:批处理个数
        n_heads:多头注意力中head的数量
        d_k:表示k的维度
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        # transpose(-1,-2)：表示将K装置
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        """
        masked_fill_(mask, value)方法
        其中mask是张量，元素是布尔值， value是要填充的值。该方法会在mask中为True的位置上填充value值。
        mask和value的形状要么是相同的， 要么是可以进行广播的， 否则会报错。
        """
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

# Multi-Head Attention实现
# 封装成类
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        """
        nn.Linear(in_features,out_features,bias)
        in_features – 每个输入样本的大小。
        out_features – 每个输出样本的大小。
        bias – 如果设置为False，该层将不会学习附加偏差。默认为：True。
        """
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # residual:为了之后进行ResNet
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        #
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model)(output + residual), attn


#Feed Forward 实现
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model)(output + residual)  # [batch_size, seq_len, d_model]

# 一个EncoderLayer实现
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        """
        刚开始的时候，都是由enc_inputs转化为QKV；
        得到加权求和后的值以及对应的权值；
        然后经过一个全连接层得到第一个Encoder的输出
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encodingg_table(src_vocab_size, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        word_emb = self.src_emb(enc_inputs)
        pos_emb = self.pos_emb(enc_inputs)
        enc_outputs = word_emb + pos_emb
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


# Decoder Layer的封装
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs:表示Decoder Layer的输入
        enc_outputs:表示Encoder Layer的输出
        dec_self_attn_mask:表示第一个Multi-Head Attention的Mask
        dec_enc_attn_mask:表示第二个Multi-Head Attention的Mask

        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


# 封装6个Decoder Layer成Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encodingg_table(tgt_vocab_size, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        word_emb = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        pos_emb = self.pos_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = word_emb + pos_emb
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequent_mask = get_attn_subsequence_mask(dec_inputs) # [batch_size, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0) # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

# 组合成Transformer
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
        # print("2")

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        # print("3")
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # print("4")
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        # print("1")
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


# 模型
model = Transformer()
# 损失函数 交叉熵
# pytorch中的CrossEntropyLoss()函数其实就是把输出结果进行sigmoid（将数据设置到0-1之间），
# 随后再放到传统的交叉熵函数中，就会得到结果。
criterion = nn.CrossEntropyLoss(ignore_index=0)
# 优化器 随机梯度下降算法 加 动量
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
# print("end...")
# 训练
for epoch in range(100):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        enc_inputs, dec_inputs, dec_outputs = enc_inputs, dec_inputs, dec_outputs
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        # print(enc_inputs,dec_inputs)
        # print("edn...")
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

        loss = criterion(outputs, dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        # 标准三步
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def greedy_decoder(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype)], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["."]:
            terminal = True
        print(next_word)
    return dec_input


# Test
enc_inputs, _, _ = next(iter(loader))
enc_inputs = enc_inputs
for i in range(len(enc_inputs)):
    greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"])
    predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])













