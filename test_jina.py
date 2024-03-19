# -*- coding: utf-8 -*-
# ---
# @File: test_jina.py
# @Author: Jiahuan Luo
# @Institution: Webank, Shenzhen, China
# @E-mail: luojiahuan001@gmail.com
# @Time: 2024/3/18
# ---
from transformers import AutoModel, BertConfig, BertModel
from numpy.linalg import norm

cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
model = AutoModel.from_pretrained('official_model/download/jina-embeddings-v2-base-zh', trust_remote_code=True) # trust_remote_code is needed to use the encode method
embeddings = model.encode(['How is the weather today?', '今天天气怎么样?'])
print(cos_sim(embeddings[0], embeddings[1]))