from elasticsearch import Elasticsearch, helpers
import numpy as np
from tqdm import tqdm, trange
import re


#--------------------------------------------------------------------#
RE_D = re.compile('\d')
def has_numbers(string, regex_pattern=RE_D):
    return bool(regex_pattern.search(string))

#--------------------------------------------------------------------#

with open('./data/hit_stopwords.txt', 'r', encoding='utf-8') as fr:
    stopwords = fr.readlines()
    stopwords = [x.strip() for x in stopwords]
    STOPWORDS = set([stopword for stopword in stopwords if stopword])
    
class Keyword():
    def __init__(self) -> None:
        # 索引到搜索引擎
        self.es = Elasticsearch(['IP:PORT'], timeout=1000, max_retries=10, retry_on_timeout=True,
                                http_auth=("elastic", "PASSWORD"))
        self.index = "YOUR_INDEX"

        
    # 查找特定ID的tf-idf信息
    def get_tfidfs(self, ids, index='', fields='split_text'):
        if not index:
            index = self.index
        if not (index and fields and ids):
            print('PLZ secify all params')
        res = []
        batchsize = 1  # 如果大于1 这里返回时乱序 一定注意 需要额外逻辑去保证顺序
        for i in trange(len(ids)//batchsize+1):
            qids = ids[i*batchsize:(i+1)*batchsize]
            # print('qids',qids)
            if not qids:
                break
            res.extend(self.es.mtermvectors(index=self.index, ids=qids,
                    fields=[fields], term_statistics=True)['docs'])
        # print(len(res))
        return res
    
    def sort_filter(self, sent, tfidf_dict): # 输入需要的句子和字典，返回通过字典筛选后的词
        pos_dict = {} # 记录位置和需要的(词, tfidf)对
        for key, val in tfidf_dict.items():
            index = sent.find(key)
            if key not in pos_dict or len(key) > len(pos_dict[index][0]):  #如果该词不存在或者出现重叠且该词更长
                pos_dict[index] = (key, val)
        res = list(pos_dict.items())
        res.sort() # 根据位置排序
        res = [x[1] for x in res]
        return res
        
    def keyword_generate(self, sents, ids = ''):

        tf_idfs = self.get_tfidfs(ids) 
        N = tf_idfs[0]['term_vectors']['split_text']['field_statistics']['doc_count'] # 任取一个获得doc_count
        
        with open('key_word_result.txt', 'a', encoding='utf-8') as fw: 
            for i, sent_info in enumerate(tf_idfs): 
                tmp_list = [] # 保留需要的词和tfidf
                tmp_dict = {} # 记录出现过的词位置，长度，方便筛选排序
                words_info = sent_info['term_vectors']['split_text']['terms']
                for word, word_info in words_info.items():
                    df = word_info['doc_freq']
                    idf = np.log(N / df) # 逆文档频率
                    tf = word_info['term_freq'] # 最原始的tf计算方式
                    word_tf_idf = tf * idf
                    if  not has_numbers(word):  tmp_dict[word] = word_tf_idf; # tmp_list.append( (word,word_tf_idf))
                    tmp_list = self.sort_filter(sents[i], tmp_dict) # 对词进行排序和筛选
                tmp_list_tfidf = [x[1] for x in tmp_list]
                
                ind = np.argsort(tmp_list_tfidf)[::-1] # 为了避免扰乱初始顺序 采用index排序
                ind = ind[:(len(sents[i])//5)] # 每5个字一个关键词
                ind.sort()
                tmp_list = [tmp_list[i] for i in ind] 
                
                # D = D[D > threshold] 最好还通过最低tfidf筛选部分数据
                tmp_list = [x for x in tmp_list if x[1] > 2]
                tmp_list = [x for x in tmp_list if x[0] not in STOPWORDS]
                if not tmp_list: break; # 如果没有关键词 这条跳过
                tmp_list = [x[0] for x in tmp_list] # 去掉分数
                tmp_list = ' '.join(tmp_list) # 写出空格分隔的str形式
                
                # print(tmp_list + '[SEP]' + sents[i] + '[SEP]')
                fw.write(tmp_list + '[SEP]' + sents[i] + '[SEP]' + '\n')
#                 tmp_list = tmp_list 
#                 tmp_list.append(sents[i])

    def key_generate_byScan(self):
        ids = []
        sents = []
        batch_size = 1000
        for i, hit in enumerate(tqdm(helpers.scan(self.es, index=self.index))):
            ids.append(hit['_id'])
            sents.append(hit['_source']["raw_text"])
            # print(hit['_source']["split_text"])
            if i % batch_size == 0 and i > 0:
                self.keyword_generate(sents, ids)
                ids = []
                sents = []
            # if i == 100: break
        
        if ids: # 如果还有
            self.keyword_generate(sents, ids)
        return
            
if __name__ == '__main__':
    keyword_extractor = Keyword()
    keyword_extractor.key_generate_byScan()