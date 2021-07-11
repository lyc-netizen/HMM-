# mark函数：为每个词（即变量si）标注状态
def mark(si):
    mark_=[]
    # 若该词的长度为1，判断其为独立成词，状态为Single('S')
    if(len(si)==1):
        mark_.append('S')
    # 否则，将该词的状态设定为['B','M',...,'M','E']('B':Begin;'M':Middle,'E':'End')
    else:
        mark_+=['B']+['M']*(len(si)-2)+['E']
    return mark_
# get_data_lines函数：从位于data_path目录的文件中提取每一行并封装为列表
def get_data_lines(data_path):
    # 为了消除文本开头的特殊字符\ufeff，设置encoding为utf-8-sig
    fp = open(data_path, 'r', encoding='utf-8-sig')
    return fp.readlines()
# HMM模型
class HMM(object):
    # 类初始化函数：初始化HMM模型的参数
    def __init__(self):
        # 文段里每个字有四种状态：
        # B:Begin（在词开头）
        # M：Middle（在词中间）
        # E：End（在词结尾）
        # S：Single（作为单字词）
        self.status=['B','M','E','S']
        # 状态转移概率矩阵，即A矩阵，用字典表示，其key为[前一个状态][当前状态]。
        self.A_dict={}
        # 观测概率矩阵，即B矩阵，用字典表示，其key为[当前状态][当前状态对应的字]
        self.B_dict={}
        # 初始概率矩阵，即π矩阵，用字典表示，其key为[当前状态]
        self.Pi_dict={}
    # HMM.HMMtrain函数：从已经分好词的文段中学习A、B、Pi矩阵。
    def HMMtrain(self,train_path):
        # 计算每个状态出现的次数，字典形式，其key为状态'B''M''E''S'
        status_count={}
        # A、Pi、B的学习初值
        for st in self.status:
            self.A_dict[st]={s:0.0 for s in self.status}
            self.Pi_dict[st]=0.0
            self.B_dict[st]={}
            status_count[st]=0
        # 获得训练数据集，以一行为元素的列表
        train_data=get_data_lines(train_path)
        # 在处理第lis行文本
        lis=0
        # 处理每一行文本
        for line in train_data:
            # 正在处理的行数+1，这里是从第1行开始算，以便于下面将初始状态转化为概率
            lis+=1
            # 去掉文本前后的特殊字符，例如换行符
            line=line.strip()
            # 如果这一行为空
            if not line:
                continue
            # ji_list：这一行含有的字表
            # 例如说，如果这一行是“所以 我 常常 要 到 那 园子 里 去”
            # 则得到的字表为['所','以','我','常','常','要','到','那','园','子','里','去']
            ji_list=[]
            for w in line:
                # ji_list不包含用于分词的空格，但是包含标点符号
                if w!=' ':
                    ji_list.append(w)
            # si_list：这一行含有的词表
            # 仍然以“所以 我 常常 要 到 那 园子 里 去”为例子
            # 得到的词表是['所以','我','常常','要','到','那','园子','里','去']
            si_list=line.split()
            # ji_status：每个字的状态（'B/M/S/E'之一种）
            # 将每个词传入mark函数，得到的是该词含有的每个字的状态的列表
            # 例如说：mark(‘如果’）=['BE'];mark('旅游业')=['BME']；
            # mark('现代控制原理基础')=['BMMMMMME'];mark('的')=['S']
            # 对于一行，以“所以 我 常常 要 到 那 园子 里 去”为例
            # 得到的每个字状态为['B','E','S','B','E','S','S','S','B','E','S','S']
            ji_status=[]
            for si in si_list:
                # 由于mark(tshi)返回值也是列表，因此使用extend函数将列表融合（attend会将列表当成一个新的元素）
                ji_status.extend(mark(si))
            # enumerate返回一个列表，其元素是二元组(i,ji_status[i])
            for index,ji_st in enumerate(ji_status):
                # 统计对应的状态数+1
                status_count[ji_st]+=1
                if index==0:
                    # 统计初始状态(即index等于0，每行首个字符），对应的状态数+1
                    self.Pi_dict[ji_st]+=1
                else:
                    # 统计状态转移次数，A的key为[前一个状态][当前状态]，从前一个状态到当前状态的转移+1
                    self.A_dict[ji_status[index-1]][ji_st]+=1
                    # 统计状态观测次数，B的key为[当前状态][当前状态对应的字]，当前字观测到当前状态的次数+1
                    # 等号右侧，查询B_dict[ji_st][ji_list[index]]，如果不存在这个二元key（因为在这个train函数中，
                    # 上面初始化A的时候同时分配了两个key，但初始化B的时候只分配了前一个key，第二个key由于取自文段中的字，
                    # 会在计算B矩阵的时候动态分配，因此初始化的时候没有顾及），则get函数返回其第二个参数
                    # 也就是0，等号右侧为1；如果存在，则等号右侧相当于等号左侧自+1.
                    self.B_dict[ji_st][ji_list[index]]=self.B_dict[ji_st].get(ji_list[index],0)+1
        # 将初始状态矩阵转化为概率，将每个状态在lis行累积出现的次数除以lis
        for key,value in self.Pi_dict.items():
            self.Pi_dict[key]=value/lis
        # 将状态转移次数转化为概率，除以（转移前的）状态总数
        for key,value in self.A_dict.items():
            for key1,value1 in value.items():
                self.A_dict[key][key1]=value1/status_count[key]
        # 将状态观测次数转化为概率，除以key对应的状态总数
        for key,value in self.B_dict.items():
            for key1,value1 in value.items():
                # 加1平滑，否则可能会在viterbi函数中报错
                self.B_dict[key][key1]=(value1+1)/status_count[key]
    # viterbi函数求解HMM最佳路径
    # 注意：对一行文本
    def viterbi(self,text):
        # delta见统计学习方法209页的变量δ
        delta=[{}]
        # 最佳路径
        path={}
        # 初始化变量delta和path
        for st in self.status:
            # 若B_dict[st][text[0]]存在，则返回此值，否则返回0（表示之前训练的时候没有见过这个字）
            delta[0][st]=self.Pi_dict[st]*self.B_dict[st].get(text[0],0)
            path[st]=st
        #从第2个字开始（即除去初始状态以外）
        for i in range(1,len(text)):
            delta.append({})
            new_path={}
            # 检查这个字是不是不在B矩阵的第二个key里，也就是这个字是不是在之前训练的时候没有见过
            not_in_B_dict=text[i] not in self.B_dict['B'].keys() and text[i] not in self.B_dict['M'].keys() \
                    and text[i] not in self.B_dict['S'].keys() and text[i] not in self.B_dict['E'].keys()
            # 遍历每一个状态
            for st in self.status:
                if not not_in_B_dict:
                    # 如果之前训练的时候见过这个字，则Pro_B等于b_i(o_i)
                    Pro_B=self.B_dict[st].get(text[i],0)
                else:
                    # 如果见过，则等于0
                    Pro_B=0
                # 见统计学习方法209页：
                # 以下t为第t个状态，等效于本代码中的第i个字符
                # delta_t_i=max{1<=j<=N}[delta_t-1(j)*a[j][i]]b_i(o_t)
                # phi=argmax{1<=j<=N}[delta_t-1(j)*a[j][i]]
                (pro,phi)=max([(delta[i-1][st_]*self.A_dict[st_].get(st,0)*Pro_B,st_) for st_ in self.status if delta[i-1][st_]>0])
                delta[i][st]=pro
                # 更新路径，添加当前的状态
                new_path[st]=path[phi]+st
            path=new_path
        # 更新最后一个P*和phi
        (pro, phi) = max([(delta[len(text) - 1][st_], st_) for st_ in self.status])
        # 返回最优路径的最大概率和最优路径
        return pro,path[phi]
    # 分词函数
    def divide(self,text):
        # 从测试文本中获得最大概率和最优路径
        prob,path=self.viterbi(text)
        # 这个分词开始的地方和下一个分词开始的地方
        begin, next = 0, 0
        # 遍历这一文本（行）的每个字
        for i, char in enumerate(text):
            pos = path[i]
            # 如果这个字被判断为状态B，将它指定为分词开始
            if pos == "B":
                begin = i
            # 如果这个字被判断为状态E，将它指定为分词结束，获取begin到该字符的子串作为分词之一
            # 将该字符的下一个位置作为下一次分词的开始
            elif pos == "E":
                yield text[begin:i + 1]
                next = i + 1
            # 如果这个字被判断为状态S，将它指定为单字成词，获取该字作为分词之一
            elif pos == 'S':
                yield char
                next = i + 1
        # 将该行剩余的字符作为一个分词
        if next < len(text):
            yield text[next:]
# main是测试函数
def main():
    # HMM模型
    H=HMM()
    # 训练的数据集是pku_training.utf8。获得A、B、Pi矩阵。
    H.HMMtrain('pku_training.utf8')
    # 测试数据集是novel.txt，打开编码同样是utf-8-sig，避免\ufeff等特殊字符的产生
    test_f = open("novel.txt", "r", encoding="utf-8-sig")
    # 读取，获得以每一行内容为元素的列表
    test_text = test_f.readlines()
    # 去除每一行的换行符
    for i in range(len(test_text)):
        test_text[i]=test_text[i].strip()
    # 显示原文
    print("原文：")
    for i in range(len(test_text)):
        print(test_text[i])
    print('----------------------------------------------------')
    # 显示分词结果
    print("分词结果：")
    for i in range(len(test_text)):
        # H.divide(test_text[i])获得分词结果（yield关键字获得一个generator对象，其可迭代，内容是分词结果）
        for j in H.divide(test_text[i]):
            print(j,end='/')
        print('\n',end='')
main()



