# TODO

- [X] 完善retrieve模块，让模型选择怎么retrieve
  - [X] 利用神经元之间的连接强度来进行retrieve
  - [X] 如何在检索后更新神经元的连接强度
  - [X] 如何产生一个新的engram
  - [X] 怎么让LLM自己判断是否需要retrieve
  - [X] 可能需要在每次启动的时候保留一点上次的对话
  - [X] 可能需要在persona里加一个简要的信息
- [X] 需要定义一个List[Engram] 或者 Dict[UUID1, Engram]的类来简化代码
- [X] reflection逻辑还需要优化，怎么自动做reflection
- [X] 怎么存储
  - [X] 现在把embedding都存到json文件里，显然有点冗余，但是不存embedding又会很慢，后面把embedding单独提取出来存到一个地方
  - [X] 存到mongoDB里
- [ ] 考虑是否需要给用户也设置一个agent和schedule类并存储下来
- [ ] 加入schedule和experience的更改逻辑
- [X] 同时和多个用户对话

### 神经元之间的连接

神经元之间的连接的可能用处：（指的是原版的chat序列、chat和thought、engram代表的连接）

- 用于merge、delete神经元等
  - 可以在reflection或者转成engram的时候计算每个神经元和代表神经元的强度，过低的可以删掉
- 用于retrieve
  - retrieve根据相似度和强度来retrieve最相似的，而连接则在于返回的时候可以返回上下文
  - 后续思考怎么通过连接强度来检索；Markov chain是一个思路但是太复杂

### schedule还需要处理

将schedule的更新加入到现有的system中：

- 用schedule记录特殊的行程还是用一个event记录，还是都要，或者给schedule一个从experience event到schedule event的映射

### elo的想法

 本质上还是处理神经元之间的连接，每个神经元都有一个elo值。
 可以把一次retrieve进来的query当作是一次信号的传递，那么按照engram本来的定义就是几个神经元去竞争这个信号获得激活，这些神经元的信号就会保留下来形成这个事件的印迹。

elo是可以用来刻画这种竞争的，某种程度上就相当于上面提到的strength。

现在假设每个神经元有一个初始elo，当一个信号进来的时候，（这里暂时把query的embedding当成信号），每个神经元可以根据自己的embedding去计算相似度。将这些相似度做一个归一化，比如softmax，作为一个初始的信号分配。之后就是竞争信号的过程。将这个初始的信号权重乘elo，得到了该神经元在这次竞争中的战斗力。在排名检索之后根据这个结果来更新原始的elo权重。这样就是一个动态更新strength的方法，他会比每次检索到都简单加一个delta值会更平衡。

这样的合理之处在于，engram有一个summary信息，当我们通过summary检索到engram的时候，如果我们发现这个检索的summary不足以完成任务的时候我们需要进一步对engram的细节信息进行检索。因为engram本身是一个有主题的事件，通过这样的操作可以更集中在与engram主题相关的神经元上。

举例来说，就好像我的engram是我上周去看了eason的演唱会。如果每个人都问我他安可唱了什么歌，那么相应的神经元elo就会提高。这说明当我需要检索我去看了eason演唱会这个事件的时候，有很大概率别人会继续问我他安可唱了什么。这提供了一种更主动的检索方式。
