from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import re
import numpy as np
import gudhi as gd

#读取文件    主要就是证明了其对称性
with open("lhvertices.txt","r",encoding="utf-8") as f:
    data = f.readlines()#根据打开的文件按行读取数据
pattern = re.compile("\[(.*?)\]")
data_vertices = []
for value in data:
    temp = pattern.findall(value.strip())[0].split(" ")
    data_vertices.append([float(i) for i in temp if i != ""])#vertices的内容，特别是坐标点的内容都是浮点数，所以用float(i)
data_vertices = np.array(data_vertices)
with open("lhtriangles.txt","r",encoding="utf-8") as f:
    data = f.readlines()
data_triangles = []
for value in data:
    data_triangles.append([int(i) for i in value.split(" ") if i != ""][1:])#triangle 里面都是整数，所以用int(i)
    # split() 通过指定分隔符对字符串进行切片，str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等。
data_triangles = np.array(data_triangles)
# 创建绘图对象
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")#111表明只有一个图  121就是显示在两个图中得左图  122就是显示在两个图中得右图
ax.plot_trisurf(data_vertices[:,0],data_vertices[:,1],data_vertices[:,2],triangles = data_triangles)
#ax.plot_trisurf(coords[:,0], coords[:,1], coords[:,2], triangles=triangles)
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.view_init(0, 0)#(-90, 90)
plt.title('brain_graph')
plt.savefig('brain_graph.jpg')
plt.show()


st = gd.SimplexTree()#创建一个空的简单复合体。
"""st = gd.RipsComplex(points=X, max_edge_length=1.).create_simplex_tree(max_dimension=2)"""
#len(data_triangles): 20480
for i in range(len(data_triangles)):
    #print("data_triangles[i,:]:",data_triangles[i,:])#data_triangles[i,:]: [10161  9918 10241]
    st.insert([v for v in data_triangles[i,:]], -10)##Simplicies可以挨个插入,顶点由整数索引
for i in range(len(data_vertices)):
    #例如，我们为每个单纯形分配其维数作为过滤值
    st.assign_filtration([i], data_vertices[i,1])
'''＃获取所有简化程序的列表    注意，如果一条边不在顶点中，则插入边会自动插入其顶点'''
_ = st.make_filtration_non_decreasing()


'''计算过滤后的复杂体的持久性图,默认情况下，它停在1维，使用persistence_dim_max = True
计算所有维度的同源性'''
dgm = st.persistence(persistence_dim_max=True)
""" dgm = st.persistence()"""
#print("dgm:",dgm)# 有0，1，2维，0是点 1是线段 2是三角形
#但是只有一个2维  (2, (78.57119751, inf))   其余的0，1还是正常的 (1, (7.01657343, 29.42458725))  (0, (-117.12751007, inf))
#＃绘制一个持久性图
gd.plot_persistence_diagram([pt for pt in dgm if pt[0] == 0])
"""plot = gd.plot_persistence_diagram(dgm)"""
plt.title('persistence diagram with point')
plt.show()
gd.plot_persistence_diagram(dgm)
plt.title('plot_persistence_diagram')
plt.show()
gd.plot_persistence_barcode(dgm)
plt.title('plot_persistence_barcode')
plt.show()


# 获取所有simplex的列表
# 注意，如果一条边不在顶点中，则插入边会自动插入其顶点
# 与之前的问题相同：手臂在普通持久性图中不生成任何点，并且连接的组件具有无限的持久性。 因此，让我们尝试扩展持久性！
############扩展持久性  extend persistence##########
st.extend_filtration()
'''将持久性同源性扩展到代数拓扑中的其他基本概念，例如同调性和相对同源性/同调性，是很自然的'''
dgms = st.extended_persistence(min_persistence=1e-5)
'''gd.plot_persistence_barcode(dgms)
plt.title('plot_extended_persistence_barcode')
plt.show()'''
#print("dgms:",dgms)# 有0，1，2维，0是点 1是线段 2是三角形
# (0, (-23.34847831999997, -23.26331519999998)) (1, (7.016573430000008, 29.42458725000003))  (2, (-65.2574234, -73.67274474999999))
#您可能已经注意到，扩展的持久性输出了四个持久性图，而不仅仅是一个。
#这是因为拓扑结构（在扩展的持久性模块的分解中更正式地称为求和）已被划分为
#在旧过滤条件下出生和死亡的那些（called ordinary），在旧过滤条件下出生的和死亡的那些。
#在新的锥形过滤器中死亡（called extened）
#而在新的锥形过滤器中死亡并死亡的那些（called relative）。
#此外，由于新锥形过滤的阈值正在降低，因此延长和相对点的死亡时间实际上可以小于出生时间（实际上，相对点的死亡时间总是更短）。
#因此，扩展的拓扑结构也分为出生时间短于死亡时间的那些（称为extended +）和死亡时间短于出生时间的那些（称为extended-）
#同样，这些臂应该在维度1中生成拓扑结构（因为它们创建了相对同源的循环），其中一条腿应该在维度0中创建一个普通结构（另一条腿实际上是整个连接组件的一部分），
#而连接 组件和形状内部的空隙应分别创建尺寸为0和2的扩展结构。
"""横轴出生、纵轴死亡"""
fig, axs = plt.subplots(2, 2, figsize=(10,10))
'''总共创建了四幅图片，因此在第一幅图片里面创建散点图，前两行是长度相同的x,y的输入数据  '''
axs[0,0].scatter([dgms[0][i][1][0] for i in range(len(dgms[0])) if dgms[0][i][0] == 0],
                 [dgms[0][i][1][1] for i in range(len(dgms[0])) if dgms[0][i][0] == 0])
"""b = a[i:j]   表示复制a[i]到a[j-1]，以生成新的list对象
a = [0,1,2,3,4,5,6,7,8,9]
b = a[1:3]   # [1,2]"""
axs[0,0].plot([-1,1],[-1,1])
axs[0,0].set_title("Ordinary 0 PD(persistence dimention) dgms[0]")

axs[0,1].scatter([dgms[1][i][1][0] for i in range(len(dgms[1])) if dgms[1][i][0] == 1],
                 [dgms[1][i][1][1] for i in range(len(dgms[1])) if dgms[1][i][0] == 1])
axs[0,1].plot([-1,1],[-1,1])
axs[0,1].set_title("Relative 1 PD(persistence dimention) dgms[1]")

axs[1,0].scatter([dgms[2][i][1][0] for i in range(len(dgms[2])) if dgms[2][i][0] == 0],
                 [dgms[2][i][1][1] for i in range(len(dgms[2])) if dgms[2][i][0] == 0])
axs[1,0].plot([-1,1],[-1,1])
axs[1,0].set_title("Extended+ 0 PD dgms[2]")

axs[1,1].scatter([dgms[3][i][1][0] for i in range(len(dgms[3])) if dgms[3][i][0] == 2],
                 [dgms[3][i][1][1] for i in range(len(dgms[3])) if dgms[3][i][0] == 2])
axs[1,1].plot([-1,1],[-1,1])
axs[1,1].set_title("Extended- 2 PD dgms[3]")
plt.show()


# 确实，我们可以清楚地看到相对1持久性图中的两个分支，以及其他持久性图中的拓扑结构！
# 对称定理
# 另外，请注意，对于像这样的3D形状的平滑流形，存在用于扩展持久性的对称定理，在此进行描述（请参见第7节）。
# 基本上，由于我们3D形状的固有尺寸为2（它是一个表面），因此对于所有尺寸而言，一个都应具有
#对于所有的维度r,直至反射到较小的对角线（即，变换(x,y)->(-y,-x))
#和/或横跨主对角线的反射（即变换(x,y)->(y,x))
# (1) {Ordinary}_r(f) = {Ordinary}_{1−r}(−f)
# (2) {Extended}_r(f) = {Extended}_{2−r}(−f)
# (3) {Relative}_r(f) = {Relative}_{3−r}(−f)
######与最上面时一样的
st2 = gd.SimplexTree()
for i in range(len(data_triangles)):
    st2.insert([v for v in data_triangles[i,:]], -10)
for i in range(len(data_vertices)):
    st2.assign_filtration([i], -data_vertices[i,1])
_ = st2.make_filtration_non_decreasing()
#下面时扩展持续必备的
st2.extend_filtration()
dgms2 = st2.extended_persistence(min_persistence=1e-5)


'''# Let's check (1) for r = 0 and r=1.
# r=0 :dgms[0][i][0] == 0    dgms2[0][i][0] == 0 一般持续和扩展持续
fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs[0,0].scatter([dgms[0][i][1][0] for i in range(len(dgms[0])) if dgms[0][i][0] == 0],
                 [dgms[0][i][1][1] for i in range(len(dgms[0])) if dgms[0][i][0] == 0])
axs[0,0].plot([-1,1],[-1,1])
axs[0,0].set_title("Ordinary 0 PD(f)")

axs[0,1].scatter([dgms2[0][i][1][0] for i in range(len(dgms2[0])) if dgms2[0][i][0] == 1],
                 [dgms2[0][i][1][1] for i in range(len(dgms2[0])) if dgms2[0][i][0] == 1])
axs[0,1].plot([-1,1],[-1,1])
axs[0,1].set_title("Ordinary 1 PD(-f)")

axs[1,0].scatter([dgms[0][i][1][0] for i in range(len(dgms[0])) if dgms[0][i][0] == 1],
                 [dgms[0][i][1][1] for i in range(len(dgms[0])) if dgms[0][i][0] == 1])
axs[1,0].plot([-1,1],[-1,1])
axs[1,0].set_title("Ordinary 1 PD(f)")

axs[1,1].scatter([dgms2[0][i][1][0] for i in range(len(dgms2[0])) if dgms2[0][i][0] == 0],
                 [dgms2[0][i][1][1] for i in range(len(dgms2[0])) if dgms2[0][i][0] == 0])
axs[1,1].plot([-1,1],[-1,1])
axs[1,1].set_title("Ordinary 0 PD(-f)")
plt.show()


# Let's check (3) for r = 1 and r=2.
fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs[0,0].scatter([dgms[1][i][1][0] for i in range(len(dgms[1])) if dgms[1][i][0] == 1],
                 [dgms[1][i][1][1] for i in range(len(dgms[1])) if dgms[1][i][0] == 1])
axs[0,0].plot([-1,1],[-1,1])
axs[0,0].set_title("Relative 1 PD(f)")

axs[0,1].scatter([dgms2[1][i][1][0] for i in range(len(dgms2[1])) if dgms2[1][i][0] == 2],
                 [dgms2[1][i][1][1] for i in range(len(dgms2[1])) if dgms2[1][i][0] == 2])
axs[0,1].plot([-1,1],[-1,1])
axs[0,1].set_title("Relative 2 PD(-f)")

axs[1,0].scatter([dgms[1][i][1][0] for i in range(len(dgms[1])) if dgms[1][i][0] == 2],
                 [dgms[1][i][1][1] for i in range(len(dgms[1])) if dgms[1][i][0] == 2])
axs[1,0].plot([-1,1],[-1,1])
axs[1,0].set_title("Relative 2 PD(f)")

axs[1,1].scatter([dgms2[1][i][1][0] for i in range(len(dgms2[1])) if dgms2[1][i][0] == 1],
                 [dgms2[1][i][1][1] for i in range(len(dgms2[1])) if dgms2[1][i][0] == 1])
axs[1,1].plot([-1,1],[-1,1])
axs[1,1].set_title("Relative 1 PD(-f)")
plt.show()


# Let's check (2) for r = 0 and r=2.
fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs[0,0].scatter([dgms[2][i][1][0] for i in range(len(dgms[2])) if dgms[2][i][0] == 0] +
                 [dgms[3][i][1][0] for i in range(len(dgms[3])) if dgms[3][i][0] == 0],
                 [dgms[2][i][1][1] for i in range(len(dgms[2])) if dgms[2][i][0] == 0] +
                 [dgms[3][i][1][1] for i in range(len(dgms[3])) if dgms[3][i][0] == 0])
axs[0,0].plot([-1,1],[-1,1])
axs[0,0].set_title("Extended 0 PD(f)")

axs[0,1].scatter([dgms2[2][i][1][0] for i in range(len(dgms2[2])) if dgms2[2][i][0] == 2] +
                 [dgms2[3][i][1][0] for i in range(len(dgms2[3])) if dgms2[3][i][0] == 2],
                 [dgms2[2][i][1][1] for i in range(len(dgms2[2])) if dgms2[2][i][0] == 2] +
                 [dgms2[3][i][1][1] for i in range(len(dgms2[3])) if dgms2[3][i][0] == 2])
axs[0,1].plot([-1,1],[-1,1])
axs[0,1].set_title("Extended 2 PD(-f)")

axs[1,0].scatter([dgms[2][i][1][0] for i in range(len(dgms[2])) if dgms[2][i][0] == 2] +
                 [dgms[3][i][1][0] for i in range(len(dgms[3])) if dgms[3][i][0] == 2],
                 [dgms[2][i][1][1] for i in range(len(dgms[2])) if dgms[2][i][0] == 2] +
                 [dgms[3][i][1][1] for i in range(len(dgms[3])) if dgms[3][i][0] == 2])
axs[1,0].plot([-1,1],[-1,1])
axs[1,0].set_title("Extended 2 PD(f)")

axs[1,1].scatter([dgms2[2][i][1][0] for i in range(len(dgms2[2])) if dgms2[2][i][0] == 0] +
                 [dgms2[3][i][1][0] for i in range(len(dgms2[3])) if dgms2[3][i][0] == 0],
                 [dgms2[2][i][1][1] for i in range(len(dgms2[2])) if dgms2[2][i][0] == 0] +
                 [dgms2[3][i][1][1] for i in range(len(dgms2[3])) if dgms2[3][i][0] == 0])
axs[1,1].plot([-1,1],[-1,1])
axs[1,1].set_title("Extended 0 PD(-f)")
plt.show()

# 此外，这项工作还证明了一些其他的对称结果（同样，直到以上定义的反射）：
# (4) {Ordinary}_r(f)   =  {Relative}_{r+1}(-f)
# (5) {Extended}_r(f)   =  {Extended}_r(-f)
# (6) {Relative}_r(f)   =  {Ordinary}_{r-1}(-f)
# Let's check (4) for r = 0 and r=1.
fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs[0,0].scatter([dgms[0][i][1][0] for i in range(len(dgms[0])) if dgms[0][i][0] == 0],
                 [dgms[0][i][1][1] for i in range(len(dgms[0])) if dgms[0][i][0] == 0])
axs[0,0].plot([-1,1],[-1,1])
axs[0,0].set_title("Ordinary 0 PD(f)")
axs[0,1].scatter([dgms2[1][i][1][0] for i in range(len(dgms2[1])) if dgms2[1][i][0] == 1],
                 [dgms2[1][i][1][1] for i in range(len(dgms2[1])) if dgms2[1][i][0] == 1])
axs[0,1].plot([-1,1],[-1,1])
axs[0,1].set_title("Relative 1 PD(-f)")
axs[1,0].scatter([dgms[0][i][1][0] for i in range(len(dgms[0])) if dgms[0][i][0] == 1],
                 [dgms[0][i][1][1] for i in range(len(dgms[0])) if dgms[0][i][0] == 1])
axs[1,0].plot([-1,1],[-1,1])
axs[1,0].set_title("Ordinary 1 PD(f)")
axs[1,1].scatter([dgms2[1][i][1][0] for i in range(len(dgms2[1])) if dgms2[1][i][0] == 2],
                 [dgms2[1][i][1][1] for i in range(len(dgms2[1])) if dgms2[1][i][0] == 2])
axs[1,1].plot([-1,1],[-1,1])
axs[1,1].set_title("Relative 2 PD(-f)")
plt.show()

# Let's check (6) for $r = 1$ and $r=2$.
fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs[0,0].scatter([dgms[1][i][1][0] for i in range(len(dgms[1])) if dgms[1][i][0] == 1],
                 [dgms[1][i][1][1] for i in range(len(dgms[1])) if dgms[1][i][0] == 1])
axs[0,0].plot([-1,1],[-1,1])
axs[0,0].set_title("Relative 1 PD(f)")
axs[0,1].scatter([dgms2[0][i][1][0] for i in range(len(dgms2[0])) if dgms2[0][i][0] == 0],
                 [dgms2[0][i][1][1] for i in range(len(dgms2[0])) if dgms2[0][i][0] == 0])
axs[0,1].plot([-1,1],[-1,1])
axs[0,1].set_title("Ordinary 0 PD(-f)")
axs[1,0].scatter([dgms[1][i][1][0] for i in range(len(dgms[1])) if dgms[1][i][0] == 2],
                 [dgms[1][i][1][1] for i in range(len(dgms[1])) if dgms[1][i][0] == 2])
axs[1,0].plot([-1,1],[-1,1])
axs[1,0].set_title("Relative 2 PD(f)")
axs[1,1].scatter([dgms2[0][i][1][0] for i in range(len(dgms2[0])) if dgms2[0][i][0] == 1],
                 [dgms2[0][i][1][1] for i in range(len(dgms2[0])) if dgms2[0][i][0] == 1])
axs[1,1].plot([-1,1],[-1,1])
axs[1,1].set_title("Ordinary 1 PD(-f)")
plt.show()

# Let's check (5) for $r = 0$ and $r=2$.
fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs[0,0].scatter([dgms[2][i][1][0] for i in range(len(dgms[2])) if dgms[2][i][0] == 0] +
                 [dgms[3][i][1][0] for i in range(len(dgms[3])) if dgms[3][i][0] == 0],
                 [dgms[2][i][1][1] for i in range(len(dgms[2])) if dgms[2][i][0] == 0] +
                 [dgms[3][i][1][1] for i in range(len(dgms[3])) if dgms[3][i][0] == 0])
axs[0,0].plot([-1,1],[-1,1])
axs[0,0].set_title("Extended 0 PD(f)")
axs[0,1].scatter([dgms2[2][i][1][0] for i in range(len(dgms2[2])) if dgms2[2][i][0] == 0] +
                 [dgms2[3][i][1][0] for i in range(len(dgms2[3])) if dgms2[3][i][0] == 0],
                 [dgms2[2][i][1][1] for i in range(len(dgms2[2])) if dgms2[2][i][0] == 0] +
                 [dgms2[3][i][1][1] for i in range(len(dgms2[3])) if dgms2[3][i][0] == 0])
axs[0,1].plot([-1,1],[-1,1])
axs[0,1].set_title("Extended 0 PD(-f)")
axs[1,0].scatter([dgms[2][i][1][0] for i in range(len(dgms[2])) if dgms[2][i][0] == 2] +
                 [dgms[3][i][1][0] for i in range(len(dgms[3])) if dgms[3][i][0] == 2],
                 [dgms[2][i][1][1] for i in range(len(dgms[2])) if dgms[2][i][0] == 2] +
                 [dgms[3][i][1][1] for i in range(len(dgms[3])) if dgms[3][i][0] == 2])
axs[1,0].plot([-1,1],[-1,1])
axs[1,0].set_title("Extended 2 PD(f)")
axs[1,1].scatter([dgms2[2][i][1][0] for i in range(len(dgms2[2])) if dgms2[2][i][0] == 2] +
                 [dgms2[3][i][1][0] for i in range(len(dgms2[3])) if dgms2[3][i][0] == 2],
                 [dgms2[2][i][1][1] for i in range(len(dgms2[2])) if dgms2[2][i][0] == 2] +
                 [dgms2[3][i][1][1] for i in range(len(dgms2[3])) if dgms2[3][i][0] == 2])
axs[1,1].plot([-1,1],[-1,1])
axs[1,1].set_title("Extended 2 PD(-f)")
plt.show()'''


