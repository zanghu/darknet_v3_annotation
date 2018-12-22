/**
 * 关于batch的说明
 * network对象有两个属性: net->batch和net->subdivisions
 * net->batch是网络配置文件中[net]部分batch属性的值
 * network每次加载的样本个数是net->batch个, 即每进行一次"前向+反向"传播过程使用的样本个数是net->batch个
 * network每经过net->subdivisions * net->batch个样本后更新一次参数
 *
 * 在注释时, 一般约定将net->batch称为一个小batch, 将net->subdivisions * net->batch称为一个大batch
 *
 * 当存在多个gpu时, 每个GPU上训练一个network对象, 每个network对象经过net->subdivisions * net->batch个样本后更新一次参数, 因此整个模型(由多个子network对象构成)每经历net->subdivisions * net->batch * ngpus个样本更新一次参数
 */
#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#ifdef GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SECRET_NUM -1234
extern int gpu_index;

typedef struct{
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;
tree *read_tree(char *filename);

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

// 代价函数的类型, 只在cost layer中用到
// 在darknet的具体实现中, 实际只有四类代价函数, SSE和SEG实际采用相同的实现
typedef enum{
    SSE, 
    MASKED, 
    L1, 
    SEG, 
    SMOOTH,WGAN
} COST_TYPE;

typedef struct{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;

    // 利用函数指针, 实现C语言的继承和多态
    void (*forward)   (struct layer, struct network);
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    void (*forward_gpu)   (struct layer, struct network);
    void (*backward_gpu)  (struct layer, struct network);
    void (*update_gpu)    (struct layer, update_args);

    int batch_normalize; // 当前层是否启用BN的标志
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs; // 当前层输入神经元总数(不考虑batch, 只考虑单个样本)
    int outputs; // 当前层输出神经元总数(不考虑batch, 只考虑单个样本)
    int nweights; // 字面意思是权值矩阵元素个数, 但因为只有convolutional_layer.c和deconvolutional_layer.c中初始化并用到该属性, 所以实际上是四维卷积核张量的元素个数(groups不等于时要考虑groups), conected_layer.c用l.outputs*l.inputs代替
    int nbiases; // 偏置向量元素个数, 因为只有convolutional_layer.c和deconvolutional_layer.c中初始化并用到该属性, 所以实际上等于层的输出channel数, conected_layer.c用l.outputs代替
    int truths;
    int h,w,c; // 输入feature map的高度、宽度和channels数
    int out_h, out_w, out_c;

    // 出现在yolo_layer.c, region_layer.c, detection_layer.c, l2norm_layer.c, deconvolutional_layer.c, batchnorm_layer.c, convolutional_layer.c共7个不同类型的层
    int n; // 卷积层中等于输出channel数, yolo层中表示的是mask数组的元素个数, 感觉这个属性是"自由人", 不同的层中有不同的含义, 没有统一的意义
    int max_boxes;
    int groups; // 只在softmax_layer.c和convolutional_layer.c中用到, 具体作用不明, 具有减少权值的作用, 
    int size; // 卷积核的感受野边长, darknet目前只支持正方形感受野
    int side;
    int stride; // 卷积层感受野扫描步长, darknet只支持正方形步长

    // 以下三个属性只在reorg_layer.c中用到, 有他们的组合共同控制reorg_layer正向/反向传播时的行为
    // 整体来说, flatten, extra两者控制着reorg层的前向传播的总体行为, 即前向传播采取(1)faltten(数据轴变换)、(2)extra(数据原样输出)还是(3)reorg(feature map合并)中的哪一种方式
    // 具体来说, faltten属性优先级最高, extra优先级次之, 最后才考虑reorg. 只要flatten为真则进行flatten, 否则如果extra为真则采用extra, 如果前两者都非真则采用reorg
    // 如果采用flatten或reorg方式前向传播, 具体输出会根据reverse属性是否为真有所不同.
    int reverse; // yolov3中upsample_layer.c也有用到
    int flatten;
    int extra; // yolov3中iseg_layer.c也有用到

    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter; // 抖动, 数据增强的一种方法, 适用于非均衡数据集
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers; // int数组, 保存着所有输入层在net->layers中的索引号. 该属性只有route_layer(用于concatenate多个层的输出)用到
    int   * input_sizes;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output; // 输出数据, 元素个数是batch*w*h*c, 轴向是(b, c, h, w). 这里没有input属性, 是因为正向传播时layer都是从net的input属性中获取输入, 输出又会拷贝到net的input属性中
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean; // yolov3完全没有用到的属性

// BN only start+++++++++++++++++++++++++
    //以下四组10个属性有做batchnorm的layer才会生成

    // 目前已知支持BN的darknet中的layer类型包括:
    // (1)connected_layer
    // (2)conbolutional_layer
    // (3)deconvolutional_layer
    // (4)crnn_layer
    // (5)rnn_layer
    // (6)gru_layer
    // (7)lstm_layer
    float * scales;
    float * scale_updates;

    float * mean;
    float * variance;
    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;
// BN only end-------------------------

    float * m;
    float * v;
    
    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu; 

    float * binary_input;

    // 以下三项基本只在, CRNN和RNN两种类型的layer中用到
    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;
	
    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;

#ifdef GPU
    int *indexes_gpu;

    float *z_gpu;
    float *r_gpu;
    float *h_gpu;

    float *temp_gpu;
    float *temp2_gpu;
    float *temp3_gpu;

    float *dh_gpu;
    float *hh_gpu;
    float *prev_cell_gpu;
    float *cell_gpu;
    float *f_gpu;
    float *i_gpu;
    float *g_gpu;
    float *o_gpu;
    float *c_gpu;
    float *dc_gpu; 

    float *m_gpu;
    float *v_gpu;
    float *bias_m_gpu;
    float *scale_m_gpu;
    float *bias_v_gpu;
    float *scale_v_gpu;

    float * combine_gpu;
    float * combine_delta_gpu;

    float * prev_state_gpu;
    float * forgot_state_gpu;
    float * forgot_delta_gpu;
    float * state_gpu;
    float * state_delta_gpu;
    float * gate_gpu;
    float * gate_delta_gpu;
    float * save_gpu;
    float * save_delta_gpu;
    float * concat_gpu;
    float * concat_delta_gpu;

    float * binary_input_gpu;
    float * binary_weights_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;
    float * weight_change_gpu;

    float * biases_gpu;
    float * bias_updates_gpu;
    float * bias_change_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;
    float * scale_change_gpu;

    float * output_gpu;
    float * loss_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;
#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

void free_layer(layer);

// network的policy属性的类型, 代表采用何种学习速率变化策略
typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{

    // layers是网络的每一层的句柄组成的数组
    // n是layers数组的元素个数, 也就是网络的层数
    // 注意: darknet中输入不算一个"层", 网络层数等于网络配置文件中除去[network]段之外的段数总和
    layer *layers;
    int n;

    // batch和subdivision联合控制每隔多少个样本更新一次单个网络的参数, 提供了一种平衡算法精度(比如BN, 为保证效果单个batch中样本数不能太小)和显存占用的方式
    // batch是指定每次输入network的样本个数, 这与网络的每一层的输入/输出矩阵的行数、内存和显存占用大小直接相关
    // subdivisions控制网络更新参数的时机, 即每隔 net->subdivisions个batch, 更新一次网络参数, 也就是每输入 net->subdivisions*net->batch 个样本更新一次网络参数
    // 在多GPU模式下, 每个GPU上运行一个模型, 每个GPU上的模型经过net->batch * net->subdivisions个样本后更新一次参数
    int batch; // 该属性值等于一个small batch的大小
    int subdivisions;

    size_t *seen; // 到目前为止网络训练经历的样本数

    int *t; // 似乎是用来记录到目前为止网络参数的累积更新次数. 函数update_network和update_network_gpu内部每执行一次参数更新, 该属性值自增1

    // 到目前为止网络训练经历的epoch数. 用来度量网络训练完成度的指标, 一般来说只有记录意义，没有实际算法价值
    // epoch定义: 假设训练样本集有N个样本, 那么模型每输入N个样本算作是经历了一个epoch
    // 注意: "经过1个epoch" 不等于 "每个样本都被训练了一次", 因为darknet加载数据时样本是有放回随机抽取的
    float epoch;


    // 注意: 下面四个指针均指向calloc分配的动态空间
    float *input; // 存储一个small batch的数据的空间, 指向net->layers中某一层的l->input. 在前向传递过程中input指向的内存会不断改变, 保证始终指向刚刚处理完毕的layer的output属性
    float *output; // 存储一个samll batch的数据的空间, 指向net->layers中输出层(被定义为net->layers中最后一个非COST类型的layer)的l->output
    float *truth; // 存储一个small batch的ground_truth数据的空间m. 是network专有的动态内存空间, 不是引用其他层属性的
    float *delta; // 存储一个small batch反向传播的灵敏度(sensitivity)
    float *workspace; // 只在少数几种layer(conv, deconv, local)中用到, 一般用来在正向传播和反向传播函数中作为缓冲区

    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps; // 该属性只在rnn中有用到, 默认值是1. 似乎是修正net->batch的一个因子系数
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;



    int train;
    int index;
    float *cost;
    float clip;

#ifdef GPU
    float *input_gpu;
    float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;
#endif

} network;

typedef struct {
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

/**
 * @brief darknet的图片容器, 取值是[0-1](左闭右闭)之间的浮点数
 * @note  对于使用load_image_color()产生的image对象, 其data属性中的轴向是(c, w, h), data的channel顺序是BGR
 *        w是图像的水平轴(numpy.ndarray的轴向1), h是图像的垂直轴(numpy.ndarray的轴向0). 按照cv2.imread读取到的numpy.ndarray对象的轴向, 是(c, 1, 0)     
 */
typedef struct {
    // 举例: 分辨率1024x768的桌面背景图片, w=1024, h=768, 用cv2.imread加载后形状为(1024, 768,3)
    int w; // 图像宽度, 即横边长度
    int h; // 图像高度, 即纵边长度
    int c;
    float *data; // 每个元素是像素值(unsigned char)除以255.得到的浮点数, 转化关系见image_opencv.cpp中的ipl_to_image()函数
} image;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox; // 一个标定框
    int classes; // 类标
    float *prob; // 置信度
    float *mask;
    float objectness;
    int sort_class;
} detection;

/**
 * @note 专门为data类型准备
 */
typedef struct matrix{
    int rows, cols;
    float **vals; // 二维数组, 先calloc分配一个float*数组, 之后为每个元素calloc分配float数组
} matrix;

/**
 * @brief 专门用来存储用于目标检测任务的样本图片及其标注信息的结构体
 * @note 专门为load_args类型准备
 */
typedef struct{

    // 样本图缩放到统一大小的目标图, (w, h)是目标图尺寸
    int w, h;

     // 为什么y也是一个矩阵: 对于detection任务, y用来存储样本图上的全部bndbox的信息, 每个bndbox的信息占矩阵的一行, 一张图片上的多个bndbox构成了y矩阵的多行
    matrix X; // 存储一个batch数量的样本的数据, 每一行是一张目标图
    matrix y; // 存储X中全部样本的标注信息. 每行是X对应行的样本的全部bbox信息. y中一行的元素, 每相邻的5个是一组, 表示对样本的一个bbox的描述: (x, y, w, h, id). 注意: y中每个元素的值都是[0, 1]之间的相对值, 不是绝对像素值

    // 在析构data对象时需要用到的标志位, 确定是浅层析构(析构matrix对象X和y的外壳)还是深层析构(析构X和y的vals属性)
    // 如果shallow=1, 则直接free(d.X.vals)和free(d.y.vals)
    // 如果shallow=0, 则调用free_matrix(d.X)和free_matrix(d.y)
    // 详见data.c: free_data()
    int shallow;

    int *num_boxes; // int数组, 每个元素代表X中一个样本bndbox数量
    box **boxes; // bndbox的标注信息, 这个属性似乎没有用处
} data;

/**
 * @brief loadargs中type属性的类型, 每一种类型在data.c中都有自己的数据加载函数
 */
typedef enum {
    CLASSIFICATION_DATA, 
    DETECTION_DATA,  // train_detector()函数中指定的数据类型
    CAPTCHA_DATA, 
    REGION_DATA, 
    IMAGE_DATA, 
    COMPARE_DATA, 
    WRITING_DATA, 
    SWAG_DATA, 
    TAG_DATA, 
    OLD_CLASSIFICATION_DATA, 
    STUDY_DATA, DET_DATA, 
    SUPER_DATA, 
    LETTERBOX_DATA,  // validate_detector.*()函数族中指定的数据类型
    REGRESSION_DATA, 
    SEGMENTATION_DATA, 
    INSTANCE_DATA, 
    ISEG_DATA
} data_type;

/**
 * @struct 加载训练/验证数据所依赖的全部信息的集合
 *         各个属性来源于数据配置文件(数据集信息: 样本、真值的路径，类标等等)和网络配置文件(数据输入规格信息: batch大小, 尺寸等等)
 *         初始化过程见detector.c的train_detector()函数, 使用方法见data.c的load_data()函数
 */
typedef struct load_args{

// 第一部分: 网络配置文件

    // 以下属性的初始化大部分直接复制自network对象中的同名属性值
    // 实际上它们都来自网络配置文件的[net]部分的内容, 对于属性名与配置文件项名称同名的情况, 不再详细说明来源
    int h; // 来自"height"项, 默认值0
    int w; // 来自"width"项, 默认值0
    float angle; // 默认值0
    float aspect; // 默认值1
    int center; // 默认值0
    float saturation; // 默认值1
    float exposure; // 默认值1
    float hue; // 默认值0

    int min; // 例外: net->min_crop, 网络配置文件[net]部分的"min_crop"项, 默认值net->w
    int max; // 例外: net->max_crop, 网络配置文件[net]部分的"max_crop"项, 默认值2*net->w
    int size; // 例外: args.size = net->w;

    // 来自于network对象最后一层的layer对象的同名属性
    int classes; // 类标种类数. 网络配置文件laye部分的"classes"项, yolo_layer和region_layer中是训练集类标默认值种类数, 例如coco是80, pascal_voc是20
    int num_boxes; // 最大输出候选框个数. 网络配置文件layer部分的"max"项, yolo_layer和detection_layer默认值是90, region_layer默认值是30
    int coords; // 用途不明. 网络配置文件layer部分的"coords"项, 只有在属性 type=INSTANCE_DATA 时才有用
    float jitter; // 生成抖动数据时的抖动系数. 网络配置文件layer部分的"jitter"项, 只有yolo_layer, region_layer和detection_layer会解析此项, 默认值0.2

// 第二部分: 数据配置文件

    // 以下3个属性都是根据数据配置文件中的信息初始化
    char **paths; // 训练集所有输入图片路径组成的字符串数组, 每个元素是一个训练图片的路径
    int m; // 训练集的样本总数, 也就是paths属性的元素个数. 来自数据配置文件"train"项, 该项指定一个.txt文件, 其中每一行是一个训练图片的路径

    // 完成模型一次参数更新所经过的样本数, 如果是多GPU模式, 就是所有GPU上的模型都更新一次所需的样本数. net->batch * net->subdivisions * ngpus
    // net->subdivisions来自[net]部分"subdivisions"项, 默认值1
    // net->batch的值 = [net]部分"batch"项的值(默认值1) / net->subdivisions
    // ngpus是用户执行shell命令: ./darknet detectot train ... 时'-gpu'参数输入的, 默认是1
    int n;

    char *path; // 对paths属性中某个元素的引用. 在validate_detector.*()方法中有用到, train_detector()训练过程中未初始化该属性

    // 网络自后向前第一个满足布尔表达式 l.out_w && l.out_h && l.out_c 的layer对象, 其输出l.out_w和l.out_h
    // 以下两个属性只有在保存网络权值时才有用, 见example/writing.c
    int out_w;
    int out_h;

    // 以下两个参数似乎也只有example目录下某些源文件中有用到
    int background;
    int scale;

    // 以下两个参数似乎没有用
    int nh;
    int nw;

// 第三部分: 代码中直接指定

    int threads;
    data_type type;

// 第四部分: 加载数据集得到的

    data *d;

// 第五部分: 未调研
    char **labels;

    image *im;
    image *resized;

    tree *hierarchy;
} load_args;

/**
 * @brief 图片中的一个bbox的信息
 */
typedef struct{
    int id;

    // bbox中心点坐标x, y和宽、高
    float x,y,w,h; // 训练集样本加载时是直接从真值文件中读取

    // 矩形区域的另一种表示方法: 横坐标最小值, 横坐标最大值, 纵坐标最小值, 纵坐标最大值
    float left;   // left   = x - w/2
    float right;  // right  = x + w/2
    float top;    // top    = y - h/2
    float bottom; // bottom = y + h/2
} box_label;


network *load_network(char *cfg, char *weights, int clear);
load_args get_base_args(network *net);

void free_data(data d);

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

pthread_t load_data(load_args args);
list *read_data_cfg(char *filename);
list *read_cfg(char *filename);
unsigned char *read_file(char *filename);
data resize_data(data orig, int w, int h);
data *tile_data(data orig, int divs, int size);
data select_data(data *orig, int *inds);

void forward_network(network *net);
void backward_network(network *net);
void update_network(network *net);


float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void softmax(float *input, int n, float temp, int stride, float *output);

int best_3d_shift_r(image a, image b, int min, int max);
#ifdef GPU
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void fill_gpu(int N, float ALPHA, float * X, int INCX);
void scal_gpu(int N, float ALPHA, float * X, int INCX);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);

void cuda_set_device(int n);
void cuda_free(float *x_gpu);
float *cuda_make_array(float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
float cuda_mag_array(float *x_gpu, size_t n);
void cuda_push_array(float *x_gpu, float *x, size_t n);

void forward_network_gpu(network *net);
void backward_network_gpu(network *net);
void update_network_gpu(network *net);

float train_networks(network **nets, int n, data d, int interval);
void sync_nets(network **nets, int n, int interval);
void harmless_update_network_gpu(network *net);
#endif
image get_label(image **characters, char *string, int size);
void draw_label(image a, int r, int c, image label, const float *rgb);
void save_image(image im, const char *name);
void save_image_options(image im, const char *name, IMTYPE f, int quality);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void grayscale_image_3c(image im);
void normalize_image(image p);
void matrix_to_csv(matrix m);
float train_network_sgd(network *net, data d, int n);
void rgbgr_image(image im);
data copy_data(data d);
data concat_data(data d1, data d2);
data load_cifar10_data(char *filename);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
matrix csv_to_matrix(char *filename);
float *network_accuracies(network *net, data d, int n);
float train_network_datum(network *net);
image make_random_image(int w, int h, int c);

void denormalize_connected_layer(layer l);
void denormalize_convolutional_layer(layer l);
void statistics_connected_layer(layer l);
void rescale_weights(layer l, float scale, float trans);
void rgbgr_weights(layer l);
image *get_weights(layer l);

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, float hier_thresh, int w, int h, int fps, int fullscreen);
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);

network *parse_network_cfg(char *filename);
void save_weights(network *net, char *filename);
void load_weights(network *net, char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);

void zero_objectness(layer l);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
void free_network(network *net);
void set_batch_network(network *net, int b);
void set_temp_network(network *net, float t);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
void censor_image(image im, int dx, int dy, int w, int h);
image letterbox_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image center_crop_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
image threshold_image(image im, float thresh);
image mask_to_rgb(image mask);
int resize_network(network *net, int w, int h);
void free_matrix(matrix m);
void test_resize(char *filename);
int show_image(image p, const char *name, int ms);
image copy_image(image p);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
float get_current_rate(network *net);
void composite_3d(char *f1, char *f2, char *out, int delta);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
size_t get_current_batch(network *net);
void constrain_image(image im);
image get_network_image_layer(network *net, int i);
layer get_network_output_layer(network *net);
void top_predictions(network *net, int n, int *index);
void flip_image(image a);
image float_to_image(int w, int h, int c, float *data);
void ghost_image(image source, image dest, int dx, int dy);
float network_accuracy(network *net, data d);
void random_distort_image(image im, float hue, float saturation, float exposure);
void fill_image(image m, float s);
image grayscale_image(image im);
void rotate_image_cw(image im, int times);
double what_time_is_it_now();
image rotate_image(image m, float rad);
void visualize_network(network *net);
float box_iou(box a, box b);
data load_all_cifar10();
box_label *read_boxes(char *filename, int *n);
box float_to_box(float *f, int stride);
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);

matrix network_predict_data(network *net, data test);
image **load_alphabet();
image get_network_image(network *net);
float *network_predict(network *net, float *input);

int network_width(network *net);
int network_height(network *net);
float *network_predict_image(network *net, image im);
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets);
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
void free_detections(detection *dets, int n);

void reset_network_state(network *net, int b);

char **get_labels(char *filename);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void do_nms_sort(detection *dets, int total, int classes, float thresh);

matrix make_matrix(int rows, int cols);

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
void make_window(char *name, int w, int h, int fullscreen);
#endif

void free_image(image m);
float train_network(network *net, data d);
pthread_t load_data_in_thread(load_args args);
void load_data_blocking(load_args args);
list *get_paths(char *filename);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);

int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
float sec(clock_t clocks);
void **list_to_array(list *l);
void top_k(float *a, int n, int k, int *index);
int *read_map(char *filename);
void error(const char *s);
int max_index(float *a, int n);
int max_int_index(int *a, int n);
int sample_array(float *a, int n);
int *random_index_order(int min, int max);
void free_list(list *l);
float mse_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
void scale_array(float *a, int n, float s);
float mean_array(float *a, int n);
float sum_array(float *a, int n);
void normalize_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);
size_t rand_size_t();
float rand_normal();
float rand_uniform(float min, float max);

#ifdef __cplusplus
}
#endif
#endif
