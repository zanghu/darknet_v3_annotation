#include "shortcut_layer.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"

#include <stdio.h>
#include <assert.h>
/**
 * @brief shortcut层接收两个输入层的输入. 一个是前一层, 另一个是from层(网络配置文件[shotcut]部分的from项的值指出其索引号)
 *        通过阅读网络配置文件yolov3.cfg，感觉shortcut层的作用是类似于ResNet中的信号越级传递机制, 即每隔几个卷积层就加入一个shortcut层, 该层即接收前一层的输出，又接受前K层的输出.
 */


/**
 * @brief 创建一个shortcut层对象, 该层接收两个输入层的输入. 一个是前一层, 另一个是from层(网络配置文件[shotcut]部分的from项的值指出其索引号)
 * @param batch      一个batch的输入样本个数
 * @param index      from层对象在net->layers数组中的索引号. 这里已经经过换算, 所以不会是负数
 * @param w, h, c    前一层的输入feature map的w, h, c
 * @param w2, h2, c2 from层的输入feature map的w, h, c
 */
layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c);
    layer l = {0};
    l.type = SHORTCUT;
    l.batch = batch;

    // shortcut层输入尺寸等于from层的输出尺寸
    l.w = w2;
    l.h = h2;
    l.c = c2;

    // shortcut层的输出尺寸等于前一层的输出尺寸
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs; // 输入神经元个数与输出神经元个数

    l.index = index;

    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;
    #ifdef GPU
    l.forward_gpu = forward_shortcut_layer_gpu;
    l.backward_gpu = backward_shortcut_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    return l;
}

void resize_shortcut_layer(layer *l, int w, int h)
{
    assert(l->w == l->out_w);
    assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w*h*l->out_c;
    l->inputs = l->outputs;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}

void forward_shortcut_layer(const layer l, network net)
{
    // 将前一层输出从其输出缓冲区复制到当前shortcut层输出缓冲区(二者尺寸相等)
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);

    // 将from层输出与前一层输出按一定比例相加, 结果保存在l.output中
    // shortcut层前向传播时线性部分计算函数, 基本思想就是通过步长省略掉feature map尺寸较大的输入的feature map上的一些点, 使得两个输入的feature map的剩余点之间产生一一对应关系, 对应点相加
    shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output); // blas.c

    // 经过非线性激活函数
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_shortcut_layer(const layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    axpy_cpu(l.outputs*l.batch, l.alpha, l.delta, 1, net.delta, 1);
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_shortcut_layer_gpu(const layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    axpy_gpu(l.outputs*l.batch, l.alpha, l.delta_gpu, 1, net.delta_gpu, 1);
    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta_gpu);
}
#endif
