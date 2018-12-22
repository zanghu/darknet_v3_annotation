#include "data.h"
#include "utils.h"
#include "image.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 此处的互斥锁用于保证rand()的线程安全, 不知道为什么不使用线程安全版的rand_r()替换rand()
// 此外根据资料显示, rand()的随机性也并不太好, random()函数就具有比rand()更好的随机性
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * @brief 将filename所指文件的每一行作为一个元素, 保存在一个单链表中
 *
 * @note 该函数设计存在以下问题:
 *       (1)darknet中保存样本图像路径的.list文件行数(等于样本数)可能会非常大(10^4至10^6), 每一行依次链表的每次动态分配内存可能造成很大的效率和内存碎片问题
 *       (2)链表不利于随机访问
 *       这里似乎更适合使用基于内存池的顺序表结构来存储
 */
list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

/*
char **get_random_paths_indexes(char **paths, int n, int m, int *indexes)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        int index = rand()%m;
        indexes[i] = index;
        random_paths[i] = paths[index];
        if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}
*/

/**
 * @brief 从样本全集(共m个样本)中随机选取n个样本(的索引号), 返回全部选中样本的路径.
 *        拷贝时只拷贝了引用, 并未拷贝字符串
 * @param n 需要从paths数组中随机选取的元素个数
 * @param m paths数组的元素总数
 */
char **get_random_paths(char **paths, int n, int m)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex); // 通过加锁实现rand()函数线程安全
    for(i = 0; i < n; ++i){
        int index = rand()%m;
        random_paths[i] = paths[index];
        //if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}

char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
    char **replace_paths = calloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        char replaced[4096];
        find_replace(paths[i], find, replace, replaced);
        replace_paths[i] = copy_string(replaced);
    }
    return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image(paths[i], w, h, 3);

        image gray = grayscale_image(im);
        free_image(im);
        im = gray;

        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_paths(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], w, h);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop;
        if(center){
            crop = center_crop_image(im, size, size);
        } else {
            crop = random_augment_image(im, angle, aspect, min, max, size, size);
        }
        int flip = rand()%2;
        if (flip) flip_image(crop);
        random_distort_image(crop, hue, saturation, exposure);

        /*
        show_image(im, "orig");
        show_image(crop, "crop");
        cvWaitKey(0);
        */
        //grayscale_image_3c(crop);
        free_image(im);
        X.vals[i] = crop.data;
        X.cols = crop.h*crop.w*crop.c;
    }
    return X;
}

/**
 * @brief 读取的标注信息文件并不是xml
 *        要求文件中每一行包含对样本的一个bndbox的标注, 行的内容是用空格分隔的五个数字, 分别表示: id, 中心坐标x, 中心坐标y, 图片高度, 图片宽度
 *        文件包含的行数量不受限制
 *
 * @param filename 一个样本图片的标注信息文件的路径
 * @param n 返回的box_label数组的元素个数, 也就是实际读取到的bndbox个数
 * @return box_label对象组成的数组(动态分配)
 */
box_label *read_boxes(char *filename, int *n)
{
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    float x, y, h, w;
    int id;
    int count = 0;
    int size = 64;
    box_label *boxes = calloc(size, sizeof(box_label));
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){

        // 数组空间自动增长, 指数级增长
        if(count == size) {
            size = size * 2;
            boxes = realloc(boxes, size*sizeof(box_label));
        }
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

/**
 * @brief 对一个样本图片的所有标注信息组成的数组做一个随机乱序
 */
void randomize_boxes(box_label *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        box_label swap = b[i];
        int index = rand()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

/**
 * @brief 根据样本图片加载时所做的样本增强变化(抖动、水平翻转等), 对其真值信息进行必要的修正
 *
 * @param boxes 已加载到的一张图片的全部bbox信息的集合
 * @param n     boxes数组的元素个数
 * @param dx    样本图水平平移
 * @param dy    样本图垂直平移
 * @param sx    样本图水平缩放
 * @param sy    样本图垂直缩放
 * @param flip  int作为逻辑表达式使用, 表示样本是否进行了水平翻转
 */
void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
    int i;
    for(i = 0; i < n; ++i){
        if(boxes[i].x == 0 && boxes[i].y == 0) {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }

        // 先修正bbox的第二种表示
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        if(flip){
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        // 将bbox第二种表示中所有取值不在区间(0, 1)的值限制为边界值0或1
        boxes[i].left =  constrain(0, 1, boxes[i].left); // float constrain(float min, float max, float a), utils.c
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        // 重新计算bbox的第一种表示
        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count && i < 90; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .0 || h < .0) continue;

        int index = (4+classes) * i;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;

        if (id < classes) truth[index+id] = 1;
    }
    free(boxes);
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);

    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".png", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .005 || h < .005) continue;

        int col = (int)(x*num_boxes);
        int row = (int)(y*num_boxes);

        x = x*num_boxes - col;
        y = y*num_boxes - row;

        int index = (col+row*num_boxes)*(5+classes);
        if (truth[index]) continue;
        truth[index++] = 1;

        if (id < classes) truth[index+id] = 1;
        index += classes;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;
    }
    free(boxes);
}

void load_rle(image im, int *rle, int n)
{
    int count = 0;
    int curr = 0;
    int i,j;
    for(i = 0; i < n; ++i){
        for(j = 0; j < rle[i]; ++j){
            im.data[count++] = curr;
        }
        curr = 1 - curr;
    }
    for(; count < im.h*im.w*im.c; ++count){
        im.data[count] = curr;
    }
}

void or_image(image src, image dest, int c)
{
    int i;
    for(i = 0; i < src.w*src.h; ++i){
        if(src.data[i]) dest.data[dest.w*dest.h*c + i] = 1;
    }
}

void exclusive_image(image src)
{
    int k, j, i;
    int s = src.w*src.h;
    for(k = 0; k < src.c-1; ++k){
        for(i = 0; i < s; ++i){
            if (src.data[k*s + i]){
                for(j = k+1; j < src.c; ++j){
                    src.data[j*s + i] = 0;
                }
            }
        }
    }
}

box bound_image(image im)
{
    int x,y;
    int minx = im.w;
    int miny = im.h;
    int maxx = 0;
    int maxy = 0;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            if(im.data[y*im.w + x]){
                minx = (x < minx) ? x : minx;
                miny = (y < miny) ? y : miny;
                maxx = (x > maxx) ? x : maxx;
                maxy = (y > maxy) ? y : maxy;
            }
        }
    }
    box b = {minx, miny, maxx-minx + 1, maxy-miny + 1};
    //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
    return b;
}

void fill_truth_iseg(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    int i = 0;
    int j;
    image part = make_image(w, h, 1);
    while((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
        if(flip) flip_image(sized);

        image mask = resize_image(sized, mw, mh);
        truth[i*(mw*mh+1)] = id;
        for(j = 0; j < mw*mh; ++j){
            truth[i*(mw*mh + 1) + 1 + j] = mask.data[j];
        }
        ++i;

        free_image(mask);
        free_image(sized);
        free(rle);
    }
    if(i < num_boxes) truth[i*(mw*mh+1)] = -1;
    fclose(file);
    free_image(part);
}

void fill_truth_mask(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    int i = 0;
    image part = make_image(w, h, 1);
    while((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
        if(flip) flip_image(sized);
        box b = bound_image(sized);
        if(b.w > 0){
            image crop = crop_image(sized, b.x, b.y, b.w, b.h);
            image mask = resize_image(crop, mw, mh);
            truth[i*(4 + mw*mh + 1) + 0] = (b.x + b.w/2.)/sized.w;
            truth[i*(4 + mw*mh + 1) + 1] = (b.y + b.h/2.)/sized.h;
            truth[i*(4 + mw*mh + 1) + 2] = b.w/sized.w;
            truth[i*(4 + mw*mh + 1) + 3] = b.h/sized.h;
            int j;
            for(j = 0; j < mw*mh; ++j){
                truth[i*(4 + mw*mh + 1) + 4 + j] = mask.data[j];
            }
            truth[i*(4 + mw*mh + 1) + 4 + mw*mh] = id;
            free_image(crop);
            free_image(mask);
            ++i;
        }
        free_image(sized);
        free(rle);
    }
    fclose(file);
    free_image(part);
}

/**
 * @brief 读取一张样本图片对应的真值信息
 *        注意: 如果样本图片在读取时进行了样本增强(抖动、翻转等等), 那么真值信息也必须进行相应的变换
 * 
 * @param path      准备读取真值信息的样本图片的路径
 *                  darknet要求每个样本图片和它所对应的真值信息描述文件之间的路径必须符合一定的对应关系, 因此可以根据样本图片路径得到真值信息文件的路径
 * @param num_boxes 对单个样本中bndbox数量的限制, 超过该数量的bndbox将被忽略(yolo_v2默认30, yolo_v3默认90)
 * @param truth     指向预分配好的(num_boxes*sizeof(float))用来保存读取到的真值信息的内存空间的指针. 每5个元素是对一个bbox的描述, 格式是(x, y, w, h, id). 在ctw-baseline中, id就是类标的数字表示
                    注意: 由于会过滤掉边长过小的bbox, 所以实际上truth中被填充的有效元素数可能小于90(yolo_v3)个. 
 * @param classes   数据集的类别数
 * @param flip      样本图片是否进行了水平翻转
 * @param dx, dy    真值框的水平偏移量和垂直偏移量(因为样本图缩放后嵌入画布时进行了随机平移). 注意这里的平移量不是像素数绝对值, 而是分别相对于w和h的相对值(-dx/w, -dy/h)
 * @param sx, sy    真值框的宽度和长度的缩放系数(因为样本图进行了随机缩放)
 */
void fill_truth_detection(char *path, int num_boxes, float *truth, int classes, int flip, float dx, float dy, float sx, float sy) // (random_paths[i], boxes, d.y.vals[i], classes, flip, -dx/w, -dy/h, nw/w, nh/h)
{
    char labelpath[4096];

    // void find_replace(char *str, char *orig, char *rep, char *output): 将str中的第一个orig字符串替换为rep字符串, 之后将替换结果写入output
    // 如果str中不包含orig, 那么返回的output中就是str的原样拷贝
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);
    find_replace(labelpath, "raw", "labels", labelpath);

    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".png", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);

/* 真值文件内容示例: 
 * 来自于ctw-baseline, 文件路径: detection/products/trainval/0000172_6_0.txt, 对应于样本图片detection/products/trainval/0000172_6_0.jpg
 * 每行内容: "id, 中心坐标x, 中心坐标y, 图片宽度, 图片高度", 注意: 这里的坐标和长度信息都不是绝对像素值, 而是相对于图片边长的相对值(bbox的x和w是相对于样本图片宽度, y和h是相对于样本图片的高度, 具体可参考ctw-baseline的detection/prepare_train_data.py)
 * 在ctw-baseline中, id就是类标的数字表示, ctw-baseline训练真值信息文件具体生成方法参考: detection/prepare_train_data.py
 * 以下为文件内容:
47 0.900858 0.287963 0.131950 0.228359
266 0.064299 0.246023 0.133742 0.228904
811 0.252052 0.246023 0.190325 0.228904
145 0.214759 0.962409 0.069443 0.081863
45 0.275200 0.962620 0.051439 0.080559
173 0.325353 0.962789 0.048867 0.079520
172 0.375506 0.962954 0.051439 0.078498
95 0.426945 0.963125 0.051439 0.077441
116 0.480957 0.963305 0.051439 0.076331
147 0.532396 0.963477 0.051439 0.075274

*/
    int count = 0;

    // 返回一个box_label对象组成的数组(动态分配)及其长度, 数组的每个元素是一个bndbox的真值信息, 对应于label文件的一行(用空格分隔开的5个数)
    // 返回的boxes所有坐标和长度信息都不是像素绝对值, 而是某种相对值
    box_label *boxes = read_boxes(labelpath, &count); // 返回参数count保存返回bodx_label数组的元素个数

    // 对标注信息数组进行随机乱序, 这样做的必要性是?
    randomize_boxes(boxes, count);

    // 根据图片加载时所做的样本增强操作, 对真值信息进行必要的修正
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);

    // 如果图片的bndbox数目超过count个, 则自动忽略索引大于等于count的那些bndbox
    if(count > num_boxes) count = num_boxes;
    float x,y,w,h;
    int id;
    int i;

    // 边长过小的bbox会被忽略(只要长和宽有一个过小)
    int sub = 0; // 被忽略的bbox个数

    // 把box_label对象的数据存储到矩阵中, 注意每个bndbox的5元组描述顺序由标注信息文件中的(id, x, y, w, h)变为(x, y, w, h, id)
    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if ((w < .001 || h < .001)) { // 忽略边长过小的bbox, 不参与训练
            ++sub;
            continue;
        }

        truth[(i-sub)*5+0] = x;
        truth[(i-sub)*5+1] = y;
        truth[(i-sub)*5+2] = w;
        truth[(i-sub)*5+3] = h;
        truth[(i-sub)*5+4] = id;
    }
    free(boxes);
}

#define NUMCHARS 37

void print_letters(float *pred, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        int index = max_index(pred+i*NUMCHARS, NUMCHARS);
        printf("%c", int_to_alphanum(index));
    }
    printf("\n");
}

void fill_truth_captcha(char *path, int n, float *truth)
{
    char *begin = strrchr(path, '/');
    ++begin;
    int i;
    for(i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i){
        int index = alphanum_to_int(begin[i]);
        if(index > 35) printf("Bad %c\n", begin[i]);
        truth[i*NUMCHARS+index] = 1;
    }
    for(;i < n; ++i){
        truth[i*NUMCHARS + NUMCHARS-1] = 1;
    }
}

data load_data_captcha(char **paths, int n, int m, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = make_matrix(n, k*NUMCHARS);
    int i;
    for(i = 0; i < n; ++i){
        fill_truth_captcha(paths[i], k, d.y.vals[i]);
    }
    if(m) free(paths);
    return d;
}

data load_data_captcha_encode(char **paths, int n, int m, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.X.cols = 17100;
    d.y = d.X;
    if(m) free(paths);
    return d;
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    int count = 0;
    for(i = 0; i < k; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
            ++count;
            //printf("%s %s %d\n", path, labels[i], i);
        }
    }
    if(count != 1 && (k != 1 || count != 0)) printf("Too many or too few labels: %d, %s\n", count, path);
}

void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
    int j;
    for(j = 0; j < k; ++j){
        if(truth[j]){
            int parent = hierarchy->parent[j];
            while(parent >= 0){
                truth[parent] = 1;
                parent = hierarchy->parent[parent];
            }
        }
    }
    int i;
    int count = 0;
    for(j = 0; j < hierarchy->groups; ++j){
        //printf("%d\n", count);
        int mask = 1;
        for(i = 0; i < hierarchy->group_size[j]; ++i){
            if(truth[count + i]){
                mask = 0;
                break;
            }
        }
        if (mask) {
            for(i = 0; i < hierarchy->group_size[j]; ++i){
                truth[count + i] = SECRET_NUM;
            }
        }
        count += hierarchy->group_size[j];
    }
}

matrix load_regression_labels_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i,j;
    for(i = 0; i < n; ++i){
        char labelpath[4096];
        find_replace(paths[i], "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".BMP", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);
        find_replace(labelpath, ".JPG", ".txt", labelpath);
        find_replace(labelpath, ".JPeG", ".txt", labelpath);
        find_replace(labelpath, ".Jpeg", ".txt", labelpath);
        find_replace(labelpath, ".PNG", ".txt", labelpath);
        find_replace(labelpath, ".TIF", ".txt", labelpath);
        find_replace(labelpath, ".bmp", ".txt", labelpath);
        find_replace(labelpath, ".jpeg", ".txt", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".png", ".txt", labelpath);
        find_replace(labelpath, ".tif", ".txt", labelpath);

        FILE *file = fopen(labelpath, "r");
        for(j = 0; j < k; ++j){
            fscanf(file, "%f", &(y.vals[i][j]));
        }
        fclose(file);
    }
    return y;
}

matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy)
{
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n && labels; ++i){
        fill_truth(paths[i], labels, k, y.vals[i]);
        if(hierarchy){
            fill_hierarchy(y.vals[i], k, hierarchy);
        }
    }
    return y;
}

matrix load_tags_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    //int count = 0;
    for(i = 0; i < n; ++i){
        char label[4096];
        find_replace(paths[i], "images", "labels", label);
        find_replace(label, ".jpg", ".txt", label);
        FILE *file = fopen(label, "r");
        if (!file) continue;
        //++count;
        int tag;
        while(fscanf(file, "%d", &tag) == 1){
            if(tag < k){
                y.vals[i][tag] = 1;
            }
        }
        fclose(file);
    }
    //printf("%d/%d\n", count, n);
    return y;
}

char **get_labels(char *filename)
{
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

/**
 * @brief 根据d.shallow是否为0确定是浅层析构(shallow=1)还是深层析构(shallow=0). 浅层析构还是深层析构都不需要析构d.X和d.y, 因为它们是data结构的"实在"属性, 不是指针属性
 *        (1)浅层析构: 只析构matrix.vals表, 不析构表中每个元素所指的空间
 *        (2)深层析构: 先析构matrix.vals表的每个元素所指的空间, 再释放matrix.vals表本身
 *        综上, (2)包含了(1)的操作
 */
void free_data(data d)
{
    // 注意: 由于matrix对象是直接以值属性(而不是指针属性)的形式存在于data结构中, 因此一般不需要释放matrix对象的外壳(可以随data对象的释放被释放)
    // 下面的浅层析构(shallow!=0)和深层析构(shallow=1)的区别是: 浅层析构只析构了matrix对象X和y的指针表vals, 但是没有依次释放每一个 vals[i]所指的数据
    if(!d.shallow){ // 深层析构
        free_matrix(d.X); // 释放matrix的每一条引用数据, 不释放索引表d.X.vals
        free_matrix(d.y); // 释放matrix的每一条引用数据, 不释放索引表d.X.vals
    }
    else{ // 浅层析构
        free(d.X.vals); // 释放matrix对象的索引表d.X.vals, 但不释放索引表中的引用数据
        free(d.y.vals); // 释放matrix对象的索引表d.X.vals, 但不释放索引表中的引用数据
    }
}

image get_segmentation_image(char *path, int w, int h, int classes)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    image mask = make_image(w, h, classes);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    image part = make_image(w, h, 1);
    while(fscanf(file, "%d %s", &id, buff) == 2){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        or_image(part, mask, id);
        free(rle);
    }
    //exclusive_image(mask);
    fclose(file);
    free_image(part);
    return mask;
}

image get_segmentation_image2(char *path, int w, int h, int classes)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    image mask = make_image(w, h, classes+1);
    int i;
    for(i = 0; i < w*h; ++i){
        mask.data[w*h*classes + i] = 1;
    }
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    image part = make_image(w, h, 1);
    while(fscanf(file, "%d %s", &id, buff) == 2){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        or_image(part, mask, id);
        for(i = 0; i < w*h; ++i){
            if(part.data[i]) mask.data[w*h*classes + i] = 0;
        }
        free(rle);
    }
    //exclusive_image(mask);
    fclose(file);
    free_image(part);
    return mask;
}

data load_data_seg(int n, char **paths, int m, int w, int h, int classes, int min, int max, float angle, float aspect, float hue, float saturation, float exposure, int div)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;


    d.y.rows = n;
    d.y.cols = h*w*classes/div/div;
    d.y.vals = calloc(d.X.rows, sizeof(float*));

    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;

        image mask = get_segmentation_image(random_paths[i], orig.w, orig.h, classes);
        //image mask = make_image(orig.w, orig.h, classes+1);
        image sized_m = rotate_crop_image(mask, a.rad, a.scale/div, a.w/div, a.h/div, a.dx/div, a.dy/div, a.aspect);

        if(flip) flip_image(sized_m);
        d.y.vals[i] = sized_m.data;

        free_image(orig);
        free_image(mask);

        /*
           image rgb = mask_to_rgb(sized_m, classes);
           show_image(rgb, "part");
           show_image(sized, "orig");
           cvWaitKey(0);
           free_image(rgb);
         */
    }
    free(random_paths);
    return d;
}

data load_data_iseg(int n, char **paths, int m, int w, int h, int classes, int boxes, int div, int min, int max, float angle, float aspect, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, (((w/div)*(h/div))+1)*boxes);

    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;
        //show_image(sized, "image");

        fill_truth_iseg(random_paths[i], boxes, d.y.vals[i], classes, orig.w, orig.h, a, flip, w/div, h/div);

        free_image(orig);

        /*
           image rgb = mask_to_rgb(sized_m, classes);
           show_image(rgb, "part");
           show_image(sized, "orig");
           cvWaitKey(0);
           free_image(rgb);
         */
    }
    free(random_paths);
    return d;
}

data load_data_mask(int n, char **paths, int m, int w, int h, int classes, int boxes, int coords, int min, int max, float angle, float aspect, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, (coords+1)*boxes);

    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;
        //show_image(sized, "image");

        fill_truth_mask(random_paths[i], boxes, d.y.vals[i], classes, orig.w, orig.h, a, flip, 14, 14);

        free_image(orig);

        /*
           image rgb = mask_to_rgb(sized_m, classes);
           show_image(rgb, "part");
           show_image(sized, "orig");
           cvWaitKey(0);
           free_image(rgb);
         */
    }
    free(random_paths);
    return d;
}

data load_data_region(int n, char **paths, int m, int w, int h, int size, int classes, float jitter, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;


    int k = size*size*(5+classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);

        int oh = orig.h;
        int ow = orig.w;

        int dw = (ow*jitter);
        int dh = (oh*jitter);

        int pleft  = rand_uniform(-dw, dw);
        int pright = rand_uniform(-dw, dw);
        int ptop   = rand_uniform(-dh, dh);
        int pbot   = rand_uniform(-dh, dh);

        int swidth =  ow - pleft - pright;
        int sheight = oh - ptop - pbot;

        float sx = (float)swidth  / ow;
        float sy = (float)sheight / oh;

        int flip = rand()%2;
        image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

        float dx = ((float)pleft/ow)/sx;
        float dy = ((float)ptop /oh)/sy;

        image sized = resize_image(cropped, w, h);
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;

        fill_truth_region(random_paths[i], d.y.vals[i], classes, size, flip, dx, dy, 1./sx, 1./sy);

        free_image(orig);
        free_image(cropped);
    }
    free(random_paths);
    return d;
}

data load_data_compare(int n, char **paths, int m, int classes, int w, int h)
{
    if(m) paths = get_random_paths(paths, 2*n, m);
    int i,j;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*6;

    int k = 2*(classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image im1 = load_image_color(paths[i*2],   w, h);
        image im2 = load_image_color(paths[i*2+1], w, h);

        d.X.vals[i] = calloc(d.X.cols, sizeof(float));
        memcpy(d.X.vals[i],         im1.data, h*w*3*sizeof(float));
        memcpy(d.X.vals[i] + h*w*3, im2.data, h*w*3*sizeof(float));

        int id;
        float iou;

        char imlabel1[4096];
        char imlabel2[4096];
        find_replace(paths[i*2],   "imgs", "labels", imlabel1);
        find_replace(imlabel1, "jpg", "txt", imlabel1);
        FILE *fp1 = fopen(imlabel1, "r");

        while(fscanf(fp1, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id] < iou) d.y.vals[i][2*id] = iou;
        }

        find_replace(paths[i*2+1], "imgs", "labels", imlabel2);
        find_replace(imlabel2, "jpg", "txt", imlabel2);
        FILE *fp2 = fopen(imlabel2, "r");

        while(fscanf(fp2, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id + 1] < iou) d.y.vals[i][2*id + 1] = iou;
        }

        for (j = 0; j < classes; ++j){
            if (d.y.vals[i][2*j] > .5 &&  d.y.vals[i][2*j+1] < .5){
                d.y.vals[i][2*j] = 1;
                d.y.vals[i][2*j+1] = 0;
            } else if (d.y.vals[i][2*j] < .5 &&  d.y.vals[i][2*j+1] > .5){
                d.y.vals[i][2*j] = 0;
                d.y.vals[i][2*j+1] = 1;
            } else {
                d.y.vals[i][2*j]   = SECRET_NUM;
                d.y.vals[i][2*j+1] = SECRET_NUM;
            }
        }
        fclose(fp1);
        fclose(fp2);

        free_image(im1);
        free_image(im2);
    }
    if(m) free(paths);
    return d;
}

data load_data_swag(char **paths, int n, int classes, float jitter)
{
    int index = rand()%n;
    char *random_path = paths[index];

    image orig = load_image_color(random_path, 0, 0);
    int h = orig.h;
    int w = orig.w;

    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;

    d.X.rows = 1;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    int k = (4+classes)*90;
    d.y = make_matrix(1, k);

    int dw = w*jitter;
    int dh = h*jitter;

    int pleft  = rand_uniform(-dw, dw);
    int pright = rand_uniform(-dw, dw);
    int ptop   = rand_uniform(-dh, dh);
    int pbot   = rand_uniform(-dh, dh);

    int swidth =  w - pleft - pright;
    int sheight = h - ptop - pbot;

    float sx = (float)swidth  / w;
    float sy = (float)sheight / h;

    int flip = rand()%2;
    image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

    float dx = ((float)pleft/w)/sx;
    float dy = ((float)ptop /h)/sy;

    image sized = resize_image(cropped, w, h);
    if(flip) flip_image(sized);
    d.X.vals[0] = sized.data;

    fill_truth_swag(random_path, d.y.vals[0], classes, flip, dx, dy, 1./sx, 1./sy);

    free_image(orig);
    free_image(cropped);

    return d;
}

/**
 * @brief 单个数据读取线程读取数据类型是DETECTION_DATA时所执行的函数.
 *        函数读取训练样本的图像数据(特征)和标注信息(真值)
 *        函数在读取图像时做的样本边还包括: (1)尺度随机放缩; (2)jitter, 即所谓抖动(可选); (3) 随机平移(与放缩和抖动相关); (4)HSV空间随机歪曲; (5)随机水平翻转; 
 *                                          上述变换中除了(2)是可选的其他都是随机的; 注意这里没有做旋转变换, 可能是认为(2)或者(4)可以代替旋转??
 *
 * @param n          函数需要加载的样本总数, 一般等于样本集总数(就是1个epoch的样本数)/数据加载线程数.
 * @param paths      样本全集的路径, 每个元素是一个图片的路径, 函数从中随机选n个加载之. 
 *                   注意: 当前函数假设样本对应的标注信息文件的路径与图片路径只有特定的几个目录名和后缀替换, 因此标注文件路径可以由图像路径求出, 函数加载标注信息时事实上也就是这么做的.
 * @param  m         样本全集的样本总数.
 * @param w, h       (net->w, net->h), 来源于网络配置文件里[net]段的"width"项和"height"项的值, 网络输入数据的尺寸. 加载图片需要按照某种方式缩放(或嵌入)到该尺寸.
                     由于函数内会对原始输入图片进行放缩, 使之恰当的嵌入到符合网络输入要求尺寸的画布中, 所以从paths中读取到的原始图的尺寸实际上可以是任意的
 * @param boxes      每个样本图最大允许的bndbox数, 超过该数量的bndbox标注信息将被忽略. 
 *                   该值来源于网络配置文件中最后一层的"max"项, yolo_v2的detection_layer默认是30, yolo_v3的yolo_layer默认是90
 * @param classes    样本集的类别总数, 该数值来源于网络配置文件中最后一层的段的"classes"项, 默认值是1.
 * @param jitter     这个参数为缩放抖动系数，就是图片缩放抖动的剧烈程度. 所谓缩放抖动，就是在宽高上插值缩放图片，宽高两方向上缩放的系数不一定相同. 
 *                   来源于网络配置文件的最后一段中的jitter项, 默认值0.2
 * @param hue        （hsv颜色空间）数据增强参数：色调（取值0度到360度）偏差最大值，实际色调偏差为-hue~hue之间的随机值.
 *                    来源于net的cfg文件的[net]段的"hue"项, 默认值0
 * @param saturation （hsv颜色空间）数据增强参数：饱和度（取值范围0~1）缩放最大值，实际为范围内的随机值.
 *                    来源于net的cfg文件的[net]段的"saturation"项, 默认值1
 * @param exposure   （hsv颜色空间）据增强参数：明度（色彩明亮程度，0~1）, 缩放最大值，实际为范围内的随机值.
 *                    来源于net的cfg文件的[net]段的"exposure"项, 默认值1
 *
 * @attention hue, saturation, exposure三个参数的唯一用途是共同用作图像在HSV色彩空间中实现图像随机歪曲
 * @attention 由于函数内会对原始输入图片进行放缩, 使之恰当的嵌入到符合网络输入要求尺寸的画布中, 所以从paths中读取到的原始图的尺寸实际上可以是任意的, 甚至可以每张样本图尺寸都不一样
 */
data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure)
{
    // 从样本全集(共m个样本)中随机选取n个样本(的索引号), 返回全部选中样本的路径
    char **random_paths = get_random_paths(paths, n, m); // random_paths指向动态分配的内存空间
    int i;

    // d是最终用来返回的"成果": 读取到的并处理好的数据
    data d = {0};
    d.shallow = 0;
    // d的X矩阵次采用手动创建方式, 只分配了行表, 没有为每一行分配空间, 这样一来后面可以直接把加载到的数据作为行表的元素挂在行表中
    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;
    // 使用make_matrix()函数为y矩阵分配空间, 一次性将矩阵行表和每一行的空间都分配完毕
    d.y = make_matrix(n, 5*boxes); // 每个样本对应于y矩阵的一行, 这一行由多个描述bbox的5元组依次拼接成(yolov2的detection_layer默认是30, yolo_v3的yolo_layer默认是90), 每行是一个五元向量(id, x, y, w, h), 描述了样本的一个bndbox

    // 循环读取每一张样本图, 对其进行变换(变换在一个虚拟的中间图上进行), 生成变换后的目标图
    for(i = 0; i < n; ++i){

        // 1.读取原始图
        // 最后两个参数w,h只要有一个为0, 那么返回原始图片; 如果w,h都不为0, 那么返回的就是原始图片按照w,h放缩后的结果 
        image orig = load_image_color(random_paths[i], 0, 0); // 轴向(c, 0, 1), BGR, 元素取值[0, 1](左闭右闭, 像素值/255.得到)

        // 2.创建目标图绘版(canvas)
        image sized = make_image(w, h, orig.c);
        fill_image(sized, .5);

        // 3.计算虚拟的中间图的形状参数, 宽和高分别为(nw, nh), 放入canvas时的偏移量为(dx, dy)

        // 一定范围内随机选取新宽高比
        float dw = jitter * orig.w; // 宽度抖动幅度
        float dh = jitter * orig.h; // 高度抖动幅度
        float new_ar = (orig.w + rand_uniform(-dw, dw)) / (orig.h + rand_uniform(-dh, dh)); // 抖动变形后图像宽高比

        //float scale = rand_uniform(.25, 2);
        float scale = 1; // yolov3开始不再随机选择缩放系数, 而是恒定使用1，即不缩放

        // 确定中间图的宽和高
        // 下面分两种情况计算, 猜想目的是为了避免将小于1的数做分母，避免浮点数上溢, 总之是为了保证计算的数值稳定性
        // 情况1: 宽 < 高, 先根据scale放缩h得到新的nh, 再根据新宽高比new_ar和nh计算出新的nw
        // 情况2: 宽 >= 高, 先根据scale放缩w得到新的nw, 再根据新宽高比new_ar和nw计算出新的nh
        // 两种情况的实质是: 保持new_ar中较长的边不变, 缩短较短的边, 使两边比例达到new_ar, 最终图片: 一条边不变另外一条边缩短
        float nw, nh;
        if(new_ar < 1){ // 情况1
            nh = scale * h;
            nw = nh * new_ar; // new_ar < 1, 所以nw < nh
        } else {        // 情况2
            nw = scale * w;
            nh = nw / new_ar; // new_ar >= 1, 所以 nh < nw
        }

        // (dx, dy)是将原始图"嵌入“画布时, 嵌入图和画布二者间左上角坐标的偏移量
        // 注意: 当nw大于w或nh大于h时, dx或dy是小于0的
        // rand_uniform函数自动选择两个输入参数中较小的作为min, 较大的作为max, 之后返回一个区间[min, max]上依均匀分布产生的随机数
        float dx = rand_uniform(0, w - nw);
        float dy = rand_uniform(0, h - nh);

        // 4.将原始图放缩到(nw, nh)后, 按照一定的中心平移(dx, dy)"嵌入"到画布sized中
        // 利用中间图尺寸(nw, nh)和放置时偏移量(dx, dy), 再加上place_image()函数中的中间图像素到原始图像素的映射规则, 共同实现jitter变换
        // 并不保证虚拟图上的每一点(x, y)的变换(x+dx, y+dy)都在目标图sized的合法范围内
        //  但是place_image()在实现将(x+dx, y+dy)写入sized中时使用了set_pixel()函数, 该函数会自动忽略对坐标不合法点值的设置请求
        place_image(orig, nw, nh, dx, dy, sized); // void place_image(image im, int w, int h, int dx, int dy, image canvas), 将原始图orig先缩放到(nw, nh), 之后再"嵌入"到画布sized之中, 嵌入时orig左上角相对于sized的左上角偏移量为(dx, dy)

        // 5.此函数先将输入的颜色图片由RGB颜色空间转换至hsv空间，而后在hsv空间的h,s,v三通道上在添加噪声，以实现数据增强
        random_distort_image(sized, hue, saturation, exposure);

        // 6.按照50%概率随机对部分样本图片进行水平翻转
        int flip = rand()%2;
        if(flip) flip_image(sized);
        d.X.vals[i] = sized.data;

        // 7.读取当前图片的标注信息
        // 注意对于检测任务, 图像的抖动、扭曲、平移和翻转都会导致真值信息的改变
        fill_truth_detection(random_paths[i], boxes, d.y.vals[i], classes, flip, -dx/w, -dy/h, nw/w, nh/h); // d.y.vals[i]中每5个元素是一个用于描述同一个bbox的5元组(x, y, w, h, id). 在ctw-baseline中, id就是类标的数字表示

        free_image(orig);
    }
    free(random_paths);
    return d;
}

void *load_thread(void *ptr)
{
    //printf("Loading data: %d\n", rand());
    load_args a = *(struct load_args*)ptr;
    if(a.exposure == 0) a.exposure = 1;
    if(a.saturation == 0) a.saturation = 1;
    if(a.aspect == 0) a.aspect = 1;

    if (a.type == OLD_CLASSIFICATION_DATA){
        *a.d = load_data_old(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h);
    } else if (a.type == REGRESSION_DATA){
        *a.d = load_data_regression(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    } else if (a.type == CLASSIFICATION_DATA){
        *a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes, a.hierarchy, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.center);
    } else if (a.type == SUPER_DATA){
        *a.d = load_data_super(a.paths, a.n, a.m, a.w, a.h, a.scale);
    } else if (a.type == WRITING_DATA){
        *a.d = load_data_writing(a.paths, a.n, a.m, a.w, a.h, a.out_w, a.out_h);
    } else if (a.type == ISEG_DATA){
        *a.d = load_data_iseg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.scale, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    } else if (a.type == INSTANCE_DATA){
        *a.d = load_data_mask(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.coords, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    } else if (a.type == SEGMENTATION_DATA){
        *a.d = load_data_seg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.scale);
    } else if (a.type == REGION_DATA){
        *a.d = load_data_region(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    } else if (a.type == DETECTION_DATA){ // train_detector()
        *a.d = load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    } else if (a.type == SWAG_DATA){
        *a.d = load_data_swag(a.paths, a.n, a.classes, a.jitter);
    } else if (a.type == COMPARE_DATA){
        *a.d = load_data_compare(a.n, a.paths, a.m, a.classes, a.w, a.h);
    } else if (a.type == IMAGE_DATA){
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = resize_image(*(a.im), a.w, a.h);
    } else if (a.type == LETTERBOX_DATA){ // validate_detector.*()
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = letterbox_image(*(a.im), a.w, a.h); // 保比例长边对齐放缩, 原图是a.im, 放缩画布是(a.w, a.h)
    } else if (a.type == TAG_DATA){
        *a.d = load_data_tag(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    }
    free(ptr);
    return 0;
}

pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}

/**
 * @brief 数据处理线程集群的总控线程(负责线程的工作分配)的工作函数
 *        总控线程负责启动n个线程构成线程集群, 为线程集群中每个线程分配全数据集中的一个需要加载的样本数作为其任务
 * @param ptr load_args *, 加载的数据保存在ptr->d中
 */
void *load_threads(void *ptr)
{
    int i;
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    data *out = args.d; // out指向一个data对象
    int total = args.n;
    free(ptr); // 虽然ptr被释放, 但ptr传入的数据存储空间的句柄已经由out接管
    data *buffers = calloc(args.threads, sizeof(data));
    pthread_t *threads = calloc(args.threads, sizeof(pthread_t));

    // 分配任务时, 每个线程只需要知道自己要都取多少张(args.n)图片就可以了, 因为每个线程都是通过随机选择的方法从全部args.m张图片中选出args.n张图片读取
    for(i = 0; i < args.threads; ++i){
        // 确定索引为i的工作线程用来保存"成果"的缓冲区, 以及它所需要处理的样本数
        args.d = buffers + i;

        // 将完成一个epoch所需全部样本集图片平均分给所有线程, 由于加载数据函数采用随机样本选择, 因此只需要传一个需要加载的样本总数, 不需要传每个线程处理的子batch的起始索引
        args.n = (i+1) * total/args.threads - i * total/args.threads;

        // 创建数据加载线程, 处理后的"成果"将会被保存在args中
        threads[i] = load_data_in_thread(args);
    }

    // 等待所有工作线程结束
    for(i = 0; i < args.threads; ++i){
        pthread_join(threads[i], 0);
    }

    // 将args.threads个data对象组成的数组buffsers中的数据合并成一个data对象out, 注意合并的时候其实是对每个矩阵逐行拷贝引用
    // 注意: 这里out句柄所控制的对象身份没有改变, 是在修改out所指向的data对象的全部属性的值
    *out = concat_datas(buffers, args.threads);
    out->shallow = 0;
    for(i = 0; i < args.threads; ++i){
        buffers[i].shallow = 1;
        free_data(buffers[i]);
    }
    free(buffers);
    free(threads);
    return 0;
}

void load_data_blocking(load_args args)
{
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    load_thread(ptr);
}

/**
 * @brief 启动一个线程集群, 负责读取args.n个图片数据, 非阻塞模式, 加载到的数据保存在args.d中. 在detector.c的 train_detector()函数中被调用
 *        输入参数args是训练数据相关的信息, 一部分来源于数据配置文件(数据集相关信息), 一部分来源于网络配置文件(网络输入数据格式相关信息)
 *        读取的样本共有args.n个, 但这些样本的每个都是从样本全集中随机抽取的(有放回的)
 * @note 注意数据加载过程中args不是不变的, 原因包括:
 *       (1)args要保存加载到的数据, 这部分属性会发生变化
 *       (2)args某些属性可能需要修改
 */
pthread_t load_data(load_args args)
{
    pthread_t thread; // 总控线程

    // 之所以重新分配, 是因为args还承担着保存加载到的训练数据的职责, 因此不是只读的
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    // 创建一个总控线程, 该线程会启全部具体执行数据读取和处理的线程, 并为每个数据加载线程分配任务
    if(pthread_create(&thread, 0, load_threads, ptr)) error("Thread creation failed");
    return thread;
}

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h)
{
    if(m) paths = get_random_paths(paths, n, m);
    char **replace_paths = find_replace_paths(paths, n, ".png", "-label.png");
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_image_paths_gray(replace_paths, n, out_w, out_h);
    if(m) free(paths);
    int i;
    for(i = 0; i < n; ++i) free(replace_paths[i]);
    free(replace_paths);
    return d;
}

data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_labels_paths(paths, n, labels, k, 0);
    if(m) free(paths);
    return d;
}

/*
   data load_data_study(char **paths, int n, int m, char **labels, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
   {
   data d = {0};
   d.indexes = calloc(n, sizeof(int));
   if(m) paths = get_random_paths_indexes(paths, n, m, d.indexes);
   d.shallow = 0;
   d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure);
   d.y = load_labels_paths(paths, n, labels, k);
   if(m) free(paths);
   return d;
   }
 */

data load_data_super(char **paths, int n, int m, int w, int h, int scale)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;

    int i;
    d.X.rows = n;
    d.X.vals = calloc(n, sizeof(float*));
    d.X.cols = w*h*3;

    d.y.rows = n;
    d.y.vals = calloc(n, sizeof(float*));
    d.y.cols = w*scale * h*scale * 3;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop = random_crop_image(im, w*scale, h*scale);
        int flip = rand()%2;
        if (flip) flip_image(crop);
        image resize = resize_image(crop, w, h);
        d.X.vals[i] = resize.data;
        d.y.vals[i] = crop.data;
        free_image(im);
    }

    if(m) free(paths);
    return d;
}

data load_data_regression(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0);
    d.y = load_regression_labels_paths(paths, n, k);
    if(m) free(paths);
    return d;
}

data select_data(data *orig, int *inds)
{
    data d = {0};
    d.shallow = 1;
    d.w = orig[0].w;
    d.h = orig[0].h;

    d.X.rows = orig[0].X.rows;
    d.y.rows = orig[0].X.rows;

    d.X.cols = orig[0].X.cols;
    d.y.cols = orig[0].y.cols;

    d.X.vals = calloc(orig[0].X.rows, sizeof(float *));
    d.y.vals = calloc(orig[0].y.rows, sizeof(float *));
    int i;
    for(i = 0; i < d.X.rows; ++i){
        d.X.vals[i] = orig[inds[i]].X.vals[i];
        d.y.vals[i] = orig[inds[i]].y.vals[i];
    }
    return d;
}

data *tile_data(data orig, int divs, int size)
{
    data *ds = calloc(divs*divs, sizeof(data));
    int i, j;
#pragma omp parallel for
    for(i = 0; i < divs*divs; ++i){
        data d;
        d.shallow = 0;
        d.w = orig.w/divs * size;
        d.h = orig.h/divs * size;
        d.X.rows = orig.X.rows;
        d.X.cols = d.w*d.h*3;
        d.X.vals = calloc(d.X.rows, sizeof(float*));

        d.y = copy_matrix(orig.y);
#pragma omp parallel for
        for(j = 0; j < orig.X.rows; ++j){
            int x = (i%divs) * orig.w / divs - (d.w - orig.w/divs)/2;
            int y = (i/divs) * orig.h / divs - (d.h - orig.h/divs)/2;
            image im = float_to_image(orig.w, orig.h, 3, orig.X.vals[j]);
            d.X.vals[j] = crop_image(im, x, y, d.w, d.h).data;
        }
        ds[i] = d;
    }
    return ds;
}

data resize_data(data orig, int w, int h)
{
    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;
    int i;
    d.X.rows = orig.X.rows;
    d.X.cols = w*h*3;
    d.X.vals = calloc(d.X.rows, sizeof(float*));

    d.y = copy_matrix(orig.y);
#pragma omp parallel for
    for(i = 0; i < orig.X.rows; ++i){
        image im = float_to_image(orig.w, orig.h, 3, orig.X.vals[i]);
        d.X.vals[i] = resize_image(im, w, h).data;
    }
    return d;
}

data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.w=size;
    d.h=size;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, center);
    d.y = load_labels_paths(paths, n, labels, k, hierarchy);
    if(m) free(paths);
    return d;
}

data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.w = size;
    d.h = size;
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0);
    d.y = load_tags_paths(paths, n, k);
    if(m) free(paths);
    return d;
}

matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    matrix m;
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = calloc(m1.rows + m2.rows, sizeof(float*));
    for(i = 0; i < m1.rows; ++i){
        m.vals[count++] = m1.vals[i];
    }
    for(i = 0; i < m2.rows; ++i){
        m.vals[count++] = m2.vals[i];
    }
    return m;
}

data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    d.w = d1.w;
    d.h = d1.h;
    return d;
}

data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i){
        data new = concat_data(d[i], out);
        free_data(out);
        out = new;
    }
    return out;
}

data load_categorical_data_csv(char *filename, int target, int k)
{
    data d = {0};
    d.shallow = 0;
    matrix X = csv_to_matrix(filename);
    float *truth_1d = pop_column(&X, target);
    float **truth = one_hot_encode(truth_1d, X.rows, k);
    matrix y;
    y.rows = X.rows;
    y.cols = k;
    y.vals = truth;
    d.X = X;
    d.y = y;
    free(truth_1d);
    return d;
}

data load_cifar10_data(char *filename)
{
    data d = {0};
    d.shallow = 0;
    long i,j;
    matrix X = make_matrix(10000, 3072);
    matrix y = make_matrix(10000, 10);
    d.X = X;
    d.y = y;

    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    for(i = 0; i < 10000; ++i){
        unsigned char bytes[3073];
        fread(bytes, 1, 3073, fp);
        int class = bytes[0];
        y.vals[i][class] = 1;
        for(j = 0; j < X.cols; ++j){
            X.vals[i][j] = (double)bytes[j+1];
        }
    }
    scale_data_rows(d, 1./255);
    //normalize_data_rows(d);
    fclose(fp);
    return d;
}

void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = rand()%d.X.rows;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

/**
 * @brief 从保存着一个big batch的训练数据的缓冲区d中取出一个small data的数据拷贝到网络输入矩阵net->input
 *
 * @param d 保存着一个big batch的训练数据的缓冲区
 * @param n net->input可接收的样本数. 也是一个small batch的大小
 * @param offset 从d中复制数据的起点的样本索引号
 * @param X 返回参数, 保存从d中复制的一个small batch的样本图片数据
 * @param y 返回参数, 保存从d中复制的一个small batch的样本真值数据
 */
void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        if(y) memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void smooth_data(data d)
{
    int i, j;
    float scale = 1. / d.y.cols;
    float eps = .1;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; ++j){
            d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
        }
    }
}

data load_all_cifar10()
{
    data d = {0};
    d.shallow = 0;
    int i,j,b;
    matrix X = make_matrix(50000, 3072);
    matrix y = make_matrix(50000, 10);
    d.X = X;
    d.y = y;


    for(b = 0; b < 5; ++b){
        char buff[256];
        sprintf(buff, "data/cifar/cifar-10-batches-bin/data_batch_%d.bin", b+1);
        FILE *fp = fopen(buff, "rb");
        if(!fp) file_error(buff);
        for(i = 0; i < 10000; ++i){
            unsigned char bytes[3073];
            fread(bytes, 1, 3073, fp);
            int class = bytes[0];
            y.vals[i+b*10000][class] = 1;
            for(j = 0; j < X.cols; ++j){
                X.vals[i+b*10000][j] = (double)bytes[j+1];
            }
        }
        fclose(fp);
    }
    //normalize_data_rows(d);
    scale_data_rows(d, 1./255);
    smooth_data(d);
    return d;
}

data load_go(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    matrix X = make_matrix(3363059, 361);
    matrix y = make_matrix(3363059, 361);
    int row, col;

    if(!fp) file_error(filename);
    char *label;
    int count = 0;
    while((label = fgetl(fp))){
        int i;
        if(count == X.rows){
            X = resize_matrix(X, count*2);
            y = resize_matrix(y, count*2);
        }
        sscanf(label, "%d %d", &row, &col);
        char *board = fgetl(fp);

        int index = row*19 + col;
        y.vals[count][index] = 1;

        for(i = 0; i < 19*19; ++i){
            float val = 0;
            if(board[i] == '1') val = 1;
            else if(board[i] == '2') val = -1;
            X.vals[count][i] = val;
        }
        ++count;
        free(label);
        free(board);
    }
    X = resize_matrix(X, count);
    y = resize_matrix(y, count);

    data d = {0};
    d.shallow = 0;
    d.X = X;
    d.y = y;


    fclose(fp);

    return d;
}


void randomize_data(data d)
{
    int i;
    for(i = d.X.rows-1; i > 0; --i){
        int index = rand()%i;
        float *swap = d.X.vals[index];
        d.X.vals[index] = d.X.vals[i];
        d.X.vals[i] = swap;

        swap = d.y.vals[index];
        d.y.vals[index] = d.y.vals[i];
        d.y.vals[i] = swap;
    }
}

void scale_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        scale_array(d.X.vals[i], d.X.cols, s);
    }
}

void translate_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        translate_array(d.X.vals[i], d.X.cols, s);
    }
}

data copy_data(data d)
{
    data c = {0};
    c.w = d.w;
    c.h = d.h;
    c.shallow = 0;
    c.num_boxes = d.num_boxes;
    c.boxes = d.boxes;
    c.X = copy_matrix(d.X);
    c.y = copy_matrix(d.y);
    return c;
}

void normalize_data_rows(data d)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        normalize_array(d.X.vals[i], d.X.cols);
    }
}

data get_data_part(data d, int part, int total)
{
    data p = {0};
    p.shallow = 1;
    p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
    p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
    p.X.cols = d.X.cols;
    p.y.cols = d.y.cols;
    p.X.vals = d.X.vals + d.X.rows * part / total;
    p.y.vals = d.y.vals + d.y.rows * part / total;
    return p;
}

data get_random_data(data d, int num)
{
    data r = {0};
    r.shallow = 1;

    r.X.rows = num;
    r.y.rows = num;

    r.X.cols = d.X.cols;
    r.y.cols = d.y.cols;

    r.X.vals = calloc(num, sizeof(float *));
    r.y.vals = calloc(num, sizeof(float *));

    int i;
    for(i = 0; i < num; ++i){
        int index = rand()%d.X.rows;
        r.X.vals[i] = d.X.vals[index];
        r.y.vals[i] = d.y.vals[index];
    }
    return r;
}

data *split_data(data d, int part, int total)
{
    data *split = calloc(2, sizeof(data));
    int i;
    int start = part*d.X.rows/total;
    int end = (part+1)*d.X.rows/total;
    data train;
    data test;
    train.shallow = test.shallow = 1;

    test.X.rows = test.y.rows = end-start;
    train.X.rows = train.y.rows = d.X.rows - (end-start);
    train.X.cols = test.X.cols = d.X.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.X.vals = calloc(train.X.rows, sizeof(float*));
    test.X.vals = calloc(test.X.rows, sizeof(float*));
    train.y.vals = calloc(train.y.rows, sizeof(float*));
    test.y.vals = calloc(test.y.rows, sizeof(float*));

    for(i = 0; i < start; ++i){
        train.X.vals[i] = d.X.vals[i];
        train.y.vals[i] = d.y.vals[i];
    }
    for(i = start; i < end; ++i){
        test.X.vals[i-start] = d.X.vals[i];
        test.y.vals[i-start] = d.y.vals[i];
    }
    for(i = end; i < d.X.rows; ++i){
        train.X.vals[i-(end-start)] = d.X.vals[i];
        train.y.vals[i-(end-start)] = d.y.vals[i];
    }
    split[0] = train;
    split[1] = test;
    return split;
}

