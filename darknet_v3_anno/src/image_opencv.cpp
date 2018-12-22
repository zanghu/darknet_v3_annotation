#ifdef OPENCV

#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image.h"

using namespace cv;

extern "C" {

IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

/**
 * @brief 将图像从cv::IplImage对象转化为darkenet的image对象
 * @note  转化后图像im.data中的轴向是(c, h, w)也就是python图像处理中的(c, 0, 1), w是图像的水平轴, h是图像的垂直轴
 *        按照cv2.imread读取到的numpy.ndarray对象的轴向, 是(c, 0, 1)
 *        由于cv::IplImage是RGB图, 所以转化的image对象channel序也是RGB
 */
image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                // cv::IplImage的imageData属性采用的是类似(h, w, c)的轴向，即cv2.imread加载后的(0, 1, c)轴向
                // image.data中的轴向是(c, 0, 1), 也就是(c, h, w)
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.; // 将unsigned char转化为float
            }
        }
    }
    return im;
}

Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy); // RGB图转BGR图

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

/**
 * @brief cv::Mat 转换为 cv::IplImage
 * @note  opencv里有三种表示图像的数据结构, 分别是: cvMat, Mat和IplImage, 三者之间可以相互转化
 *        从查到的资料看, Mat更适合数学运算(矩阵乘法之类), 而另外两者则更加突出"图像"特性, 适合进行图像操作
 */
image mat_to_image(Mat m)
{
    IplImage ipl = m; // cv::Mat转为cv::IplImage
    image im = ipl_to_image(&ipl);
    rgbgr_image(im); // RGB图转BGR图
    return im;
}

void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return (void *) cap;
}

image get_image_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_image(0,0,0);
    return mat_to_image(m);
}

/**
 * @brief 如果编译时开启opencv支持, 则使用此函数从指定路径加载图片
 */
image load_image_cv(char *filename, int channels)
{
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m;
    m = imread(filename, flag); // 图片加载为cv::Mat对象

    // 某些图片使用opencv加载确实会失败
    if(!m.data){
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename); // 组装shell命令
        system(buff);
        return make_image(10,10,3); // 生成一个10x10x3的image对象
        //exit(0);
    }
    image im = mat_to_image(m);
    return im;
}

int show_image_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}

void make_window(char *name, int w, int h, int fullscreen)
{
    namedWindow(name, WINDOW_NORMAL); 
    if (fullscreen) {
        setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}

}

#endif
