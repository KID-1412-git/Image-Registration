#include <chrono>
#include <cmath>
#include <iostream>
#include "CL/sycl.hpp"
#include "dpc_common.hpp"


#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

using namespace std;
using namespace sycl;

static void ReportTime(const string& msg, event e) {
    cl_ulong time_start =
        e.get_profiling_info<info::event_profiling::command_start>();

    cl_ulong time_end =
        e.get_profiling_info<info::event_profiling::command_end>();

    double elapsed = (time_end - time_start) / 1e6;
    cout << msg << elapsed << " milliseconds\n";
}


//���ƽ�����㷨����������ģ����ͼƬ�����ͣ����洢������result��
__attribute__((always_inline)) static void ApplyFilter(uint8_t* I,
    uint8_t* T,
    float* result,
    int i,
    int j,
    int Iw,
    int Ih,
    int Tw,
    int Th) {

    if (i >= Ih - Th + 1 || j >= Iw - Tw + 1) {
        return;
    }
    float sum = 0.0;
    for (int k = 0; k < Th; k++) {
        for (int s = 0; s < Tw; s++) {
            float diff = I[(i + k) * Iw + j + s] - T[k * Tw + s];
            sum += diff * diff;
        }
    }
    result[i * Iw + j] = sum;
}


int main(int argc, char** argv) {
    // ����ԭʼͼƬ��ͼƬ���ߡ�ͨ����
    int src_img_width, src_img_height, src_channels;
    // ʹ�ûҶ�ͼ��
    // ����ͼƬ ԴͼƬ
    uint8_t* src_image = stbi_load("./1.jpg", &src_img_width, &src_img_height, &src_channels, 1);
    if (src_image == NULL) {
        cout << "Error in loading the image\n";
        exit(1);
    }
    //���ԭͼƬ�Ŀ��ߡ�ͨ����
    cout << "Loaded src image with a width of " << src_img_width << ", a height of "
        << src_img_height << " and " << src_channels << " channels\n";

    // ����ģ��ͼƬ
    int template_img_width, template_img_height, template_channels;
    // ����ͼƬ ģ��ͼƬ
    uint8_t* template_image = stbi_load("./2.jpg", &template_img_width, &template_img_height, &template_channels, 1);
    if (template_image == NULL) {
        cout << "Error in loading the image\n";
        exit(1);
    }
    cout << "Loaded template image with a width of " << template_img_width << ", a height of "
        << template_img_height << " and " << template_channels << " channels\n";
    //ԭͼƬ�ߴ�С��ģ��ͼƬ�������
    if (src_img_width < template_img_width || src_img_height < template_img_height) {
        cout << "Error: The template is larger than the picture\n";
        exit(1);
    }


    // ����Ľ���ڴ�
    size_t num_counts = src_img_height * src_img_width;
    size_t src_size = src_img_height * src_img_width;
    size_t template_size = template_img_width * template_img_height;
    // �������ͼ����ڴ�
    // �������������ڴ棬��¼���
    float* result = new float[num_counts];
    // ��ʼ��

    event e1, e2;

    try {
        // ѡ�����ʺϵ��豸
        auto prop_list = property_list{ property::queue::enable_profiling() };
        queue q(default_selector{}, dpc_common::exception_handler, prop_list);

        // �鿴����ʹ�õ��豸
        cout << "Running on " << q.get_device().get_info<info::device::name>()
            << "\n";
        // Դͼ��buffer
        buffer src_image_buf(src_image, range(src_size));
        // ģ��ͼ��buffer
        buffer template_image_buf(template_image, range(template_size));
        // �����buffer
        buffer result_buf(result, range(num_counts));
        cout << "Submitting lambda kernel...\n";


        // �õ�����ıȽ�֮��Ľ��
        e1 = q.submit([&](auto& h) {
            //����buffer�ķ�����
            accessor src_image_acc(src_image_buf, h, read_only);
        accessor template_image_acc(template_image_buf, h, read_only);
        accessor result_acc(result_buf, h, write_only);
        // ʹ�ö�ά�߳���
        h.parallel_for(range<2>{(size_t)src_img_height, (size_t)src_img_width}, [=](id<2> index) {
            // �ں˳���ִ��
            ApplyFilter(src_image_acc.get_pointer(), template_image_acc.get_pointer(), result_acc.get_pointer(), index[0], index[1], src_img_width, src_img_height, template_img_width, template_img_height);
            });
            });
        q.wait_and_throw();
    }
    catch (sycl::exception e) {
        cout << "SYCL exception caught: " << e.what() << "\n";
        return 1;
    }

    // report execution times:
    ReportTime("Lambda kernel time: ", e1);


    // �õ�ƥ��λ�õ���Сֵ
    int x, y;
    float minresult = result[0];
    for (int i = 0; i < src_img_height - template_img_height + 1; i++) {
        for (int j = 0; j < src_img_width - template_img_width + 1; j++) {
            if (minresult > result[i * src_img_width + j]) {
                y = i;
                x = j;
                minresult = result[i * src_img_width + j];
            }
        }
    }

    int x1 = x;
    int x2 = x + template_img_width - 1;
    int y1 = y;
    int y2 = y + template_img_height - 1;

    cout << x1 << "  " << x2 << "  " << y1 << "  " << y2 << "  ";

    // ��ͼƬ���б��棬��ƥ������򻭿��ע

    // �ȱ����������
    for (int i = x1; i <= x2; i++) {
        src_image[y1 * src_img_width + i] = 255;
        src_image[y2 * src_img_width + i] = 255;
    }
    //����
    for (int i = y1 + 1; i < y2; i++) {
        src_image[i * src_img_width + x1] = 255;
        src_image[i * src_img_width + x2] = 255;
    }

    stbi_write_png("sepia_ref.png", src_img_width, src_img_height, src_channels, src_image, src_img_width * src_channels);
    return 0;
}
