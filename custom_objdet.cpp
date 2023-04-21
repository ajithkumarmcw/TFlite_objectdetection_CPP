#include <iostream>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/register.h"
#include "opencv2/opencv.hpp"

int main() {
    // Load TFLite model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("./ssd_mobilenet_v1_1_metadata_1.tflite");
    if (!model) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to build interpreter!" << std::endl;
        return -1;
    }

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!" << std::endl;
        return -1;
    }

    // Get input and output tensors
    TfLiteTensor* input_tensor = interpreter->input_tensor(0);
    TfLiteTensor* output_boxes_tensor = interpreter->output_tensor(0);
    TfLiteTensor* output_scores_tensor = interpreter->output_tensor(1);
    TfLiteTensor* output_classes_tensor = interpreter->output_tensor(2);

    // Load image
    cv::Mat image = cv::imread("./dog.jpg");

    // Preprocess image
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(input_tensor->dims->data[2], input_tensor->dims->data[1]));
    //memcpy(input_tensor->data.f, image.data, sizeof(float) * input_tensor->dims->data[1] * input_tensor->dims->data[2] * input_tensor->dims->data[3]);


    if (input_tensor->type == kTfLiteFloat32) {
        memcpy(input_tensor->data.f, image.data, sizeof(float) * input_tensor->dims->data[1] * input_tensor->dims->data[2] * input_tensor->dims->data[3]);
    } else if (input_tensor->type == kTfLiteUInt8) {
        std::cout << "input type uint8" << std::endl;
        memcpy(input_tensor->data.uint8, image.data, sizeof(uint8_t) * input_tensor->dims->data[1] * input_tensor->dims->data[2] * input_tensor->dims->data[3]);
        // // Convert image to uint8_t
        // cv::Mat image_uint8;
        // image.convertTo(image_uint8, CV_8UC3);

        // // Copy data to tensor buffer
        // const int image_height = input_tensor->dims->data[1];
        // const int image_width = input_tensor->dims->data[2];
        // const int channels = input_tensor->dims->data[3];
        // const int image_size = image_height * image_width * channels;
        // uint8_t* input_data = input_tensor->data.uint8;
        // for (int i = 0; i < image_size; i++) {
        //     const int pixel_idx = i / channels;
        //     const int channel_idx = i % channels;
        //     const int image_idx = pixel_idx * channels + (channels - 1 - channel_idx);
        //     input_data[i] = image_uint8.data[image_idx];
        // }
    } else {
        std::cerr << "Input tensor data type not supported!" << std::endl;
        return -1;
    }

    // Perform inference
    interpreter->Invoke();

    // // Postprocess detections
    float* boxes = output_boxes_tensor->data.f;
    float* scores = output_scores_tensor->data.f;
    float* classes = output_classes_tensor->data.f;
    std::vector<cv::Rect> detections;
    for (int i = 0; i < output_scores_tensor->dims->data[1]; i++) {
        std::cout << " reach scores " << (int)scores[i]<< std::endl;
        std::cout << " reach classes " << (int)classes[i]<< std::endl;
        if ((scores[i] > 0.5) && (scores[i] <= 1.0)) {
            int xmin = static_cast<int>(boxes[i * 4 + 1] * image.cols);
            int ymin = static_cast<int>(boxes[i * 4 + 0] * image.rows);
            int xmax = static_cast<int>(boxes[i * 4 + 3] * image.cols);
            int ymax = static_cast<int>(boxes[i * 4 + 2] * image.rows);
            detections.emplace_back(xmin, ymin, xmax - xmin, ymax - ymin);
            // Draw bounding box on original image
            cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0), 2);


        }
    }

    // Display image
    cv::imshow("Output", image);
    cv::waitKey(0);

    return 0;
}