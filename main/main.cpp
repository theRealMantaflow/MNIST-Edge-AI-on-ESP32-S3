#include "driver/usb_serial_jtag.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
// #include <hal/usb_serial_jtag_ll.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "esp_rom_crc.h"
#include "model_weights.h"

static const char* TAG = "USB_COMMS";

// Defining each layer of the model;
class LinearLayer {
    const int8_t* weights;
    const int8_t* biases;
    float w_scale;
    float b_scale;
    int in_features;
    int out_features;
    bool use_relu;

   public:
    LinearLayer(const int8_t* w, float w_s, const int8_t* b, float b_s,
                int in_f, int out_f, bool relu)
        : weights(w),
          biases(b),
          w_scale(w_s),
          b_scale(b_s),
          in_features(in_f),
          out_features(out_f),
          use_relu(relu) {}

    void operator()(const float* input, float* output) const {
        for (int o = 0; o < out_features; o++) {
            float acc = 0.0f;
            for (int i = 0; i < in_features; i++) {
                acc +=
                    input[i] * static_cast<float>(weights[o * in_features + i]);
            }

            acc = (acc / w_scale) + (static_cast<float>(biases[o]) / b_scale);
            if (use_relu) {
                acc = std::max(0.0f, acc);
            }
            output[o] = acc;
        }
    }
};

// Defining the complete model with three linear layers
class LinearModel {
   private:
    LinearLayer layer1;
    LinearLayer layer2;
    LinearLayer layer3;

   public:
    LinearModel()
        : layer1(network_0_weight, network_0_weight_scale, network_0_bias,
                 network_0_bias_scale, 784, 100, true),
          layer2(network_2_weight, network_2_weight_scale, network_2_bias,
                 network_2_bias_scale, 100, 40, true),
          layer3(network_4_weight, network_4_weight_scale, network_4_bias,
                 network_4_bias_scale, 40, 10, false) {}

    void get_prediction(const float* input_layer, float* hidden_layer1,
                        float* hidden_layer2, float* output_layer) {
        layer1(input_layer, hidden_layer1);
        layer2(hidden_layer1, hidden_layer2);
        layer3(hidden_layer2, output_layer);
    }
};

class InferenceEngine {
   private:
    static const uint16_t kMagic = 0xAA55;
    static const uint16_t kEndMarker = 0x55AA;
    static const size_t kImageSize = 784;

    void usb_print(const char* str) {
        if (str != nullptr) {
            usb_serial_jtag_write_bytes((const uint8_t*)str, strlen(str),
                                        portMAX_DELAY);
        }
    }

    // Reads the exact number of bytes from the USB serial/jtag interface into
    // the provided buffer.
    bool read_exact(uint8_t* buffer, size_t length) {
        size_t offset = 0;
        while (offset < length) {
            int bytes_read = usb_serial_jtag_read_bytes(
                buffer + offset, length - offset, portMAX_DELAY);
            if (bytes_read <= 0) {
                ESP_LOGE(TAG, "USB read failed while reading %u bytes",
                         static_cast<unsigned int>(length));
                return false;
            }
            offset += static_cast<size_t>(bytes_read);
        }

        return true;
    }

   public:
    InferenceEngine() {}
    void run() {
        // Configure the usb jtag driver for bigger buffers to handle the image
        // data
        usb_serial_jtag_driver_config_t usb_config = {
            .tx_buffer_size = 256,
            .rx_buffer_size = 1024,
        };

        esp_err_t ret = usb_serial_jtag_driver_install(&usb_config);
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, "USB Serial/JTAG driver installed successfully!");
        } else {
            ESP_LOGE(TAG,
                     "Failed to install USB Serial/JTAG driver! Error code: %d",
                     ret);
        }

        // send READY every 1000 ms to host till the host sends back ACK
        while (true) {
            usb_print("READY\n");
            uint8_t ack_buffer[4];
            int bytes_read = usb_serial_jtag_read_bytes(ack_buffer, 4, 1000);
            if (bytes_read == 4 && std::memcmp(ack_buffer, "ACK\n", 4) == 0) {
                ESP_LOGI(TAG, "Received ACK from host. Starting inference.");
                break;
            }
            ESP_LOGI(TAG, "Waiting for ACK from host...");
            vTaskDelay(pdMS_TO_TICKS(1000));
        }

        static uint8_t raw_usb_pixels[kImageSize];
        // Buffers for the model layers
        static float input_buffer[kImageSize];
        static float hidden1_buffer[100];
        static float hidden2_buffer[40];
        static float output_buffer[10];

        LinearModel model;

        // Main loop
        while (true) {
            uint16_t magic = 0;
            uint16_t image_len = 0;
            uint32_t crc_value = 0;
            uint16_t end_marker = 0;

            if (!read_exact(reinterpret_cast<uint8_t*>(&magic),
                            sizeof(magic)) ||
                magic != kMagic) {
                ESP_LOGE(TAG, "Invalid packet magic: 0x%04x", magic);
                continue;
            }

            if (!read_exact(reinterpret_cast<uint8_t*>(&image_len),
                            sizeof(image_len))) {
                ESP_LOGE(TAG, "Failed to read image length!");
                continue;
            }
            if (image_len != kImageSize) {
                ESP_LOGE(TAG, "Unexpected image length: %u",
                         static_cast<unsigned int>(image_len));
                continue;
            }

            if (!read_exact(raw_usb_pixels, image_len)) {
                ESP_LOGE(TAG, "Failed to read image data!");
                continue;
            }

            if (!read_exact(reinterpret_cast<uint8_t*>(&crc_value),
                            sizeof(crc_value))) {
                ESP_LOGE(TAG, "Failed to read CRC value!");
                continue;
            }

            if (!read_exact(reinterpret_cast<uint8_t*>(&end_marker),
                            sizeof(end_marker)) ||
                end_marker != kEndMarker) {
                ESP_LOGE(TAG, "Invalid packet end marker: 0x%04x", end_marker);
                continue;
            }

            // Check the image integrity using CRC32
            uint32_t calculated_crc =
                esp_rom_crc32_le(0, raw_usb_pixels, image_len);

            if (calculated_crc != crc_value) {
                ESP_LOGE(TAG,
                         "CRC Mismatch! Received: 0x%08lx, Calculated: 0x%08lx",
                         static_cast<unsigned long>(crc_value),
                         static_cast<unsigned long>(calculated_crc));
                continue;
            }

            for (int i = 0; i < kImageSize; i++) {
                input_buffer[i] = static_cast<float>(raw_usb_pixels[i]) / 255.0f;
            }

            // Pass the buffers through the functor chain
            model.get_prediction(input_buffer, hidden1_buffer, hidden2_buffer,
                                 output_buffer);

            // Finding the predicted digit
            int predicted_digit = 0;
            float max_val = output_buffer[0];
            for (int i = 1; i < 10; i++) {
                if (output_buffer[i] > max_val) {
                    max_val = output_buffer[i];
                    predicted_digit = i;
                }
            }

            char prediction_str[64];
            snprintf(prediction_str, sizeof(prediction_str),
                     "Predicted Digit: %d\n", predicted_digit);
            usb_print(prediction_str);
        }
    }
};

extern "C" void app_main(void) {
    InferenceEngine engine;
    engine.run();
}