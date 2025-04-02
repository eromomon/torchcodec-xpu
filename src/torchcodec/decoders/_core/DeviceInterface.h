// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include "FFMPEGCommon.h"
#include "src/torchcodec/decoders/_core/VideoDecoder.h"

namespace facebook::torchcodec {

// Note that all these device functions should only be called if the device is
// not a CPU device. CPU device functions are already implemented in the
// VideoDecoder implementation.
// These functions should only be called from within an if block like this:
// if (device.type() != torch::kCPU) {
//   deviceFunction(device, ...);
// }

class DeviceInterface {
 public:
  DeviceInterface(const std::string& device) : device_(device) {}

  virtual ~DeviceInterface(){};

  torch::Device& device() {
    return device_;
  };

  virtual std::optional<const AVCodec*> findCodec(const AVCodecID& codecId) = 0;

  // Initialize the hardware device that is specified in `device`. Some builds
  // support CUDA and others only support CPU.
  virtual void initializeContext(AVCodecContext* codecContext) = 0;

  virtual void convertAVFrameToFrameOutput(
      const VideoDecoder::VideoStreamOptions& videoStreamOptions,
      UniqueAVFrame& avFrame,
      VideoDecoder::FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt) = 0;

  virtual void releaseContext(AVCodecContext* codecContext) = 0;

 protected:
  torch::Device device_;
};

using CreateDeviceInterfaceFn =
    std::function<DeviceInterface*(const std::string& device)>;

bool registerDeviceInterface(
    const std::string deviceType,
    const CreateDeviceInterfaceFn createInterface);

std::shared_ptr<DeviceInterface> createDeviceInterface(
    const std::string device);

} // namespace facebook::torchcodec
