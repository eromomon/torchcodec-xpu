#include "src/torchcodec/decoders/_core/DeviceInterface.h"
#include "src/torchcodec/decoders/_core/FFMPEGCommon.h"
#include "src/torchcodec/decoders/_core/VideoDecoder.h"

extern "C" {
//#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

const int MAX_XPU_GPUS = 128;
// Set to -1 to have an infinitely sized cache. Set it to 0 to disable caching.
// Set to a positive number to have a cache of that size.
const int MAX_CONTEXTS_PER_GPU_IN_CACHE = -1;
std::vector<AVBufferRef*> g_cached_hw_device_ctxs[MAX_XPU_GPUS];
std::mutex g_cached_hw_device_mutexes[MAX_XPU_GPUS];

torch::DeviceIndex getFFMPEGCompatibleDeviceIndex(const torch::Device& device) {
  torch::DeviceIndex deviceIndex = device.index();
  deviceIndex = std::max<at::DeviceIndex>(deviceIndex, 0);
  TORCH_CHECK(deviceIndex >= 0, "Device index out of range");
  // For single GPU- machines libtorch returns -1 for the device index. So for
  // that case we set the device index to 0.
  return deviceIndex;
}

AVBufferRef* getFromCache(const torch::Device& device) {
  torch::DeviceIndex deviceIndex = getFFMPEGCompatibleDeviceIndex(device);
  if (static_cast<int>(deviceIndex) >= MAX_XPU_GPUS) {
    return nullptr;
  }
  std::scoped_lock lock(g_cached_hw_device_mutexes[deviceIndex]);
  if (g_cached_hw_device_ctxs[deviceIndex].size() > 0) {
    AVBufferRef* hw_device_ctx = g_cached_hw_device_ctxs[deviceIndex].back();
    g_cached_hw_device_ctxs[deviceIndex].pop_back();
    return hw_device_ctx;
  }
  return nullptr;
}

AVBufferRef* getFFMPEGContextFromNewCudaContext(
    [[maybe_unused]] const torch::Device& device,
    torch::DeviceIndex nonNegativeDeviceIndex,
    enum AVHWDeviceType type) {
  AVBufferRef* hw_device_ctx = nullptr;
  std::string deviceOrdinal = std::to_string(nonNegativeDeviceIndex);
  int err = av_hwdevice_ctx_create(
      &hw_device_ctx, type, deviceOrdinal.c_str(), nullptr, 0);
  if (err < 0) {
    TORCH_CHECK(
        false,
        "Failed to create specified HW device",
        getFFMPEGErrorStringFromErrorCode(err));
  }
  return hw_device_ctx;
}

AVBufferRef* getVaapiContext(const torch::Device& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("vaapi");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find vaapi device");
  torch::DeviceIndex nonNegativeDeviceIndex =
      getFFMPEGCompatibleDeviceIndex(device);

  AVBufferRef* hw_device_ctx = getFromCache(device);
  if (hw_device_ctx != nullptr) {
    return hw_device_ctx;
  }

  std::string deviceOrdinal = std::to_string(nonNegativeDeviceIndex);
  int err = av_hwdevice_ctx_create(
      &hw_device_ctx, type, "/dev/dri/renderD128", nullptr, 0);
  if (err < 0) {
    TORCH_CHECK(
        false,
        "Failed to create specified HW device: ",
        getFFMPEGErrorStringFromErrorCode(err));
  }
  return hw_device_ctx;
}

void throwErrorIfNonXpuDevice(const torch::Device& device) {
  TORCH_CHECK(
      device.type() != torch::kCPU,
      "Device functions should only be called if the device is not CPU.")
  if (device.type() != torch::kXPU) {
    throw std::runtime_error("Unsupported device: " + device.str());
  }
}
} // namespace

void initializeContextOnXpu(
    const torch::Device& device,
    AVCodecContext* codecContext) {
  throwErrorIfNonXpuDevice(device);
  // It is important for pytorch itself to create the xpu context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the xpu context.
  torch::Tensor dummyTensorForCudaInitialization = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
  codecContext->hw_device_ctx = getVaapiContext(device);
  return;
}

void convertAVFrameToFrameOutputOnXpu(
    const torch::Device& device,
    const VideoDecoder::VideoStreamOptions& videoStreamOptions,
    VideoDecoder::AVFrameStream& avFrameStream,
    VideoDecoder::FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  AVFrame* avFrame = avFrameStream.avFrame.get();

  TORCH_CHECK(
      avFrame->format == AV_PIX_FMT_VAAPI,
      "Expected format to be AV_PIX_FMT_VAAPI, got " +
          std::string(av_get_pix_fmt_name((AVPixelFormat)avFrame->format)));
}


// inspired by https://github.com/FFmpeg/FFmpeg/commit/ad67ea9
// we have to do this because of an FFmpeg bug where hardware decoding is not
// appropriately set, so we just go off and find the matching codec for the CUDA
// device
std::optional<const AVCodec*> findXpuCodec(
    const torch::Device& device,
    const AVCodecID& codecId) {
  throwErrorIfNonXpuDevice(device);

  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    if (codec->id != codecId || !av_codec_is_decoder(codec)) {
      continue;
    }

    const AVCodecHWConfig* config = nullptr;
    for (int j = 0; (config = avcodec_get_hw_config(codec, j)) != nullptr;
         ++j) {
      if (config->device_type == AV_HWDEVICE_TYPE_VAAPI) {
	printf(">>> dvrogozh: findXpuCodec: %p\n", codec);
        return codec;
      }
    }
  }

  printf(">>> dvrogozh: findXpuCodec: null\n");
  return std::nullopt;
}

} // namespace facebook::torchcodec
