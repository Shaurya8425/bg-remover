import type * as ort from 'onnxruntime-web/webgpu';
import { ModelCache } from '../utils/modelCache';

const ORT_RUNTIME_URL =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.webgpu.min.mjs';
type OrtRuntime = typeof ort;
let ortRuntime: OrtRuntime | null = null;

// Detect mobile environments (iOS / Android)
const isMobile = typeof navigator !== 'undefined' && /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

async function getOrtRuntime(): Promise<OrtRuntime> {
  if (!ortRuntime) {
    const runtime = await import(/* @vite-ignore */ ORT_RUNTIME_URL);
    runtime.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';
    
    // Lock to 1 thread on mobile to prevent WASM SIMD memory corruption on iOS/Android
    runtime.env.wasm.numThreads = isMobile ? 1 : Math.max(1, Math.floor((navigator.hardwareConcurrency || 4) / 2));
    ortRuntime = runtime;
  }

  return ortRuntime as OrtRuntime;
}

let session: ort.InferenceSession | null = null;
let isBusy = false;
const cache = new ModelCache();

export interface WorkerMessageData {
  type: 'PROCESS_IMAGE';
  imageData: ImageData;
}

self.onmessage = async (event: MessageEvent<WorkerMessageData>) => {
  const { type, imageData } = event.data;

  if (type !== 'PROCESS_IMAGE' || !imageData) return;

  if (isBusy) {
    self.postMessage({
      type: 'ERROR',
      error: 'Inference is already in progress. Please wait.',
    });
    return;
  }

  isBusy = true;

  try {
    const ort = await getOrtRuntime();

    if (!session) {
      const modelUrl = 'https://huggingface.co/shaurya7084/bg-removal-onnx/resolve/main/model_quantized.onnx';
      let modelBuffer = await cache.getAsset(modelUrl);

      if (!modelBuffer) {
        self.postMessage({ type: 'PROGRESS', progress: 10, key: 'Downloading Model' });
        const response = await fetch(modelUrl);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        modelBuffer = await response.arrayBuffer();
        await cache.setAsset(modelUrl, modelBuffer.slice(0));
      }

      self.postMessage({ type: 'PROGRESS', progress: 35, key: 'Initializing ONNX Engine' });

      const modelBytes = new Uint8Array(modelBuffer);

      // Mobile WebGPU / WebGL Execution Provider fallbacks
      try {
        session = await ort.InferenceSession.create(modelBytes, {
          executionProviders: isMobile ? ['webgl', 'wasm'] : ['webgpu', 'wasm'],
          graphOptimizationLevel: 'all',
        });
      } catch {
        session = await ort.InferenceSession.create(modelBytes, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'basic',
        });
      }
    }

    self.postMessage({ type: 'PROGRESS', progress: 50, key: 'Processing Tensor' });

    const targetSize = 1024;
    const tensor = prepareTensor(imageData, targetSize, ort);

    const inputName = session.inputNames[0];
    const feeds: Record<string, ort.Tensor> = { [inputName]: tensor };

    self.postMessage({ type: 'PROGRESS', progress: 70, key: 'Running Neural Inference' });
    const results = await session.run(feeds);

    self.postMessage({ type: 'PROGRESS', progress: 85, key: 'Refining Mask' });

    const outputName = session.outputNames[0];
    const outputTensor = results[outputName];
    const rawFloatData = outputTensor.data as Float32Array;

    if (!rawFloatData) throw new Error('Failed to extract tensor output array.');

    const maskPixels = postProcessMask(rawFloatData, targetSize);
    const transferBuffer = maskPixels.buffer as ArrayBuffer;

    (self as unknown as Worker).postMessage(
      {
        type: 'SUCCESS',
        maskPixels,
        maskWidth: targetSize,
        maskHeight: targetSize,
      },
      [transferBuffer]
    );
  } catch (err: any) {
    self.postMessage({
      type: 'ERROR',
      error: err?.message || 'Inference execution failed.',
    });
  } finally {
    isBusy = false;
  }
};

/**
 * Normalizes pixels specifically for shaurya7084/bg-removal-onnx (range 0.0 - 1.0)
 * Explicitly disables image smoothing to prevent high-DPI (Retina) scaling distortion on mobile.
 */
function prepareTensor(imgData: ImageData, size: number, runtime: OrtRuntime): ort.Tensor {
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = false;

  const tempCanvas = new OffscreenCanvas(imgData.width, imgData.height);
  const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true })!;
  tempCtx.imageSmoothingEnabled = false;
  tempCtx.putImageData(imgData, 0, 0);

  ctx.drawImage(tempCanvas, 0, 0, size, size);
  const resizedData = ctx.getImageData(0, 0, size, size).data;

  const float32Data = new Float32Array(3 * size * size);
  const imagePixelCount = size * size;

  for (let i = 0; i < imagePixelCount; i++) {
    float32Data[i] = resizedData[i * 4] / 255.0;
    float32Data[imagePixelCount + i] = resizedData[i * 4 + 1] / 255.0;
    float32Data[2 * imagePixelCount + i] = resizedData[i * 4 + 2] / 255.0;
  }

  return new runtime.Tensor('float32', float32Data, [1, 3, size, size]);
}

/**
 * Generates an RGBA Uint8ClampedArray mask with correct alpha scaling
 */
function postProcessMask(outputData: Float32Array, size: number): Uint8ClampedArray {
  const pixelCount = size * size;
  const rgba = new Uint8ClampedArray(pixelCount * 4);

  for (let i = 0; i < pixelCount; i++) {
    const val = Math.max(0, Math.min(255, Math.round(outputData[i] * 255)));
    const idx = i * 4;
    rgba[idx] = 255;
    rgba[idx + 1] = 255;
    rgba[idx + 2] = 255;
    rgba[idx + 3] = val;
  }

  return rgba;
}