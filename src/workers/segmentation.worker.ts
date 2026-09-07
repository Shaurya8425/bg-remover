import * as ort from 'onnxruntime-web';
import wasmBinaryUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';
import { ModelCache } from '../utils/modelCache';

ort.env.wasm.wasmPaths = { wasm: wasmBinaryUrl };
ort.env.wasm.numThreads = 1;

let session: ort.InferenceSession | null = null;
let isBusy = false; // Flag to prevent concurrent execution attempts
const cache = new ModelCache();

export interface WorkerMessageData {
  type: 'PROCESS_IMAGE';
  imageData: ImageData;
}

self.onmessage = async (event: MessageEvent<WorkerMessageData>) => {
  const { type, imageData } = event.data;

  if (type === 'PROCESS_IMAGE') {
    // Prevent starting a new inference if session is already processing
    if (isBusy) {
      self.postMessage({
        type: 'ERROR',
        error: 'Inference is already in progress. Please wait.',
      });
      return;
    }

    isBusy = true;

    try {
      // 1. Lazy Session Initialization
      if (!session) {
        const modelUrl = 'https://huggingface.co/shaurya7084/bg-removal-onnx/resolve/main/model_quantized.onnx';
        let modelBuffer: ArrayBuffer | null = await cache.getAsset(modelUrl);

        if (!modelBuffer) {
          self.postMessage({ type: 'PROGRESS', progress: 10, key: 'Downloading Model' });
          const response = await fetch(modelUrl);
          if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
          modelBuffer = await response.arrayBuffer();
          await cache.setAsset(modelUrl, modelBuffer);
        }

        self.postMessage({ type: 'PROGRESS', progress: 35, key: 'Initializing ONNX Engine' });

        try {
          session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['webgpu', 'wasm'],
          });
        } catch {
          session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
          });
        }
      }

      self.postMessage({ type: 'PROGRESS', progress: 50, key: 'Processing Tensor' });

      // 2. Pre-process Image
      const targetSize = 1024;
      const tensor = prepareTensor(imageData, targetSize);

      // 3. Execute Model safely
      const inputName = session.inputNames[0];
      const feeds: Record<string, ort.Tensor> = { [inputName]: tensor };
      const results = await session.run(feeds);

      self.postMessage({ type: 'PROGRESS', progress: 85, key: 'Refining Mask' });

      // 4. Extract Output Data
      const outputName = session.outputNames[0];
      const outputTensor = results[outputName];
      const rawFloatData = outputTensor.data as Float32Array;

      if (!rawFloatData) throw new Error('Failed to extract tensor output array.');

      const maskPixels = postProcessMask(rawFloatData, targetSize);

      self.postMessage({
        type: 'SUCCESS',
        maskPixels,
        maskWidth: targetSize,
        maskHeight: targetSize,
      });

    } catch (err: any) {
      self.postMessage({
        type: 'ERROR',
        error: err?.message || 'Inference execution failed.',
      });
    } finally {
      isBusy = false; // Always release worker lock
    }
  }
};

function prepareTensor(imgData: ImageData, size: number): ort.Tensor {
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d')!;

  const tempCanvas = new OffscreenCanvas(imgData.width, imgData.height);
  const tempCtx = tempCanvas.getContext('2d')!;
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

  return new ort.Tensor('float32', float32Data, [1, 3, size, size]);
}

function postProcessMask(outputData: Float32Array, size: number): Uint8ClampedArray {
  const pixelCount = size * size;
  const rgba = new Uint8ClampedArray(pixelCount * 4);

  for (let i = 0; i < pixelCount; i++) {
    const val = Math.round(outputData[i] * 255);
    rgba[i * 4 + 3] = val;
  }

  return rgba;
}