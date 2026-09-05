import * as ort from 'onnxruntime-web';
import wasmBinaryUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';

ort.env.wasm.wasmPaths = { wasm: wasmBinaryUrl };
ort.env.wasm.numThreads = 1;

let session: ort.InferenceSession | null = null;

export interface WorkerMessageData {
  type: 'PROCESS_IMAGE';
  imageData: ImageData;
}

self.onmessage = async (event: MessageEvent<WorkerMessageData>) => {
  const { type, imageData } = event.data;

  if (type === 'PROCESS_IMAGE') {
    try {
      // 1. Lazy Session Initialization
      if (!session) {
        self.postMessage({ type: 'PROGRESS', progress: 20, key: 'Loading Model' });
        
        // Single-file ONNX path
        const modelPath = '/models/model_quantized.onnx';
        
        try {
          session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['webgpu', 'wasm'],
          });
        } catch {
          session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['wasm'],
          });
        }
      }

      self.postMessage({ type: 'PROGRESS', progress: 50, key: 'Processing Tensor' });

      // 2. Pre-process Image to 1024x1024 NCHW Tensor
      const targetSize = 1024;
      const tensor = prepareTensor(imageData, targetSize);

      // 3. Execute Model
      const inputName = session.inputNames[0];
      const feeds: Record<string, ort.Tensor> = { [inputName]: tensor };
      const results = await session.run(feeds);

      self.postMessage({ type: 'PROGRESS', progress: 85, key: 'Refining Mask' });

      // 4. Safely Extract Output Data (.data property, not .getValue())
      const outputName = session.outputNames[0];
      const outputTensor = results[outputName];

      // Access raw float array directly via .data
      const rawFloatData = outputTensor.data as Float32Array;

      if (!rawFloatData) {
        throw new Error('Failed to extract tensor output array.');
      }

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
    }
  }
};

/**
 * Normalizes ImageData to a [1, 3, 1024, 1024] Float32 NCHW Tensor
 */
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
    float32Data[i] = resizedData[i * 4] / 255.0;                         // Red
    float32Data[imagePixelCount + i] = resizedData[i * 4 + 1] / 255.0;    // Green
    float32Data[2 * imagePixelCount + i] = resizedData[i * 4 + 2] / 255.0;// Blue
  }

  return new ort.Tensor('float32', float32Data, [1, 3, size, size]);
}

/**
 * Converts float probability tensor output to RGBA bytes
 */
function postProcessMask(outputData: Float32Array, size: number): Uint8ClampedArray {
  const pixelCount = size * size;
  const rgba = new Uint8ClampedArray(pixelCount * 4);

  for (let i = 0; i < pixelCount; i++) {
    const val = Math.round(outputData[i] * 255);
    rgba[i * 4] = 0;
    rgba[i * 4 + 1] = 0;
    rgba[i * 4 + 2] = 0;
    rgba[i * 4 + 3] = val; // Set Alpha Channel
  }

  return rgba;
}