import { removeBackground, type Config } from '@imgly/background-removal';
import { ModelCache } from '../utils/modelCache';

const cache = new ModelCache();

export interface WorkerMessageData {
  type: 'PROCESS_IMAGE';
  imageBlob: Blob;
}

export interface WorkerResponseData {
  type: 'PROGRESS' | 'SUCCESS' | 'ERROR';
  progress?: number;
  key?: string;
  resultBlob?: Blob;
  error?: string;
}

// Custom cached fetch wrapper
async function cachedFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const urlString = input.toString();

  // 1. Check IndexedDB cache first
  const cachedBuffer = await cache.getAsset(urlString);
  if (cachedBuffer) {
    return new Response(cachedBuffer, {
      status: 200,
      headers: { 'Content-Type': 'application/octet-stream' },
    });
  }

  // 2. Fetch from network and persist to IndexedDB
  const response = await fetch(input, init);
  if (response.ok) {
    const buffer = await response.clone().arrayBuffer();
    await cache.setAsset(urlString, buffer);
  }
  return response;
}

self.onmessage = async (event: MessageEvent<WorkerMessageData>) => {
  const { type, imageBlob } = event.data;

  if (type === 'PROCESS_IMAGE') {
    try {
      const config: Config = {
        publicPath: 'https://staticimgly.com/@imgly/background-removal-data/1.5.8/dist/',
        
        // Pass custom fetch implementation inside fetchArgs
        fetchArgs: {
          mode: 'cors',
          fetch: cachedFetch as typeof fetch,
        },

        progress: (key: string, current: number, total: number) => {
          const percentage = Math.round((current / total) * 100);
          self.postMessage({
            type: 'PROGRESS',
            key,
            progress: isNaN(percentage) ? 0 : percentage,
          } as WorkerResponseData);
        },
        output: {
          format: 'image/png',
          quality: 1.0,
        },
      };

      const resultBlob = await removeBackground(imageBlob, config);

      self.postMessage({
        type: 'SUCCESS',
        resultBlob,
      } as WorkerResponseData);
    } catch (err: any) {
      self.postMessage({
        type: 'ERROR',
        error: err?.message || 'Background removal failed.',
      } as WorkerResponseData);
    }
  }
};