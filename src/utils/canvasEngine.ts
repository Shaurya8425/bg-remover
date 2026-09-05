export type ProcessingMode = 'ai-photo' | 'signature-doc';
export type BackgroundMode = 'transparent' | 'color' | 'blur';

export class CanvasEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private originalImageData: ImageData | null = null;
  private processedCutoutData: ImageData | null = null;

  private currentBgMode: BackgroundMode = 'transparent';
  private currentBgColor: string = '#ffffff';

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('Could not acquire 2D canvas context.');
    this.ctx = context;
  }

  public setOriginalImage(imgData: ImageData) {
    this.originalImageData = imgData;
    this.processedCutoutData = null; // Reset cached cutout on new upload
    this.canvas.width = imgData.width;
    this.canvas.height = imgData.height;
    this.ctx.putImageData(imgData, 0, 0);
  }

  /**
   * Fast Guided Filter for client-side edge refinement.
   * Uses high-res color boundaries from original ImageData to guide low-res alpha masks.
   */
  private applyGuidedFilter(
    guide: Uint8ClampedArray,
    alpha: Float32Array,
    width: number,
    height: number,
    radius: number = 4,
    eps: number = 1e-4
  ): Float32Array {
    const N = width * height;
    const result = new Float32Array(N);

    // Convert guide image to grayscale luminance
    const I = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      I[i] = (0.299 * guide[i * 4] + 0.587 * guide[i * 4 + 1] + 0.114 * guide[i * 4 + 2]) / 255.0;
    }

    const p = alpha;

    // Box blur computation via integral images
    const boxFilter = (src: Float32Array): Float32Array => {
      const dst = new Float32Array(N);
      const stride = width + 1;
      const integral = new Float32Array((width + 1) * (height + 1));

      for (let y = 0; y < height; y++) {
        let lineSum = 0;
        for (let x = 0; x < width; x++) {
          lineSum += src[y * width + x];
          integral[(y + 1) * stride + (x + 1)] = integral[y * stride + (x + 1)] + lineSum;
        }
      }

      for (let y = 0; y < height; y++) {
        const y1 = Math.max(0, y - radius);
        const y2 = Math.min(height - 1, y + radius);
        for (let x = 0; x < width; x++) {
          const x1 = Math.max(0, x - radius);
          const x2 = Math.min(width - 1, x + radius);
          const count = (x2 - x1 + 1) * (y2 - y1 + 1);

          const sum =
            integral[(y2 + 1) * stride + (x2 + 1)] -
            integral[y1 * stride + (x2 + 1)] -
            integral[(y2 + 1) * stride + x1] +
            integral[y1 * stride + x1];

          dst[y * width + x] = sum / count;
        }
      }
      return dst;
    };

    const meanI = boxFilter(I);
    const meanP = boxFilter(p);

    const Ip = new Float32Array(N);
    const II = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      Ip[i] = I[i] * p[i];
      II[i] = I[i] * I[i];
    }

    const meanIp = boxFilter(Ip);
    const meanII = boxFilter(II);

    const varI = new Float32Array(N);
    const covIp = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      varI[i] = meanII[i] - meanI[i] * meanI[i];
      covIp[i] = meanIp[i] - meanI[i] * meanP[i];
    }

    const a = new Float32Array(N);
    const b = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      a[i] = covIp[i] / (varI[i] + eps);
      b[i] = meanP[i] - a[i] * meanI[i];
    }

    const meanA = boxFilter(a);
    const meanB = boxFilter(b);

    for (let i = 0; i < N; i++) {
      let q = meanA[i] * I[i] + meanB[i];
      
      // Guard against NaN values that cause missing pixel holes
      if (Number.isNaN(q)) {
        q = p[i];
      }
      
      result[i] = Math.max(0, Math.min(1, q));
    }

    return result;
  }

  /**
   * MODE 1: Refined AI Cutout Processing using Edge-Aware Guided Filtering
   */
  public processAndApplyMask(
    maskPixels: Uint8ClampedArray,
    maskWidth: number,
    maskHeight: number,
    edgeThreshold: number = 0.5,
    edgeSharpness: number = 4.0
  ) {
    if (!this.originalImageData) return;

    const origWidth = this.originalImageData.width;
    const origHeight = this.originalImageData.height;

    // 1. Scaled Mask Canvas to Match Full Resolution
    const maskCanvas = new OffscreenCanvas(maskWidth, maskHeight);
    const maskCtx = maskCanvas.getContext('2d')!;
    const maskImgData = maskCtx.createImageData(maskWidth, maskHeight);
    maskImgData.data.set(maskPixels);
    maskCtx.putImageData(maskImgData, 0, 0);

    // 2. High-Res Canvas Context Interpolation
    const scaledMaskCanvas = new OffscreenCanvas(origWidth, origHeight);
    const scaledMaskCtx = scaledMaskCanvas.getContext('2d')!;
    scaledMaskCtx.imageSmoothingEnabled = true;
    scaledMaskCtx.imageSmoothingQuality = 'high';
    scaledMaskCtx.drawImage(maskCanvas, 0, 0, origWidth, origHeight);

    const scaledMaskData = scaledMaskCtx.getImageData(0, 0, origWidth, origHeight);
    const maskBytes = scaledMaskData.data;

    // 3. Extract raw float alpha array from scaled mask
    const totalPixels = origWidth * origHeight;
    const rawAlphaArray = new Float32Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
      rawAlphaArray[i] = maskBytes[i * 4 + 3] / 255.0;
    }

    // 4. Run Guided Filter using original image RGB high-res details as guide
    const refinedAlpha = this.applyGuidedFilter(
      this.originalImageData.data,
      rawAlphaArray,
      origWidth,
      origHeight,
      3,    // Search radius
      1e-3  // Regularization parameter
    );

    // 5. Compositing
    const finalImageData = this.ctx.createImageData(origWidth, origHeight);
    const origBytes = this.originalImageData.data;
    const outputBytes = finalImageData.data;

    for (let i = 0; i < totalPixels; i++) {
      const idx = i * 4;

      outputBytes[idx] = origBytes[idx];         // Red
      outputBytes[idx + 1] = origBytes[idx + 1]; // Green
      outputBytes[idx + 2] = origBytes[idx + 2]; // Blue

      let alphaVal = refinedAlpha[i];

      if (Number.isNaN(alphaVal)) {
        alphaVal = rawAlphaArray[i];
      }

      // Soft thresholding to prevent punching holes inside the document body
      if (alphaVal < 0.02) {
        alphaVal = 0;
      } else if (alphaVal > 0.85) {
        alphaVal = 1;
      }

      outputBytes[idx + 3] = Math.round(alphaVal * 255);
    }

    // Cache processed foreground cutout
    this.processedCutoutData = finalImageData;
    this.render();
  }

  /**
   * Mode 2: Robust Adaptive Neighborhood Binarization
   * Works on low-contrast pencil/ink, uneven paper shadows, and dim lighting.
   */
  public extractSignatureOrText(
    windowSize: number = 25,     // Neighborhood window size in pixels
    offset: number = 8,          // Luminance difference required to identify ink
    preserveOriginalColor: boolean = true
  ) {
    if (!this.originalImageData) return;

    const width = this.originalImageData.width;
    const height = this.originalImageData.height;
    const d = this.originalImageData.data;

    const finalImageData = this.ctx.createImageData(width, height);
    const out = finalImageData.data;

    // 1. Convert image to grayscale Float32 array
    const gray = new Float32Array(width * height);
    for (let i = 0; i < d.length; i += 4) {
      gray[i / 4] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    }

    // 2. Build 2D Integral Image for fast O(1) area sums
    const stride = width + 1;
    const integral = new Float32Array((width + 1) * (height + 1));

    for (let y = 0; y < height; y++) {
      let lineSum = 0;
      for (let x = 0; x < width; x++) {
        lineSum += gray[y * width + x];
        integral[(y + 1) * stride + (x + 1)] =
          integral[y * stride + (x + 1)] + lineSum;
      }
    }

    const halfWin = Math.floor(windowSize / 2);

    // 3. Evaluate each pixel relative to its surrounding paper context
    for (let y = 0; y < height; y++) {
      const y1 = Math.max(0, y - halfWin);
      const y2 = Math.min(height - 1, y + halfWin);

      for (let x = 0; x < width; x++) {
        const x1 = Math.max(0, x - halfWin);
        const x2 = Math.min(width - 1, x + halfWin);

        const area = (x2 - x1 + 1) * (y2 - y1 + 1);

        // Fast sum query from integral table
        const sum =
          integral[(y2 + 1) * stride + (x2 + 1)] -
          integral[y1 * stride + (x2 + 1)] -
          integral[(y2 + 1) * stride + x1] +
          integral[y1 * stride + x1];

        const localMean = sum / area;
        const pixelIdx = y * width + x;
        const idx = pixelIdx * 4;
        const currentVal = gray[pixelIdx];

        // Pixel is ink if it is darker than the surrounding paper average by the offset
        if (localMean - currentVal > offset) {
          if (preserveOriginalColor) {
            out[idx] = d[idx];         // Red
            out[idx + 1] = d[idx + 1]; // Green
            out[idx + 2] = d[idx + 2]; // Blue
          } else {
            out[idx] = 20;
            out[idx + 1] = 20;
            out[idx + 2] = 20;
          }
          out[idx + 3] = 255; // Keep Ink Stroke
        } else {
          out[idx + 3] = 0;   // Transparent Background
        }
      }
    }

    this.processedCutoutData = finalImageData;
    this.render();
  }

  /**
   * Background Layer Switcher (Transparent / Solid Color)
   */
  public setBackground(mode: BackgroundMode, color: string = '#ffffff') {
    this.currentBgMode = mode;
    this.currentBgColor = color;
    this.render();
  }

  /**
   * Final Canvas Composition Renderer
   */
  public render() {
    if (!this.processedCutoutData) return;

    const width = this.canvas.width;
    const height = this.canvas.height;

    this.ctx.clearRect(0, 0, width, height);

    // Render background layer
    if (this.currentBgMode === 'color') {
      this.ctx.fillStyle = this.currentBgColor;
      this.ctx.fillRect(0, 0, width, height);
    }

    // Composite foreground cutout over background
    const tempCanvas = new OffscreenCanvas(width, height);
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.putImageData(this.processedCutoutData, 0, 0);

    this.ctx.drawImage(tempCanvas, 0, 0);
  }

  public exportPNG(): string {
    return this.canvas.toDataURL('image/png', 1.0);
  }
}