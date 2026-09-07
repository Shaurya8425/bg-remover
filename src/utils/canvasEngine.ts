export type ProcessingMode = "ai-photo" | "signature-doc";
export type BackgroundMode = "transparent" | "color" | "blur";

interface CachedAiMask {
  maskPixels: Uint8ClampedArray;
  maskWidth: number;
  maskHeight: number;
}

export class CanvasEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private originalImageData: ImageData | null = null;

  // Cached cutouts for instant mode switching
  private aiCutoutData: ImageData | null = null;
  private signatureCutoutData: ImageData | null = null;
  private cachedRawAiMask: CachedAiMask | null = null;

  private currentCutoutData: ImageData | null = null;
  private currentMode: ProcessingMode = "ai-photo";
  private currentBgMode: BackgroundMode = "transparent";
  private currentBgColor: string = "#ffffff";

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Could not acquire 2D canvas context.");
    this.ctx = context;
  }

  public setOriginalImage(imgData: ImageData) {
    this.originalImageData = imgData;

    // Clear all mode caches when a new image is uploaded
    this.aiCutoutData = null;
    this.signatureCutoutData = null;
    this.cachedRawAiMask = null;
    this.currentCutoutData = null;

    this.canvas.width = imgData.width;
    this.canvas.height = imgData.height;
    this.ctx.putImageData(imgData, 0, 0);
  }

  /**
   * Check if a mode already has a rendered cutout cached.
   */
  public hasCachedMode(mode: ProcessingMode): boolean {
    if (mode === "ai-photo") return this.aiCutoutData !== null;
    if (mode === "signature-doc") return this.signatureCutoutData !== null;
    return false;
  }

  /**
   * Instant mode switch using cached cutouts (no worker/re-processing required).
   */
  public switchMode(mode: ProcessingMode): boolean {
    this.currentMode = mode;
    if (mode === "ai-photo" && this.aiCutoutData) {
      this.currentCutoutData = this.aiCutoutData;
      this.render();
      return true;
    }
    if (mode === "signature-doc" && this.signatureCutoutData) {
      this.currentCutoutData = this.signatureCutoutData;
      this.render();
      return true;
    }
    return false;
  }

  /**
   * Generic box blur via integral image.
   */
  private boxFilter(
    src: Float32Array,
    width: number,
    height: number,
    radius: number,
  ): Float32Array {
    const N = width * height;
    const dst = new Float32Array(N);
    const stride = width + 1;
    const integral = new Float32Array((width + 1) * (height + 1));

    for (let y = 0; y < height; y++) {
      let lineSum = 0;
      for (let x = 0; x < width; x++) {
        lineSum += src[y * width + x];
        integral[(y + 1) * stride + (x + 1)] =
          integral[y * stride + (x + 1)] + lineSum;
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
  }

  /**
   * Fast Guided Filter for client-side edge refinement.
   */
  private applyGuidedFilter(
    guide: Uint8ClampedArray,
    alpha: Float32Array,
    width: number,
    height: number,
    radius: number = 6,
    eps: number = 2e-3,
  ): Float32Array {
    const N = width * height;
    const result = new Float32Array(N);

    const I = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      I[i] =
        (0.299 * guide[i * 4] +
          0.587 * guide[i * 4 + 1] +
          0.114 * guide[i * 4 + 2]) /
        255.0;
    }

    const p = alpha;
    const meanI = this.boxFilter(I, width, height, radius);
    const meanP = this.boxFilter(p, width, height, radius);

    const Ip = new Float32Array(N);
    const II = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      Ip[i] = I[i] * p[i];
      II[i] = I[i] * I[i];
    }

    const meanIp = this.boxFilter(Ip, width, height, radius);
    const meanII = this.boxFilter(II, width, height, radius);

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

    const meanA = this.boxFilter(a, width, height, radius);
    const meanB = this.boxFilter(b, width, height, radius);

    for (let i = 0; i < N; i++) {
      let q = meanA[i] * I[i] + meanB[i];
      if (Number.isNaN(q)) q = p[i];
      result[i] = Math.max(0, Math.min(1, q));
    }

    return result;
  }

  /**
   * MODE 1: Refined AI Cutout Processing matched to shaurya7084/bg-removal-onnx
   */
  public processAndApplyMask(
    maskPixels: Uint8ClampedArray,
    maskWidth: number,
    maskHeight: number,
    edgeThreshold: number = 0.5,
    edgeSharpness: number = 1.2, // Reduced from 4.0 to prevent hard contrast clipping
    filterRadius: number = 3, // Reduced from 6 for localized edge snapping
    filterEps: number = 1e-2, // Increased from 2e-3 to trust raw model predictions more
  ) {
    if (!this.originalImageData) return;

    // Cache raw prediction mask
    this.cachedRawAiMask = { maskPixels, maskWidth, maskHeight };

    const origWidth = this.originalImageData.width;
    const origHeight = this.originalImageData.height;

    // 1. Rescale mask to canvas dimensions
    const maskCanvas = new OffscreenCanvas(maskWidth, maskHeight);
    const maskCtx = maskCanvas.getContext("2d")!;
    const maskImgData = maskCtx.createImageData(maskWidth, maskHeight);
    maskImgData.data.set(maskPixels);
    maskCtx.putImageData(maskImgData, 0, 0);

    const scaledMaskCanvas = new OffscreenCanvas(origWidth, origHeight);
    const scaledMaskCtx = scaledMaskCanvas.getContext("2d")!;
    scaledMaskCtx.imageSmoothingEnabled = true;
    scaledMaskCtx.imageSmoothingQuality = "high";
    scaledMaskCtx.drawImage(maskCanvas, 0, 0, origWidth, origHeight);

    const scaledMaskData = scaledMaskCtx.getImageData(
      0,
      0,
      origWidth,
      origHeight,
    );
    const maskBytes = scaledMaskData.data;

    const totalPixels = origWidth * origHeight;
    const rawAlphaArray = new Float32Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
      rawAlphaArray[i] = maskBytes[i * 4 + 3] / 255.0;
    }

    // 2. Apply Guided Filter with refined neighborhood parameters
    const refinedAlpha = this.applyGuidedFilter(
      this.originalImageData.data,
      rawAlphaArray,
      origWidth,
      origHeight,
      filterRadius,
      filterEps,
    );

    // 3. Smoothstep alpha mapping
    const halfBand = 1 / (edgeSharpness * 2);
    const lower = Math.max(0, edgeThreshold - halfBand);
    const upper = Math.min(1, edgeThreshold + halfBand);

    const smoothstep = (edge0: number, edge1: number, x: number): number => {
      if (edge1 <= edge0) return x < edge0 ? 0 : 1;
      const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
      return t * t * (3 - 2 * t);
    };

    const shapedAlpha = new Float32Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
      let alphaVal = refinedAlpha[i];
      if (Number.isNaN(alphaVal)) alphaVal = rawAlphaArray[i];

      if (alphaVal <= lower) shapedAlpha[i] = 0;
      else if (alphaVal >= upper) shapedAlpha[i] = 1;
      else shapedAlpha[i] = smoothstep(lower, upper, alphaVal);
    }

    // 4. Subtle box filter feathering
    const featheredAlpha = this.boxFilter(
      shapedAlpha,
      origWidth,
      origHeight,
      1,
    );

    const finalImageData = this.ctx.createImageData(origWidth, origHeight);
    const origBytes = this.originalImageData.data;
    const outputBytes = finalImageData.data;

    const OPAQUE_THRESHOLD = 0.98;
    const DECONTAM_WINDOW = 2;

    for (let y = 0; y < origHeight; y++) {
      for (let x = 0; x < origWidth; x++) {
        const i = y * origWidth + x;
        const idx = i * 4;
        const a = Math.max(0, Math.min(1, featheredAlpha[i]));

        if (a <= 0) {
          outputBytes[idx + 3] = 0;
          continue;
        }

        if (a >= OPAQUE_THRESHOLD) {
          outputBytes[idx] = origBytes[idx];
          outputBytes[idx + 1] = origBytes[idx + 1];
          outputBytes[idx + 2] = origBytes[idx + 2];
          outputBytes[idx + 3] = 255;
          continue;
        }

        // Color decontamination for semi-transparent border pixels
        let rSum = 0,
          gSum = 0,
          bSum = 0,
          count = 0;
        for (let dy = -DECONTAM_WINDOW; dy <= DECONTAM_WINDOW; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= origHeight) continue;
          for (let dx = -DECONTAM_WINDOW; dx <= DECONTAM_WINDOW; dx++) {
            const nx = x + dx;
            if (nx < 0 || nx >= origWidth) continue;
            const ni = ny * origWidth + nx;
            if (featheredAlpha[ni] >= OPAQUE_THRESHOLD) {
              const nidx = ni * 4;
              rSum += origBytes[nidx];
              gSum += origBytes[nidx + 1];
              bSum += origBytes[nidx + 2];
              count++;
            }
          }
        }

        if (count > 0) {
          const fgR = rSum / count;
          const fgG = gSum / count;
          const fgB = bSum / count;
          outputBytes[idx] = origBytes[idx] * a + fgR * (1 - a);
          outputBytes[idx + 1] = origBytes[idx + 1] * a + fgG * (1 - a);
          outputBytes[idx + 2] = origBytes[idx + 2] * a + fgB * (1 - a);
        } else {
          outputBytes[idx] = origBytes[idx];
          outputBytes[idx + 1] = origBytes[idx + 1];
          outputBytes[idx + 2] = origBytes[idx + 2];
        }

        outputBytes[idx + 3] = Math.round(a * 255);
      }
    }

    this.aiCutoutData = finalImageData;
    this.currentCutoutData = finalImageData;
    this.currentMode = "ai-photo";
    this.render();
  }

  /**
   * Mode 2: Robust Adaptive Neighborhood Binarization
   */
  public extractSignatureOrText(
    windowSize: number = 25,
    offset: number = 8,
    preserveOriginalColor: boolean = true,
  ) {
    if (!this.originalImageData) return;

    const width = this.originalImageData.width;
    const height = this.originalImageData.height;
    const d = this.originalImageData.data;

    const finalImageData = this.ctx.createImageData(width, height);
    const out = finalImageData.data;

    const gray = new Float32Array(width * height);
    for (let i = 0; i < d.length; i += 4) {
      gray[i / 4] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    }

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

    for (let y = 0; y < height; y++) {
      const y1 = Math.max(0, y - halfWin);
      const y2 = Math.min(height - 1, y + halfWin);

      for (let x = 0; x < width; x++) {
        const x1 = Math.max(0, x - halfWin);
        const x2 = Math.min(width - 1, x + halfWin);

        const area = (x2 - x1 + 1) * (y2 - y1 + 1);

        const sum =
          integral[(y2 + 1) * stride + (x2 + 1)] -
          integral[y1 * stride + (x2 + 1)] -
          integral[(y2 + 1) * stride + x1] +
          integral[y1 * stride + x1];

        const localMean = sum / area;
        const pixelIdx = y * width + x;
        const idx = pixelIdx * 4;
        const currentVal = gray[pixelIdx];

        if (localMean - currentVal > offset) {
          if (preserveOriginalColor) {
            out[idx] = d[idx];
            out[idx + 1] = d[idx + 1];
            out[idx + 2] = d[idx + 2];
          } else {
            out[idx] = 20;
            out[idx + 1] = 20;
            out[idx + 2] = 20;
          }
          out[idx + 3] = 255;
        } else {
          out[idx + 3] = 0;
        }
      }
    }

    // Store in Signature cutout cache & set current
    this.signatureCutoutData = finalImageData;
    this.currentCutoutData = finalImageData;
    this.currentMode = "signature-doc";
    this.render();
  }

  /**
   * Background Layer Switcher (Transparent / Solid Color)
   */
  public setBackground(mode: BackgroundMode, color: string = "#ffffff") {
    this.currentBgMode = mode;
    this.currentBgColor = color;
    this.render();
  }

  /**
   * Final Canvas Composition Renderer
   */
  public render() {
    if (!this.currentCutoutData) return;

    const width = this.canvas.width;
    const height = this.canvas.height;

    this.ctx.clearRect(0, 0, width, height);

    if (this.currentBgMode === "color") {
      this.ctx.fillStyle = this.currentBgColor;
      this.ctx.fillRect(0, 0, width, height);
    }

    const tempCanvas = new OffscreenCanvas(width, height);
    const tempCtx = tempCanvas.getContext("2d")!;
    tempCtx.putImageData(this.currentCutoutData, 0, 0);

    this.ctx.drawImage(tempCanvas, 0, 0);
  }

  public exportPNG(): string {
    return this.canvas.toDataURL("image/png", 1.0);
  }
}
