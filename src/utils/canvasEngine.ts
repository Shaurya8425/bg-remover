export type BackgroundMode = 'transparent' | 'color' | 'blur' | 'image';

export class CanvasEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private originalImage: HTMLImageElement | null = null;
  private foregroundMask: HTMLImageElement | null = null;
  private bgImage: HTMLImageElement | null = null;

  private currentMode: BackgroundMode = 'transparent';
  private currentColor: string = '#ffffff';
  private currentBlurRadius: number = 8;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Could not get 2D canvas context');
    this.ctx = context;
  }

  public setOriginalImage(img: HTMLImageElement) {
    this.originalImage = img;
    this.canvas.width = img.naturalWidth;
    this.canvas.height = img.naturalHeight;
  }

  public setForegroundMask(maskImg: HTMLImageElement) {
    this.foregroundMask = maskImg;
    this.render();
  }

  public setCustomBgImage(bgImg: HTMLImageElement) {
    this.bgImage = bgImg;
    if (this.currentMode === 'image') this.render();
  }

  public updateBackground(mode: BackgroundMode, color?: string, blurRadius?: number) {
    this.currentMode = mode;
    if (color !== undefined) this.currentColor = color;
    if (blurRadius !== undefined) this.currentBlurRadius = blurRadius;
    this.render();
  }

  public render() {
    if (!this.originalImage || !this.foregroundMask) return;

    const width = this.canvas.width;
    const height = this.canvas.height;

    // Clear canvas
    this.ctx.clearRect(0, 0, width, height);

    // Step 1: Render Selected Background
    this.ctx.save();
    switch (this.currentMode) {
      case 'transparent':
        // Left empty for alpha transparency
        break;

      case 'color':
        this.ctx.fillStyle = this.currentColor;
        this.ctx.fillRect(0, 0, width, height);
        break;

      case 'blur':
        // Render blurred copy of original image as background
        this.ctx.filter = `blur(${this.currentBlurRadius}px)`;
        this.ctx.drawImage(this.originalImage, -20, -20, width + 40, height + 40);
        this.ctx.filter = 'none';
        break;

      case 'image':
        if (this.bgImage) {
          // Cover crop standard image fit
          const hRatio = width / this.bgImage.naturalWidth;
          const vRatio = height / this.bgImage.naturalHeight;
          const ratio = Math.max(hRatio, vRatio);
          const centerShiftX = (width - this.bgImage.naturalWidth * ratio) / 2;
          const centerShiftY = (height - this.bgImage.naturalHeight * ratio) / 2;

          this.ctx.drawImage(
            this.bgImage,
            0, 0, this.bgImage.naturalWidth, this.bgImage.naturalHeight,
            centerShiftX, centerShiftY, this.bgImage.naturalWidth * ratio, this.bgImage.naturalHeight * ratio
          );
        }
        break;
    }
    this.ctx.restore();

    // Step 2: Overlay Processed Foreground Cutout
    this.ctx.drawImage(this.foregroundMask, 0, 0, width, height);
  }

  public exportPNG(): string {
    return this.canvas.toDataURL('image/png', 1.0);
  }
}