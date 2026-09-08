import { CanvasEngine, type ProcessingMode, type BackgroundMode } from "./canvasEngine";

export interface WorkspaceOptions {
  defaultBgMode?: BackgroundMode;
  defaultColor?: string;
  defaultProcMode?: ProcessingMode;
}

export function initWorkspace(options: WorkspaceOptions = {}) {
  const {
    defaultBgMode = 'transparent',
    defaultColor = '#ffffff',
    defaultProcMode = 'ai-photo',
  } = options;

  const dropZone = document.getElementById("drop-zone");
  const dropOverlay = document.getElementById("drop-overlay");
  const fileInput = document.getElementById("file-input") as HTMLInputElement | null;
  const browseBtn = document.getElementById("browse-btn");
  const replaceBtn = document.getElementById("replace-btn");
  const resetBtn = document.getElementById("reset-btn");

  const uploadState = document.getElementById("upload-state");
  const processingState = document.getElementById("processing-state");
  const canvasContainer = document.getElementById("canvas-container");
  const progressText = document.getElementById("progress-text");
  const outputCanvas = document.getElementById("output-canvas") as HTMLCanvasElement | null;
  const downloadBtn = document.getElementById("download-btn") as HTMLButtonElement | null;

  const colorPickerGroup = document.getElementById("color-picker-group");
  const bgColorPicker = document.getElementById("bg-color-picker") as HTMLInputElement | null;
  const colorSwatches = document.querySelectorAll<HTMLButtonElement>(".color-swatch");

  const modeBtns = document.querySelectorAll<HTMLButtonElement>(".bg-mode-btn");
  const procBtns = document.querySelectorAll<HTMLButtonElement>(".proc-mode-btn");

  let canvasEngine: CanvasEngine | null = null;
  let worker: Worker | null = null;
  let currentImageData: ImageData | null = null;
  let currentFile: File | null = null;
  let currentProcessingMode: ProcessingMode = defaultProcMode;
  let isProcessingWorker = false;

  if (outputCanvas) {
    canvasEngine = new CanvasEngine(outputCanvas);
  }

  function initWorker() {
    if (worker) return;

    worker = new Worker(
      new URL("../workers/segmentation.worker.ts", import.meta.url),
      { type: "module" }
    );

    worker.onmessage = (e) => {
      const { type, progress, maskPixels, maskWidth, maskHeight, error, key } = e.data;

      if (type === "PROGRESS") {
        if (progressText) progressText.innerText = `${key}... ${progress}%`;
      } else if (type === "SUCCESS" && maskPixels) {
        isProcessingWorker = false;
        canvasEngine?.processAndApplyMask(maskPixels, maskWidth, maskHeight);
        finishProcessingUI();
      } else if (type === "ERROR") {
        isProcessingWorker = false;
        alert(`Error: ${error}`);
        resetUI();
      }
    };
  }

  // File Triggers
  browseBtn?.addEventListener("click", () => fileInput?.click());
  replaceBtn?.addEventListener("click", () => fileInput?.click());

  fileInput?.addEventListener("change", (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (files?.length) processFile(files[0]);
  });

  // Drag & Drop
  ['dragenter', 'dragover'].forEach((eventName) => {
    dropZone?.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropOverlay?.classList.remove("opacity-0", "pointer-events-none");
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropZone?.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropOverlay?.classList.add("opacity-0", "pointer-events-none");
    });
  });

  dropZone?.addEventListener("drop", (e) => {
    if (e.dataTransfer?.files.length) processFile(e.dataTransfer.files[0]);
  });

  resetBtn?.addEventListener("click", resetUI);

  function processFile(file: File) {
    if (!file.type.startsWith("image/")) return;
    currentFile = file;

    uploadState?.classList.add("hidden");
    canvasContainer?.classList.add("hidden");
    processingState?.classList.remove("hidden");
    processingState?.classList.add("flex");

    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      const offscreen = document.createElement("canvas");
      offscreen.width = img.naturalWidth;
      offscreen.height = img.naturalHeight;
      const ctx = offscreen.getContext("2d", { willReadFrequently: true })!;
      ctx.drawImage(img, 0, 0);

      currentImageData = ctx.getImageData(0, 0, img.naturalWidth, img.naturalHeight);
      canvasEngine?.setOriginalImage(currentImageData);

      if (fileInput) fileInput.value = "";
      URL.revokeObjectURL(url);

      executeProcessingPipeline();
    };

    img.src = url;
  }

  function executeProcessingPipeline() {
    if (!currentImageData) return;

    if (currentProcessingMode === 'signature-doc') {
      if (progressText) progressText.innerText = 'Extracting Ink...';
      setTimeout(() => {
        canvasEngine?.extractSignatureOrText(25, 8, true);
        finishProcessingUI();
      }, 50);
    } else {
      if (isProcessingWorker) return;

      initWorker();
      isProcessingWorker = true;
      if (progressText) progressText.innerText = 'Initializing AI Model...';

      worker?.postMessage({
        type: "PROCESS_IMAGE",
        imageData: currentImageData,
      });
    }
  }

  function finishProcessingUI() {
    processingState?.classList.add("hidden");
    processingState?.classList.remove("flex");
    canvasContainer?.classList.remove("hidden");
    canvasContainer?.classList.add("flex");

    if (downloadBtn) {
      downloadBtn.disabled = false;
      downloadBtn.classList.remove("bg-slate-800", "text-slate-500", "cursor-not-allowed");
      downloadBtn.classList.add("bg-indigo-600", "hover:bg-indigo-500", "text-white", "shadow-lg", "shadow-indigo-600/20");
    }

    // Apply default background preset configurations on complete
    canvasEngine?.setBackground(defaultBgMode, defaultColor);
  }

  // Processing mode toggle
  procBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (isProcessingWorker) return;

      const newMode = btn.dataset.procMode as ProcessingMode;
      if (newMode === currentProcessingMode) return;

      procBtns.forEach((b) => {
        b.classList.remove("border-indigo-500/50", "bg-indigo-600/10", "text-indigo-300");
        b.classList.add("border-transparent", "text-slate-400");
      });

      btn.classList.remove("border-transparent", "text-slate-400");
      btn.classList.add("border-indigo-500/50", "bg-indigo-600/10", "text-indigo-300");

      currentProcessingMode = newMode;

      if (currentImageData) {
        if (canvasEngine?.hasCachedMode(currentProcessingMode)) {
          canvasEngine.switchMode(currentProcessingMode);
          finishProcessingUI();
          return;
        }

        processingState?.classList.remove("hidden");
        processingState?.classList.add("flex");
        canvasContainer?.classList.add("hidden");
        executeProcessingPipeline();
      }
    });
  });

  // Background style selection
  modeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      modeBtns.forEach((b) => {
        b.classList.remove("border-indigo-500", "bg-indigo-500/10", "text-indigo-300");
        b.classList.add("border-slate-800", "bg-slate-950", "text-slate-400");
      });

      btn.classList.remove("border-slate-800", "bg-slate-950", "text-slate-400");
      btn.classList.add("border-indigo-500", "bg-indigo-500/10", "text-indigo-300");

      const mode = btn.dataset.mode as BackgroundMode;
      colorPickerGroup?.classList.toggle("hidden", mode !== "color");
      canvasEngine?.setBackground(mode, bgColorPicker?.value || "#ffffff");
    });
  });

  // Color Swatch Selection
  colorSwatches.forEach((swatch) => {
    swatch.addEventListener("click", () => {
      const color = swatch.dataset.color || "#ffffff";
      if (bgColorPicker) bgColorPicker.value = color;
      canvasEngine?.setBackground("color", color);
    });
  });

  bgColorPicker?.addEventListener("input", (e) => {
    canvasEngine?.setBackground("color", (e.target as HTMLInputElement).value);
  });

  downloadBtn?.addEventListener("click", () => {
    if (!canvasEngine || !currentFile) return;
    const dataUrl = canvasEngine.exportPNG();
    const link = document.createElement("a");

    const fileNameParts = currentFile.name.split(".");
    fileNameParts.pop();
    link.download = `${fileNameParts.join(".")}-bg-removed.png`;

    link.href = dataUrl;
    link.click();
  });

  function resetUI() {
    currentImageData = null;
    currentFile = null;
    isProcessingWorker = false;
    if (fileInput) fileInput.value = "";

    uploadState?.classList.remove("hidden");
    processingState?.classList.add("hidden");
    canvasContainer?.classList.add("hidden");

    if (downloadBtn) {
      downloadBtn.disabled = true;
      downloadBtn.classList.add("bg-slate-800", "text-slate-500", "cursor-not-allowed");
      downloadBtn.classList.remove("bg-indigo-600", "hover:bg-indigo-500", "text-white", "shadow-lg");
    }
  }
}