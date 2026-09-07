# Local AI Background Remover

A 100% private, client-side web application built with Astro, ONNX Runtime Web, and HTML5 Canvas. Remove image backgrounds or extract signatures and text instantly in your browser without uploading your files to any external server.

---

## Features

* **100% Private & Client-Side**: All machine learning models and algorithms run locally via WebAssembly and WebGPU. Your photos never leave your device.
* **Dual Processing Engine**:
  * **AI Photo Mode**: Uses a quantized segmentation model in a dedicated Web Worker via ONNX Runtime Web.
  * **Signature / Text Mode**: Uses fast local binarization algorithms to extract ink signatures or document text cleanly.
* **Live Canvas Controls**:
  * Seamlessly switch background modes (Transparent, Solid Color, or Custom Hex Picker).
  * Fast mode-switching leveraging internal Canvas caching, preventing re-triggering model inference.
* **Modern UI & Drag-and-Drop**: Built using Tailwind CSS, featuring active worker progress updates, drop overlay states, and responsive previews.
* **High-Res PNG Export**: Retains original image resolution upon export.

---

## Tech Stack

* **Framework**: Astro
* **ML Inference Engine**: ONNX Runtime Web (`onnxruntime-web` with WebGPU and WASM fallback)
* **Styling**: Tailwind CSS
* **Threading**: Web Workers (Vite worker module loader)
* **Language**: TypeScript

---

## Project Structure

```text
├── src/
│   ├── layouts/
│   │   └── Layout.astro         # Base layout wrapper
│   ├── pages/
│   │   └── index.astro          # Main workspace UI & interaction script
│   ├── styles/
│   │   └── global.css           # Global Tailwind CSS styles
│   ├── utils/
│   │   ├── canvasEngine.ts      # Canvas render engine & mode caching logic
│   │   └── modelCache.ts        # IndexedDB/CacheStorage asset loader
│   └── workers/
│       └── segmentation.worker.ts # ONNX model inference worker script
├── public/                      # Static assets
└── package.json
```

---

## Getting Started

### Prerequisites

Ensure you have Node.js (v18 or higher recommended) and npm, pnpm, or yarn installed.

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/bg-remover.git](https://github.com/your-username/bg-remover.git)
   cd bg-remover
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open in browser:**
   Navigate to `http://localhost:4321` in WebGPU or WASM-supported modern browsers such as Chrome, Edge, Firefox, or Safari.

---

## Build for Production

To create an optimized production build:

```bash
npm run build
```

To preview the production build locally:

```bash
npm run preview
```

---

## How It Works

1. **File Loading**: When an image is dropped or uploaded, an offscreen Canvas creates an `ImageData` buffer and initializes the `CanvasEngine`.
2. **AI Segmentation**:
   * The image tensor is passed to `segmentation.worker.ts`.
   * ONNX Runtime Web loads and caches the quantized `.onnx` model into IndexedDB.
   * Model inference runs on WebGPU, falling back to WebAssembly and SIMD if WebGPU is unavailable.
3. **Canvas Engine & Caching**:
   * Generated raw segmentation masks are stored in memory.
   * Toggling between **AI Photo Mode** and **Signature / Text Mode** renders instantly from cached output buffers without re-initiating ONNX session executions.
4. **Export**: The final composited Canvas state is exported directly as a lossless `.png` file.