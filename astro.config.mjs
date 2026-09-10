import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  // Required by @astrojs/sitemap and canonical URL generation
  site: 'https://bgremoveio.pages.dev', 
  output: 'static',
  integrations: [sitemap()],
  vite: {
    plugins: [tailwindcss()],
    build: {
      rollupOptions: {
        // Keeps WASM out of dist/ to fit Cloudflare Pages 25MB limit
        external: [/\.wasm$/],
      },
    },
    optimizeDeps: {
      exclude: ['onnxruntime-web'],
    },
  },
});