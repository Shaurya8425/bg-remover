import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const pages = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/pages' }),
  schema: z.object({
    title: z.string(),
    metaDescription: z.string(),
    targetKeyword: z.string(),
    heroHeading: z.string(),
    heroSubheading: z.string(),
    ogImage: z.string().optional().default('/og-image.png'),
    canonicalURL: z.string().optional(),
    defaultBgMode: z.enum(['transparent', 'color', 'blur', 'image']).default('transparent'),
    defaultColor: z.string().optional(),
    useCases: z.array(
      z.object({
        title: z.string(),
        description: z.string(),
      })
    ),
    faq: z.array(
      z.object({
        question: z.string(),
        answer: z.string(),
      })
    ),
  }),
});

export const collections = { pages };