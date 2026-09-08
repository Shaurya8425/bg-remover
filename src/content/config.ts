import { defineCollection, z } from 'astro:content';

const pagesCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    metaDescription: z.string(),
    heroHeading: z.string(),
    heroSubheading: z.string(),
    targetKeyword: z.string(),
    defaultBgMode: z.enum(['transparent', 'color']).default('transparent'),
    defaultColor: z.string().default('#ffffff'),
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

export const collections = {
  pages: pagesCollection,
};