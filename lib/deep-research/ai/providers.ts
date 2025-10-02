import { createOpenAI } from '@ai-sdk/openai';
import { getEncoding } from 'js-tiktoken';

import { RecursiveCharacterTextSplitter } from './text-splitter';

// Model Display Information
export const AI_MODEL_DISPLAY = {
  'xai/grok-4-fast-reasoning': {
    id: 'xai/grok-4-fast-reasoning',
    name: 'Grok-4 Fast Reasoning',
    logo: '/providers/openai.webp', // Using OpenAI logo as placeholder for xAI
    vision: false,
  },
} as const;

export type AIModel = keyof typeof AI_MODEL_DISPLAY;
export type AIModelDisplayInfo = (typeof AI_MODEL_DISPLAY)[AIModel];
export const availableModels = Object.values(AI_MODEL_DISPLAY);

// OpenAI Client
const openai = createOpenAI({
  apiKey: process.env.OPENAI_KEY!,
});

// Create model instances with configurations
export function createModel(modelId: AIModel, apiKey?: string) {
  // For traditional OpenAI models, use the existing approach
  const client = createOpenAI({
    apiKey: apiKey || process.env.OPENAI_KEY!,
  });

  return client(modelId, {
    structuredOutputs: true,
  });
}

// Helper function to get model for gateway or traditional models
export function getModelForUsage(modelId: AIModel, apiKey?: string) {
  // Check if this is a gateway model (format: provider/model-name)
  if (modelId.includes('/')) {
    // For gateway models, return the model string directly
    // The Vercel AI Gateway will handle the routing automatically
    return modelId;
  }

  // For traditional OpenAI models, use the existing createModel function
  return createModel(modelId, apiKey);
}

// Token handling
const MinChunkSize = 140;
const encoder = getEncoding('o200k_base');

// trim prompt to maximum context size
export function trimPrompt(prompt: string, contextSize = 120_000) {
  if (!prompt) {
    return '';
  }

  const length = encoder.encode(prompt).length;
  if (length <= contextSize) {
    return prompt;
  }

  const overflowTokens = length - contextSize;
  // on average it's 3 characters per token, so multiply by 3 to get a rough estimate of the number of characters
  const chunkSize = prompt.length - overflowTokens * 3;
  if (chunkSize < MinChunkSize) {
    return prompt.slice(0, MinChunkSize);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap: 0,
  });
  const trimmedPrompt = splitter.splitText(prompt)[0] ?? '';

  // last catch, there's a chance that the trimmed prompt is same length as the original prompt, due to how tokens are split & innerworkings of the splitter, handle this case by just doing a hard cut
  if (trimmedPrompt.length === prompt.length) {
    return trimPrompt(prompt.slice(0, chunkSize), contextSize);
  }

  // recursively trim until the prompt is within the context size
  return trimPrompt(trimmedPrompt, contextSize);
}
