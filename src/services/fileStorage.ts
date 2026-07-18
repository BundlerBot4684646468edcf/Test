import crypto from 'crypto';
import path from 'path';

interface StorageConfig {
  isEnabled: boolean;
  provider: 'r2' | 'mock';
}

const config: StorageConfig = {
  isEnabled: Boolean(process.env.R2_ACCOUNT_ID),
  provider: process.env.R2_ACCOUNT_ID ? 'r2' : 'mock',
};

// Mock storage for development
const mockStorage: Map<string, Buffer> = new Map();

export async function uploadFile(
  fileBuffer: Buffer,
  originalFilename: string,
  prefix: string
): Promise<string> {
  const fileHash = crypto.randomBytes(8).toString('hex');
  const ext = path.extname(originalFilename);
  const filename = `${prefix}/${fileHash}${ext}`;

  if (config.provider === 'mock') {
    console.log(`[MOCK] Storing file: ${filename}`);
    mockStorage.set(filename, fileBuffer);
    return `https://mock-storage.local/${filename}`;
  }

  // Real R2 upload would go here
  // For now, return mock URL
  return `https://mock-storage.local/${filename}`;
}

export async function validatePhotoFile(file: {
  mimetype: string;
  size: number;
}): Promise<{ valid: boolean; error?: string }> {
  const maxSize = 5 * 1024 * 1024; // 5MB
  const allowedMimes = ['image/jpeg', 'image/png', 'image/webp'];

  if (file.size > maxSize) {
    return { valid: false, error: 'File size exceeds 5MB' };
  }

  if (!allowedMimes.includes(file.mimetype)) {
    return { valid: false, error: 'Only JPEG, PNG, and WebP images are allowed' };
  }

  return { valid: true };
}

export function getMockStorageMap() {
  return mockStorage;
}
