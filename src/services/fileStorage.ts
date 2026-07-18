import crypto from 'crypto';
import path from 'path';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';

// Cloudflare R2 is S3-compatible, so we use the AWS S3 SDK against R2's endpoint.
// Everything is read lazily so .env load order never matters.
function getR2Config() {
  return {
    accountId: process.env.R2_ACCOUNT_ID || '',
    accessKeyId: process.env.R2_ACCESS_KEY_ID || '',
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY || '',
    bucket: process.env.R2_BUCKET_NAME || '',
    // Public base URL for the bucket (r2.dev URL or your custom domain),
    // e.g. https://pub-xxxx.r2.dev  — REQUIRED so Twilio can fetch the image.
    publicBaseUrl: process.env.R2_PUBLIC_BASE_URL || '',
  };
}

export function isStorageConfigured(): boolean {
  const c = getR2Config();
  return Boolean(
    c.accountId && c.accessKeyId && c.secretAccessKey && c.bucket && c.publicBaseUrl
  );
}

function getClient(accountId: string, accessKeyId: string, secretAccessKey: string) {
  return new S3Client({
    region: 'auto',
    endpoint: `https://${accountId}.r2.cloudflarestorage.com`,
    credentials: { accessKeyId, secretAccessKey },
  });
}

const CONTENT_TYPES: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
};

/**
 * Uploads a file and returns a PUBLIC https URL.
 * - With R2 fully configured: uploads to the bucket, returns the public URL
 *   (which Twilio can fetch for MMS).
 * - Without R2: stores in memory and returns a non-public placeholder URL.
 *   That URL is NOT reachable by Twilio, so photo-MMS won't work until R2 is set.
 */
const mockStorage: Map<string, Buffer> = new Map();

export async function uploadFile(
  fileBuffer: Buffer,
  originalFilename: string,
  prefix: string
): Promise<string> {
  const fileHash = crypto.randomBytes(8).toString('hex');
  const ext = path.extname(originalFilename).toLowerCase();
  const key = `${prefix}/${fileHash}${ext}`;

  const c = getR2Config();

  if (!isStorageConfigured()) {
    console.warn(
      '[STORAGE] R2 not fully configured — storing photo in memory only. ' +
        'Photo-MMS will NOT work until R2_ACCOUNT_ID/ACCESS_KEY_ID/SECRET_ACCESS_KEY/' +
        'BUCKET_NAME/PUBLIC_BASE_URL are set. Set them to enable real photo sending.'
    );
    mockStorage.set(key, fileBuffer);
    return `https://mock-storage.local/${key}`;
  }

  const client = getClient(c.accountId, c.accessKeyId, c.secretAccessKey);
  await client.send(
    new PutObjectCommand({
      Bucket: c.bucket,
      Key: key,
      Body: fileBuffer,
      ContentType: CONTENT_TYPES[ext] || 'application/octet-stream',
    })
  );

  const publicUrl = `${c.publicBaseUrl.replace(/\/$/, '')}/${key}`;
  console.log(`✅ Photo uploaded to R2: ${publicUrl}`);
  return publicUrl;
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
