import crypto from 'crypto';
import path from 'path';
import { promises as fsp } from 'fs';
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
const UPLOADS_DIR = path.join(process.cwd(), 'uploads');

/** Local base URL the server serves /uploads from. */
function localBaseUrl(): string {
  return (
    process.env.PUBLIC_BASE_URL ||
    `http://localhost:${process.env.PORT || 3000}`
  );
}

/** Resolve a stored photo URL back to its file on disk (local uploads only). */
export function localPathForUrl(url: string): string | null {
  const marker = '/uploads/';
  const idx = url.indexOf(marker);
  if (idx === -1) return null;
  const rel = url.slice(idx + marker.length);
  // Prevent path traversal
  const abs = path.join(UPLOADS_DIR, rel);
  if (!abs.startsWith(UPLOADS_DIR)) return null;
  return abs;
}

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
    // Local disk storage: works fully offline. The photo is served by this
    // server at /uploads/... — perfect for previews and email testing.
    // (For MMS, Twilio needs a public URL — that's what R2 is for later.)
    const filePath = path.join(UPLOADS_DIR, key);
    await fsp.mkdir(path.dirname(filePath), { recursive: true });
    await fsp.writeFile(filePath, fileBuffer);
    const url = `${localBaseUrl()}/uploads/${key}`;
    console.log(`✅ Photo saved locally: ${url}`);
    return url;
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
